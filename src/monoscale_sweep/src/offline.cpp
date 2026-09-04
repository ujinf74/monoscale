// Offline parity runner for monoscale_sweep: drives the plane-sweep occupancy
// mapper over a cached frames .npz (written by ps_points.py load_frames) and
// writes the final ternary grid as a .npy, for byte-level comparison against
// the python reference. No ROS, no rosbag, no numpy C library.
//
//   offline <frames_cache.npz> [truth | <trajectory.tum>] <out.npy>
//
// Pose handling mirrors ps_points.py run() exactly:
//   - truth poses are re-anchored to the first truth sample (x0, y0, yaw0),
//     roll/pitch kept in the world frame (ps_points.py ~1509-1522);
//   - a TUM trajectory is used as-is (its frame already starts at its own
//     first pose), yaw = 2*atan2(qz, qw), roll = pitch = 0;
//   - each image frame takes the pose at np.searchsorted(stamps, time)
//     clipped to the table -- nearest-right assignment, NOT interpolation
//     (ps_points.py ~1582-1583) -- because parity demands the same indices.

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "monoscale_sweep/sweep.hpp"

namespace
{

// ---------------------------------------------------------------------------
// Camera constants, hardcoded from
// /home/i/ros2_ws/hero-release/src/monoscale_odometry/config/vision_fisheye.param.yaml
// (camera blocks `front` and `rear`; distortion_model: equidistant, d all
// zero, calibration_width 2560). The runner scales the intrinsics to the
// cached frame width, exactly as ps_points.py does (k * width_px / 2560).
// ---------------------------------------------------------------------------
constexpr double kCalibrationWidth = 2560.0;
// k is row-major 3x3; front and rear share it.
constexpr double kIntrinsics[9] = {
  1051.81, 0.0, 1279.5,
  0.0, 1051.81, 719.5,
  0.0, 0.0, 1.0};
constexpr double kFrontRotation[9] = {
  0.0, -0.5, 0.8660254,
  -1.0, 0.0, 0.0,
  0.0, -0.8660254, -0.5};
constexpr double kFrontTranslation[3] = {3.694, 0.0, 0.89};
constexpr double kRearRotation[9] = {
  0.0, 0.5, -0.8660254,
  1.0, 0.0, 0.0,
  0.0, -0.8660254, -0.5};
constexpr double kRearTranslation[3] = {-0.82, 0.0, 1.26};

monoscale_sweep::Lens make_lens(
  const double rotation[9], const double translation[3], double width_px)
{
  const double ratio = width_px / kCalibrationWidth;
  monoscale_sweep::Lens lens;
  lens.focal = kIntrinsics[0] * ratio;
  lens.cx = kIntrinsics[2] * ratio;
  lens.cy = kIntrinsics[5] * ratio;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      lens.rotation_base_from_camera(row, col) = rotation[3 * row + col];
    }
    lens.translation_base_from_camera(row) = translation[row];
  }
  return lens;
}

// ---------------------------------------------------------------------------
// Minimal .npz / .npy reading.
//
// A .npz is a plain ZIP archive of .npy members. np.savez writes it with
// zipfile in STORED mode (no compression) but *streams* each member: the
// local file header may carry zeroes for the sizes (general-purpose flag bit
// 3, sizes deferred to a trailing data descriptor). The central directory at
// the end of the archive always has the real sizes, so the reader works from
// there: find the End Of Central Directory record, walk the central entries
// for name / method / compressed size / local-header offset, then for each
// wanted member seek to its local header just to learn where the data starts
// (local filename/extra lengths can differ from the central ones).
// Zip64 is handled for the fields that can overflow: 0xFFFFFFFF sizes or
// offsets are resolved from the 0x0001 extra field, and a 0xFFFFFFFF central
// directory offset from the Zip64 EOCD via its locator.
// ---------------------------------------------------------------------------
std::uint16_t read_u16(const std::uint8_t * p) {return static_cast<std::uint16_t>(p[0] | (p[1] << 8));}
std::uint32_t read_u32(const std::uint8_t * p)
{
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint64_t read_u64(const std::uint8_t * p)
{
  return static_cast<std::uint64_t>(read_u32(p)) |
         (static_cast<std::uint64_t>(read_u32(p + 4)) << 32);
}

struct NpyArray
{
  std::string descr;                // e.g. "<f8", "|u1"
  std::vector<std::size_t> shape;   // () for 0-d scalars
  std::vector<std::uint8_t> data;   // raw little-endian buffer

  std::size_t count() const
  {
    std::size_t n = 1;
    for (std::size_t axis : shape) {n *= axis;}
    return n;
  }
  const double * f64() const
  {
    if (descr != "<f8") {throw std::runtime_error("expected <f8, got " + descr);}
    return reinterpret_cast<const double *>(data.data());
  }
  const std::uint8_t * u8() const
  {
    if (descr != "|u1") {throw std::runtime_error("expected |u1, got " + descr);}
    return data.data();
  }
};

// Parse one .npy member: magic \x93NUMPY, version, header length (uint16 at
// offset 8 for v1.x, uint32 for v2/v3), then an ASCII dict literal like
//   {'descr': '<f8', 'fortran_order': False, 'shape': (162, 720, 1280), }
// space-padded to alignment, then the raw data.
NpyArray parse_npy(std::vector<std::uint8_t> raw, const std::string & name)
{
  if (raw.size() < 10 || std::memcmp(raw.data(), "\x93NUMPY", 6) != 0) {
    throw std::runtime_error(name + ": not a .npy member");
  }
  const int major = raw[6];
  std::size_t header_len = 0;
  std::size_t header_start = 0;
  if (major == 1) {
    header_len = read_u16(raw.data() + 8);
    header_start = 10;
  } else {
    header_len = read_u32(raw.data() + 8);
    header_start = 12;
  }
  if (raw.size() < header_start + header_len) {
    throw std::runtime_error(name + ": truncated .npy header");
  }
  const std::string header(
    raw.begin() + header_start, raw.begin() + header_start + header_len);

  NpyArray out;
  // 'descr': quoted string.
  std::size_t at = header.find("'descr':");
  if (at == std::string::npos) {throw std::runtime_error(name + ": no descr");}
  at = header.find('\'', at + 8);
  const std::size_t end = header.find('\'', at + 1);
  out.descr = header.substr(at + 1, end - at - 1);
  // 'fortran_order': must be False -- everything savez writes here is C order.
  if (header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error(name + ": fortran_order is not False");
  }
  // 'shape': tuple of ints; "()" is a 0-d scalar, "(5,)" a vector.
  at = header.find("'shape':");
  if (at == std::string::npos) {throw std::runtime_error(name + ": no shape");}
  at = header.find('(', at);
  const std::size_t close = header.find(')', at);
  std::size_t cursor = at + 1;
  while (cursor < close) {
    if (std::isdigit(static_cast<unsigned char>(header[cursor]))) {
      std::size_t used = 0;
      out.shape.push_back(std::stoull(header.substr(cursor, close - cursor), &used));
      cursor += used;
    } else {
      ++cursor;
    }
  }

  // Element size from the descr's trailing digits ("<f8" -> 8, "|u1" -> 1).
  std::size_t digits = out.descr.size();
  while (digits > 0 && std::isdigit(static_cast<unsigned char>(out.descr[digits - 1]))) {
    --digits;
  }
  const std::size_t item = std::stoull(out.descr.substr(digits));
  const std::size_t expected = out.count() * item;
  if (raw.size() - header_start - header_len < expected) {
    throw std::runtime_error(name + ": .npy data shorter than shape demands");
  }
  out.data.assign(
    raw.begin() + header_start + header_len,
    raw.begin() + header_start + header_len + expected);
  return out;
}

class NpzFile
{
public:
  explicit NpzFile(const std::string & path)
  : file_(path, std::ios::binary)
  {
    if (!file_) {throw std::runtime_error("cannot open " + path);}
    file_.seekg(0, std::ios::end);
    const std::uint64_t size = static_cast<std::uint64_t>(file_.tellg());

    // End Of Central Directory: signature PK\x05\x06 within the last
    // 65557 bytes (22-byte record + up to 64 KiB comment).
    const std::uint64_t tail_len = std::min<std::uint64_t>(size, 65557);
    std::vector<std::uint8_t> tail(tail_len);
    file_.seekg(size - tail_len);
    file_.read(reinterpret_cast<char *>(tail.data()), tail_len);
    std::size_t eocd = tail_len;
    for (std::size_t i = tail_len >= 22 ? tail_len - 22 + 1 : 0; i-- > 0;) {
      if (read_u32(tail.data() + i) == 0x06054b50) {eocd = i; break;}
    }
    if (eocd == tail_len) {throw std::runtime_error(path + ": no ZIP end record");}

    std::uint64_t entries = read_u16(tail.data() + eocd + 10);
    std::uint64_t directory_size = read_u32(tail.data() + eocd + 12);
    std::uint64_t directory_offset = read_u32(tail.data() + eocd + 16);
    if (entries == 0xFFFF || directory_offset == 0xFFFFFFFFu) {
      // Zip64: locator sits 20 bytes before the EOCD and points at the
      // Zip64 EOCD record, which carries the 64-bit counts and offset.
      const std::uint64_t locator = size - tail_len + eocd - 20;
      std::uint8_t loc[20];
      file_.seekg(locator);
      file_.read(reinterpret_cast<char *>(loc), 20);
      if (read_u32(loc) != 0x07064b50) {
        throw std::runtime_error(path + ": Zip64 locator missing");
      }
      std::uint8_t rec[56];
      file_.seekg(read_u64(loc + 8));
      file_.read(reinterpret_cast<char *>(rec), 56);
      if (read_u32(rec) != 0x06064b50) {
        throw std::runtime_error(path + ": Zip64 end record missing");
      }
      entries = read_u64(rec + 32);
      directory_size = read_u64(rec + 40);
      directory_offset = read_u64(rec + 48);
    }

    std::vector<std::uint8_t> directory(directory_size);
    file_.seekg(directory_offset);
    file_.read(reinterpret_cast<char *>(directory.data()), directory_size);

    std::size_t at = 0;
    for (std::uint64_t entry = 0; entry < entries; ++entry) {
      if (at + 46 > directory.size() || read_u32(directory.data() + at) != 0x02014b50) {
        throw std::runtime_error(path + ": malformed central directory");
      }
      const std::uint8_t * p = directory.data() + at;
      const std::uint16_t method = read_u16(p + 10);
      std::uint64_t compressed = read_u32(p + 20);
      std::uint64_t uncompressed = read_u32(p + 24);
      const std::uint16_t name_len = read_u16(p + 28);
      const std::uint16_t extra_len = read_u16(p + 30);
      const std::uint16_t comment_len = read_u16(p + 32);
      std::uint64_t local_offset = read_u32(p + 42);
      std::string name(reinterpret_cast<const char *>(p + 46), name_len);
      // Zip64 extended info (id 0x0001): only the overflowed fields are
      // present, in the fixed order uncompressed, compressed, offset.
      std::size_t extra_at = at + 46 + name_len;
      const std::size_t extra_end = extra_at + extra_len;
      while (extra_at + 4 <= extra_end) {
        const std::uint16_t id = read_u16(directory.data() + extra_at);
        const std::uint16_t len = read_u16(directory.data() + extra_at + 2);
        if (id == 0x0001) {
          const std::uint8_t * field = directory.data() + extra_at + 4;
          if (uncompressed == 0xFFFFFFFFu) {uncompressed = read_u64(field); field += 8;}
          if (compressed == 0xFFFFFFFFu) {compressed = read_u64(field); field += 8;}
          if (local_offset == 0xFFFFFFFFu) {local_offset = read_u64(field);}
        }
        extra_at += 4 + len;
      }
      if (method != 0) {
        throw std::runtime_error(
          path + ": member " + name + " uses compression method " +
          std::to_string(method) + "; only STORED (0) is supported -- was this "
          "written with np.savez_compressed instead of np.savez?");
      }
      members_[name] = Member{local_offset, compressed};
      at += 46 + name_len + extra_len + comment_len;
      (void)uncompressed;  // for STORED it equals compressed
    }
  }

  NpyArray load(const std::string & array_name)
  {
    const std::string member = array_name + ".npy";
    const auto found = members_.find(member);
    if (found == members_.end()) {
      throw std::runtime_error("cache has no array " + array_name);
    }
    // The local header's own filename/extra lengths locate the data; sizes
    // there may be zero (streamed member), so the central size is used.
    std::uint8_t local[30];
    file_.seekg(found->second.offset);
    file_.read(reinterpret_cast<char *>(local), 30);
    if (read_u32(local) != 0x04034b50) {
      throw std::runtime_error(member + ": bad local file header");
    }
    const std::uint16_t name_len = read_u16(local + 26);
    const std::uint16_t extra_len = read_u16(local + 28);
    std::vector<std::uint8_t> raw(found->second.size);
    file_.seekg(found->second.offset + 30 + name_len + extra_len);
    file_.read(reinterpret_cast<char *>(raw.data()), raw.size());
    if (!file_) {throw std::runtime_error(member + ": truncated member data");}
    return parse_npy(std::move(raw), member);
  }

private:
  struct Member
  {
    std::uint64_t offset = 0;  // of the local file header
    std::uint64_t size = 0;    // compressed == uncompressed for STORED
  };
  std::ifstream file_;
  std::map<std::string, Member> members_;
};

// Minimal .npy writer, the inverse of parse_npy, for the int8 grid.
void write_npy_int8(const std::string & path, const cv::Mat & grid)
{
  std::ostringstream dict;
  dict << "{'descr': '|i1', 'fortran_order': False, 'shape': ("
       << grid.rows << ", " << grid.cols << "), }";
  std::string header = dict.str();
  const std::size_t unpadded = 10 + header.size() + 1;  // magic+version+len+dict+\n
  header.append((unpadded + 63) / 64 * 64 - unpadded, ' ');
  header.push_back('\n');

  std::ofstream out(path, std::ios::binary);
  if (!out) {throw std::runtime_error("cannot write " + path);}
  out.write("\x93NUMPY\x01\x00", 8);
  const std::uint16_t header_len = static_cast<std::uint16_t>(header.size());
  const std::uint8_t len_bytes[2] = {
    static_cast<std::uint8_t>(header_len & 0xFF),
    static_cast<std::uint8_t>(header_len >> 8)};
  out.write(reinterpret_cast<const char *>(len_bytes), 2);
  out.write(header.data(), header.size());
  for (int row = 0; row < grid.rows; ++row) {
    out.write(reinterpret_cast<const char *>(grid.ptr<std::int8_t>(row)), grid.cols);
  }
}

// ---------------------------------------------------------------------------
// Poses.
// ---------------------------------------------------------------------------
struct PoseTable
{
  std::vector<double> stamps;
  std::vector<monoscale_sweep::Pose5> poses;
};

// Truth rows are (stamp, x, y, yaw, roll, pitch); re-anchor to the first
// sample the way ps_points.py does: rotate x/y about the first yaw, subtract
// the first yaw, keep roll/pitch in the world frame.
PoseTable truth_poses(const NpyArray & truth)
{
  if (truth.shape.size() != 2 || truth.shape[1] < 6 || truth.shape[0] == 0) {
    throw std::runtime_error("truth array is not M x 6");
  }
  const std::size_t rows = truth.shape[0];
  const std::size_t cols = truth.shape[1];
  const double * value = truth.f64();
  const double x0 = value[1];
  const double y0 = value[2];
  const double yaw0 = value[3];
  const double c0 = std::cos(-yaw0);
  const double s0 = std::sin(-yaw0);
  PoseTable table;
  table.stamps.resize(rows);
  table.poses.resize(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    const double * v = value + row * cols;
    table.stamps[row] = v[0];
    const double dx = v[1] - x0;
    const double dy = v[2] - y0;
    table.poses[row].x = c0 * dx - s0 * dy;
    table.poses[row].y = s0 * dx + c0 * dy;
    table.poses[row].yaw = v[3] - yaw0;
    table.poses[row].roll = v[4];
    table.poses[row].pitch = v[5];
  }
  return table;
}

// TUM rows are (stamp x y z qx qy qz qw); the file's frame already starts at
// its own first pose, so nothing is re-anchored (ps_points.py tum: branch).
PoseTable tum_poses(const std::string & path)
{
  std::ifstream file(path);
  if (!file) {throw std::runtime_error("cannot open " + path);}
  PoseTable table;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {continue;}
    std::istringstream row(line);
    double stamp, x, y, z, qx, qy, qz, qw;
    if (!(row >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {continue;}
    monoscale_sweep::Pose5 pose;
    pose.x = x;
    pose.y = y;
    pose.yaw = 2.0 * std::atan2(qz, qw);
    table.stamps.push_back(stamp);
    table.poses.push_back(pose);
  }
  if (table.poses.empty()) {throw std::runtime_error(path + ": no TUM rows");}
  return table;
}

// ---------------------------------------------------------------------------
// One camera's keyframe loop, the parity target being the per-camera loop of
// ps_points.py run(): searchsorted pose assignment, cumulative travel,
// travel-offset source selection (--baseline-select off), keyframe cadence.
// ---------------------------------------------------------------------------
std::size_t run_camera(
  const monoscale_sweep::Sweep & sweep, const NpyArray & grays,
  const NpyArray & stamps, const PoseTable & table,
  monoscale_sweep::CameraGrid & grid)
{
  const std::size_t frames = stamps.count();
  if (grays.shape.size() != 3 || grays.shape[0] != frames) {
    throw std::runtime_error("gray stack and stamps disagree");
  }
  if (frames < 4) {return 0;}  // ps_points.py: len(times) < 4 -> skip camera
  const int height = static_cast<int>(grays.shape[1]);
  const int width = static_cast<int>(grays.shape[2]);
  const std::size_t pixels = static_cast<std::size_t>(height) * width;
  const double * times = stamps.f64();

  // frame_poses = local[searchsorted(stamps, times).clip(0, len - 1)]
  std::vector<monoscale_sweep::Pose5> frame_poses(frames);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    std::size_t at = std::lower_bound(
      table.stamps.begin(), table.stamps.end(), times[frame]) - table.stamps.begin();
    if (at >= table.stamps.size()) {at = table.stamps.size() - 1;}
    frame_poses[frame] = table.poses[at];
  }

  std::vector<double> travelled(frames, 0.0);
  for (std::size_t frame = 1; frame < frames; ++frame) {
    travelled[frame] = travelled[frame - 1] + std::hypot(
      frame_poses[frame].x - frame_poses[frame - 1].x,
      frame_poses[frame].y - frame_poses[frame - 1].y);
  }

  const monoscale_sweep::SweepSettings & settings = sweep.settings();
  const std::uint8_t * stack = grays.u8();
  auto slice = [&](std::size_t frame) {
    return cv::Mat(
      height, width, CV_8U,
      const_cast<std::uint8_t *>(stack + frame * pixels));
  };

  std::size_t keyframes = 0;
  double next_at = 0.0;
  for (std::size_t index = 0; index < frames; ++index) {
    if (travelled[index] < next_at) {continue;}
    std::vector<std::size_t> chosen;
    for (double offset : settings.source_offsets) {
      const double want = travelled[index] + offset;
      if (want < 0.0 || want > travelled.back()) {continue;}
      std::size_t other = 0;
      double gap = std::abs(travelled[0] - want);
      for (std::size_t candidate = 1; candidate < frames; ++candidate) {
        const double error = std::abs(travelled[candidate] - want);
        if (error < gap) {gap = error; other = candidate;}
      }
      if (gap < settings.source_tolerance && other != index) {
        chosen.push_back(other);
      }
    }
    if (chosen.size() < 2) {continue;}
    next_at = travelled[index] + settings.keyframe_travel;
    ++keyframes;

    std::vector<cv::Mat> source_grays;
    std::vector<monoscale_sweep::Pose5> source_poses;
    for (std::size_t other : chosen) {
      source_grays.push_back(slice(other));
      source_poses.push_back(frame_poses[other]);
    }
    sweep.keyframe(slice(index), frame_poses[index], source_grays, source_poses, grid);
    if (keyframes % 20 == 0) {
      std::cerr << "  keyframe " << keyframes << " @ frame " << index
                << "/" << frames << std::endl;
    }
  }
  return keyframes;
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    std::string cache_path, pose_source, out_path;
    if (argc == 4) {
      cache_path = argv[1];
      pose_source = argv[2];
      out_path = argv[3];
    } else if (argc == 3) {
      cache_path = argv[1];
      pose_source = "truth";
      out_path = argv[2];
    } else {
      std::cerr << "usage: " << argv[0]
                << " <frames_cache.npz> [truth | <trajectory.tum>] <out.npy>\n";
      return 2;
    }

    NpzFile cache(cache_path);
    const NpyArray front_grays = cache.load("front_grays");
    const NpyArray front_stamps = cache.load("front_stamps");
    const NpyArray rear_grays = cache.load("rear_grays");
    const NpyArray rear_stamps = cache.load("rear_stamps");

    const PoseTable table = pose_source == "truth"
      ? truth_poses(cache.load("truth"))
      : tum_poses(pose_source);

    const monoscale_sweep::SweepSettings settings;  // defaults ARE the operating point

    const double front_width =
      front_grays.shape[0] ? static_cast<double>(front_grays.shape[2]) : 1280.0;
    const double rear_width =
      rear_grays.shape[0] ? static_cast<double>(rear_grays.shape[2]) : 1280.0;
    const monoscale_sweep::Sweep front_sweep(
      settings, make_lens(kFrontRotation, kFrontTranslation, front_width));
    const monoscale_sweep::Sweep rear_sweep(
      settings, make_lens(kRearRotation, kRearTranslation, rear_width));

    monoscale_sweep::CameraGrid front_grid;
    monoscale_sweep::CameraGrid rear_grid;
    front_grid.reset(settings);
    rear_grid.reset(settings);

    std::size_t keyframes = 0;
    keyframes += run_camera(front_sweep, front_grays, front_stamps, table, front_grid);
    keyframes += run_camera(rear_sweep, rear_grays, rear_stamps, table, rear_grid);

    cv::Mat map = monoscale_sweep::publish(settings, {&front_grid, &rear_grid});
    if (map.type() != CV_8S) {
      cv::Mat converted;
      map.convertTo(converted, CV_8S);
      map = converted;
    }
    std::size_t free_cells = 0;
    std::size_t occupied_cells = 0;
    for (int row = 0; row < map.rows; ++row) {
      const std::int8_t * cell = map.ptr<std::int8_t>(row);
      for (int col = 0; col < map.cols; ++col) {
        if (cell[col] == 0) {++free_cells;}
        if (cell[col] == 100) {++occupied_cells;}
      }
    }
    write_npy_int8(out_path, map);
    std::cout << "keyframes=" << keyframes << " free=" << free_cells
              << " occupied=" << occupied_cells << std::endl;
  } catch (const std::exception & error) {
    std::cerr << "offline: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
