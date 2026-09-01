// Ground feature tracking, moved out of Python.
//
// The estimator's geometry was never the bottleneck: a single optical flow
// call over 600 points at 640x360 measures 2.45 ms, while the same hop costs
// about 12 ms once it has been through rclpy and numpy. That 9.5 ms of
// overhead is what stops the vehicle above roughly two metres a second, and no
// amount of tuning inside Python moved it -- making feature detection 54 times
// cheaper changed the throughput not at all.
//
// This node does only the part that has to keep up: take images, carry the
// tracked set forward, and publish where each feature went. Everything that
// gives the answer its meaning -- ground projection, the anchor map, the
// registration -- stays where it is.
//
// Published layout, one message per camera, flat float64:
//   [stamp_sec, count, width, height, id0, prev_x0, prev_y0, cur_x0, cur_y0, ...]
//
// float64, not float32. Two things in here are integers that do not survive a
// 24 bit mantissa. A Unix timestamp of 1.79e9 quantises to about 106 seconds,
// which collapses every frame of a live run onto the same instant and takes
// dt, the frame pairing and the IMU interpolation with it -- invisible offline
// only because the bags carry simulation stamps of a few tens of seconds.
// Feature identities are the other: at road speed this tracker mints half a
// million of them a minute, and past 2^24 consecutive integers land on the
// same float, which the identity matcher assumes cannot happen. A 53 bit
// mantissa holds both with room to spare, at twice the bytes on a link that
// carries 2.4 MB/s.
// Identities persist for as long as a feature is followed, which is what lets
// the estimator match a sighting against the anchor it built earlier.
//
// The frame size is in the message because the pixels are measured in the
// downscaled frame, and the estimator has to scale the intrinsics to match.
// Leaving it to both sides reading the same processing_width parameter makes
// a silent geometry error out of a configuration mismatch.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#ifdef MONOSCALE_TRACKER_HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace
{

// An Image as a single channel Mat, without cv_bridge.
//
// cv_bridge links against the OpenCV the ROS distribution was built with, and
// this node may be pointed at another one that has the CUDA modules. Two
// OpenCVs in one process is an ABI gamble; the conversion it was doing for us
// is this.
bool to_gray(const sensor_msgs::msg::Image & message, cv::Mat & gray)
{
  const std::string & encoding = message.encoding;
  int channels = 0;
  int code = -1;
  if (encoding == "mono8" || encoding == "8UC1") {
    channels = 1;
  } else if (encoding == "bgr8" || encoding == "8UC3") {
    channels = 3;
    code = cv::COLOR_BGR2GRAY;
  } else if (encoding == "rgb8") {
    channels = 3;
    code = cv::COLOR_RGB2GRAY;
  } else if (encoding == "bgra8" || encoding == "8UC4") {
    channels = 4;
    code = cv::COLOR_BGRA2GRAY;
  } else if (encoding == "rgba8") {
    channels = 4;
    code = cv::COLOR_RGBA2GRAY;
  } else {
    return false;
  }

  const int rows = static_cast<int>(message.height);
  const int columns = static_cast<int>(message.width);
  if (rows <= 0 || columns <= 0 ||
    message.data.size() < static_cast<size_t>(rows) * message.step)
  {
    return false;
  }
  // A view over the caller's buffer, honouring the row stride.
  const cv::Mat view(
    rows, columns, CV_8UC(channels), const_cast<uint8_t *>(message.data.data()),
    message.step);
  if (channels == 1) {
    view.copyTo(gray);
  } else {
    cv::cvtColor(view, gray, code);
  }
  return true;
}

#ifdef MONOSCALE_TRACKER_HAS_CUDA
// A page-locked buffer of at least `count` entries, handed back as a header
// over exactly that many. Grown when it has to be and never shrunk, so a
// steady feature count pays the allocation once and the rest of the drive
// reuses it. Page-locked because a pageable transfer is staged by the driver
// anyway and cannot overlap with a kernel; these are what let the copies ride
// the stream instead of stalling it.
cv::Mat pinned(cv::cuda::HostMem & memory, int count, int type)
{
  if (memory.empty() || memory.cols < count || memory.type() != type) {
    memory.create(1, std::max(count, 1), type);
  }
  return memory.createMatHeader().colRange(0, count);
}
#endif

// The ground under the camera, as a plane the vehicle's own motion moves.
//
// A plane's image motion between two views is exactly a homography, and for a
// camera looking at the road it does not have to be fitted from correspondences
// -- every term is known. For a plane n^T X = d and a camera motion (R, t),
//
//   b2 ~ (R + t n^T / d) b1
//
// so with the mount, the camera height and the vehicle's step the prediction is
// computed rather than estimated. Nothing it produces can be poisoned by
// features that failed to track, which is the whole point: fitting a predictor
// to a flow that is already half stationary teaches it to predict stationary.
struct GroundModel
{
  bool ready = false;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  int calibration_width = 0;
  int calibration_height = 0;
  bool equidistant = true;
  cv::Matx33d rotation_base_from_camera = cv::Matx33d::eye();
  cv::Vec3d translation_base_from_camera{0.0, 0.0, 0.0};

  // Intrinsics scale with the frame the tracker actually processes.
  void scaled(int width, int height, double & sfx, double & sfy,
    double & scx, double & scy) const
  {
    const double sx = calibration_width > 0
      ? static_cast<double>(width) / calibration_width : 1.0;
    const double sy = calibration_height > 0
      ? static_cast<double>(height) / calibration_height : 1.0;
    sfx = fx * sx; sfy = fy * sy; scx = cx * sx; scy = cy * sy;
  }

  cv::Vec3d bearing(const cv::Point2f & p, int width, int height) const
  {
    double sfx, sfy, scx, scy;
    scaled(width, height, sfx, sfy, scx, scy);
    const double xn = (p.x - scx) / sfx;
    const double yn = (p.y - scy) / sfy;
    const double theta = std::hypot(xn, yn);
    if (!equidistant) {
      const double n = std::sqrt(xn*xn + yn*yn + 1.0);
      return cv::Vec3d(xn / n, yn / n, 1.0 / n);
    }
    const double s = theta > 1e-9 ? std::sin(theta) / theta : 1.0;
    return cv::Vec3d(xn * s, yn * s, std::cos(theta));
  }

  cv::Point2f pixel(const cv::Vec3d & b, int width, int height) const
  {
    double sfx, sfy, scx, scy;
    scaled(width, height, sfx, sfy, scx, scy);
    const double r = std::hypot(b[0], b[1]);
    double k = 1.0;
    if (equidistant) {
      const double theta = std::atan2(r, b[2]);
      k = r > 1e-12 ? theta / r : 1.0;
    } else {
      k = std::abs(b[2]) > 1e-12 ? 1.0 / b[2] : 0.0;
    }
    return cv::Point2f(
      static_cast<float>(b[0] * k * sfx + scx),
      static_cast<float>(b[1] * k * sfy + scy));
  }

  // H for a forward step and a turn, both in the body frame.
  //
  // `pitch` and `roll` free the body's tilt over the same hop, and default to
  // zero so every caller that only knows the turn is untouched. The order is
  // Rz(turn) Ry(pitch) Rx(roll) -- the same order `AttitudeFilter::body_tilt`
  // builds its matrix in, so a pitch measured here means what a pitch measured
  // there means: nose-down positive, right-side-down positive.
  //
  // The tilt enters the rotation and not the translation, which is a choice.
  // Rotating about base_link the way the turn does would add a lever of
  // h dtheta -- 0.4 mm at 0.3 mrad, a tenth of a per cent of an 8 m/s step, and
  // it would land straight on top of the one quantity this family is being
  // asked to measure. That pivot is also known to be the wrong one: the body
  // pitches about its centre of mass, not the rear axle, and pivoting at
  // base_link was measured to inflate the front camera's hop 2-3x. Asserting no
  // pivot leaves whatever translation the tilt really carries to be absorbed by
  // `step`, which is the honest place for something the image cannot separate.
  cv::Matx33d homography(
    double step, double turn, double pitch = 0.0, double roll = 0.0,
    bool arc_hop = false) const
  {
    const cv::Matx33d r_cb = rotation_base_from_camera.t();
    const double c = std::cos(turn);
    const double s = std::sin(turn);
    const cv::Matx33d r_b(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
    const cv::Matx33d rot = r_cb * r_b.t() * rotation_base_from_camera;
    // The hop, along the arc the body actually follows rather than along the
    // chord.
    //
    // A vehicle turning by `turn` while advancing `step` does not translate
    // straight ahead and then rotate: its displacement in the starting body
    // frame is `(step/turn) (sin turn, 1 - cos turn)`, i.e. `step (1 - turn^2/6,
    // turn/2)` to second order. The lateral half-step was missing, and this is
    // the one geometry where that matters: the patch sits about four metres out
    // where a yaw and a sideways slide look the same, so the omitted `step
    // turn / 2` is read back as yaw. Predicted 0.0017 deg on curve_s20, where
    // the measured per-hop yaw error is 0.00046 deg and **84% of its variance
    // regresses on speed** rather than on the turn -- which is the signature of
    // a term in `step * turn`, since the yaw rate there is nearly constant.
    //
    // At `turn` = 0 both factors reduce to (1, 0) exactly, so a straight hop is
    // the hop this always used and `road_step_arc` off is bit for bit the old
    // path.
    double along = step;
    double across = 0.0;
    if (arc_hop && std::abs(turn) > 1e-9) {
      // Haversine rather than `(1 - cos turn) / turn`: near zero that
      // difference is two nearly equal doubles and loses about twelve digits,
      // and this fit is measurably sensitive to perturbations at 1e-13.
      // `2 sin^2(turn/2) / turn` is the same number with no cancellation.
      const double half = std::sin(0.5 * turn);
      along = step * s / turn;
      across = step * 2.0 * half * half / turn;
    }
    const cv::Vec3d hop(along, across, 0.0);
    const cv::Vec3d t =
      r_cb * (r_b.t() * (translation_base_from_camera - hop) -
      translation_base_from_camera);
    const cv::Vec3d n = r_cb * cv::Vec3d(0.0, 0.0, 1.0);
    const double height = translation_base_from_camera[2];
    cv::Matx33d out = rot;
    if (std::abs(height) > 1e-6) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          out(i, j) -= t[i] * n[j] / height;
        }
      }
    }
    if (pitch == 0.0 && roll == 0.0) {
      // Returned before the tilt is applied rather than through an identity
      // matrix, because r_cb * rotation_base_from_camera is an identity only to
      // rounding and the deployed path must come out bit for bit as it did.
      return out;
    }
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    const double cr = std::cos(roll);
    const double sr = std::sin(roll);
    const cv::Matx33d tilt(
      cp, sp * sr, sp * cr,
      0.0, cr, -sr,
      -sp, cp * sr, cp * cr);
    // Applied on the left, in the second view's camera frame: the plane's
    // normal belongs to the first view and is untouched by where the body ends
    // up. What multiplies through to the translation is A t, which is the step
    // seen from the tilted camera -- a rigid motion, just one whose rotation
    // pivots at the lens.
    return r_cb * tilt.t() * rotation_base_from_camera * out;
  }

  // The four derivatives of that matrix, at the point it was built.
  //
  // The fit's Jacobian is otherwise made of warps: two per freed parameter per
  // rebuild, and a warp at the lattice the precision needs is the most
  // expensive thing this file does -- 55% of the fit's warps go into building
  // the Jacobian rather than into descending. These are the same quantity
  // written out, so the descent keeps only the warps it actually steps with.
  //
  // Order matches `solve_step_esm`: step, turn, pitch, roll.
  void homography_jacobian(
    double step, double turn, double pitch, double roll,
    cv::Matx33d out[4]) const
  {
    const cv::Matx33d r_cb = rotation_base_from_camera.t();
    const double c = std::cos(turn);
    const double s = std::sin(turn);
    const cv::Matx33d r_b(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
    // d(r_b^T)/d(turn): r_b^T is the rotation by -turn, so its derivative is
    // the rotation by -turn pre-multiplied by the generator about z, negated.
    const cv::Matx33d d_rbt(-s, c, 0.0, -c, -s, 0.0, 0.0, 0.0, 0.0);
    const cv::Vec3d hop(step, 0.0, 0.0);
    const cv::Vec3d n = r_cb * cv::Vec3d(0.0, 0.0, 1.0);
    const double height = translation_base_from_camera[2];
    const double scale = std::abs(height) > 1e-6 ? 1.0 / height : 0.0;
    const auto outer = [&](const cv::Vec3d & t) {
        cv::Matx33d m = cv::Matx33d::zeros();
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            m(i, j) = t[i] * n[j] * scale;
          }
        }
        return m;
      };
    // step enters only through the translation, and only as -e_x.
    const cv::Vec3d dt_step = r_cb * (r_b.t() * cv::Vec3d(-1.0, 0.0, 0.0));
    cv::Matx33d d_step = -outer(dt_step);
    // turn enters the rotation and the translation together.
    const cv::Matx33d d_rot_turn = r_cb * d_rbt * rotation_base_from_camera;
    const cv::Vec3d dt_turn = r_cb * (d_rbt * (translation_base_from_camera - hop));
    cv::Matx33d d_turn = d_rot_turn - outer(dt_turn);

    const cv::Matx33d base = homography(step, turn);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    const double cr = std::cos(roll);
    const double sr = std::sin(roll);
    const cv::Matx33d tilt(
      cp, sp * sr, sp * cr,
      0.0, cr, -sr,
      -sp, cp * sr, cp * cr);
    const cv::Matx33d d_tilt_pitch(
      -sp, cp * sr, cp * cr,
      0.0, 0.0, 0.0,
      -cp, -sp * sr, -sp * cr);
    const cv::Matx33d d_tilt_roll(
      0.0, sp * cr, -sp * sr,
      0.0, -sr, -cr,
      0.0, cp * cr, -cp * sr);
    // The tilt multiplies on the left of the untilted matrix, so the step and
    // turn derivatives carry it and the tilt derivatives carry the base.
    const cv::Matx33d left = r_cb * tilt.t() * rotation_base_from_camera;
    out[0] = left * d_step;
    out[1] = left * d_turn;
    out[2] = r_cb * d_tilt_pitch.t() * rotation_base_from_camera * base;
    out[3] = r_cb * d_tilt_roll.t() * rotation_base_from_camera * base;
  }

  // How a pixel moves when the bearing it came from moves. Rows are (x, y),
  // columns the three components of the bearing.
  void pixel_jacobian(
    const cv::Vec3d & b, int width, int height, double out[2][3]) const
  {
    double sfx, sfy, scx, scy;
    scaled(width, height, sfx, sfy, scx, scy);
    const double r2 = b[0] * b[0] + b[1] * b[1];
    const double r = std::sqrt(r2);
    if (!equidistant) {
      const double z = std::abs(b[2]) > 1e-12 ? b[2] : 1e-12;
      const double iz = 1.0 / z;
      out[0][0] = sfx * iz;  out[0][1] = 0.0;      out[0][2] = -sfx * b[0] * iz * iz;
      out[1][0] = 0.0;       out[1][1] = sfy * iz; out[1][2] = -sfy * b[1] * iz * iz;
      return;
    }
    // k = atan2(r, z) / r, and the pixel is (b.x, b.y) * k. Both the radial
    // part of k and the direction change with the bearing.
    const double n2 = r2 + b[2] * b[2];
    const double theta = std::atan2(r, b[2]);
    const double k = r > 1e-12 ? theta / r : 1.0;
    // dtheta/db = ( z*bx, z*by, -r^2 ) / (r * n2)
    const double dtheta_x = r > 1e-12 ? b[2] * b[0] / (r * n2) : 0.0;
    const double dtheta_y = r > 1e-12 ? b[2] * b[1] / (r * n2) : 0.0;
    const double dtheta_z = -r / n2;
    // dk/db = (dtheta/db)/r - theta * (dr/db)/r^2, with dr/db = (bx, by, 0)/r
    const double dk_x = r > 1e-12 ? dtheta_x / r - theta * b[0] / (r2 * r) : 0.0;
    const double dk_y = r > 1e-12 ? dtheta_y / r - theta * b[1] / (r2 * r) : 0.0;
    const double dk_z = r > 1e-12 ? dtheta_z / r : 0.0;
    out[0][0] = sfx * (k + b[0] * dk_x);
    out[0][1] = sfx * (b[0] * dk_y);
    out[0][2] = sfx * (b[0] * dk_z);
    out[1][0] = sfy * (b[1] * dk_x);
    out[1][1] = sfy * (k + b[1] * dk_y);
    out[1][2] = sfy * (b[1] * dk_z);
  }
};

// What the split bands measure. Near against far is a pitch and left against
// right is a roll: a pitch error moves a ground point at range R by
// (R^2+h^2)/h and a height error by R, so the two separate exactly, and roll
// scales every range by the plane's sideways tilt at that lateral offset. Each
// pair needs the geometry it sat at to become an angle, and the estimator
// cannot work that out for itself because the region of interest lives here.
struct RoadBands
{
  double near_step = std::numeric_limits<double>::quiet_NaN();
  double far_step = std::numeric_limits<double>::quiet_NaN();
  double near_curve = 0.0;
  double far_curve = 0.0;
  double left_step = std::numeric_limits<double>::quiet_NaN();
  double right_step = std::numeric_limits<double>::quiet_NaN();
  double near_range = std::numeric_limits<double>::quiet_NaN();
  double far_range = std::numeric_limits<double>::quiet_NaN();
  double left_lateral = std::numeric_limits<double>::quiet_NaN();
  double right_lateral = std::numeric_limits<double>::quiet_NaN();
  double near_forward = std::numeric_limits<double>::quiet_NaN();
  double far_forward = std::numeric_limits<double>::quiet_NaN();
};

// What the four-parameter solve found, when it is asked to run.
//
// The bands above reach a pitch and a roll by comparing two one-parameter
// answers taken over two halves of the region; this reaches them by letting one
// fit over the whole region carry all four unknowns at once. The plane distance
// is deliberately not among them: scaling the whole scene about the lens leaves
// every bearing exactly where it was, so no photometric cost can see it.
struct RoadSolve
{
  double step = std::numeric_limits<double>::quiet_NaN();   // metres over the hop
  double yaw = std::numeric_limits<double>::quiet_NaN();    // radians, body
  double pitch = std::numeric_limits<double>::quiet_NaN();
  double roll = std::numeric_limits<double>::quiet_NaN();
  double score = std::numeric_limits<double>::quiet_NaN();  // ZNCC at the answer
  int iterations = 0;
  bool ok = false;
};

struct TrackState
{
  cv::Mat previous_gray;
  // ov_core's TrackKLT keeps the pyramid it built for the last image and
  // hands it straight to the next flow call. Passing raw images instead makes
  // OpenCV rebuild both sides on every call, and there are two calls a frame
  // -- four builds where one is needed.
  std::vector<cv::Mat> previous_pyramid;
  std::vector<cv::Point2f> points;
  // Where each point moved on the hop before this one, used to start the
  // search in the right place rather than at the feature's last position.
  std::vector<cv::Point2f> velocities;
  // The same thing said once for the whole frame instead of once per feature.
  // Image motion of a plane between two views is exactly a homography, and the
  // ground is a plane, so one 3x3 describes every ground feature's hop -- new
  // ones included, which a per-feature velocity cannot. It also carries the
  // perspective: apply it to a point that has moved closer and it returns a
  // larger step, which is the acceleration a constant pixel velocity misses.
  // Empty until two frames have been seen.
  cv::Mat plane_motion;
  // Where the vehicle's own motion says each held point will be. Filled before
  // the flow and consumed by it; empty means fall back to the per-feature hop.
  std::vector<cv::Point2f> predicted;
  // The previous frame resampled into where the vehicle's motion says it will
  // land. Matching against this instead of the raw previous frame removes the
  // patch deformation rather than the displacement: a ground point approaching
  // a downward camera does not merely move, it grows and shears, and a
  // translation-only flow cannot match a patch that changed shape however well
  // it is aimed. Measured, that is the failure at speed -- the front camera's
  // photometric rejections go from 0 at 2 m/s to 191 at 8.
  std::vector<cv::Mat> warped_pyramid;
  std::vector<int64_t> identities;
  int64_t next_identity = 0;
  // When the last frame arrived and how long the hop before it took. A
  // dropped frame doubles the gap, and a prediction that does not know that
  // is wrong by a whole frame of motion.
  double stamp = 0.0;
  double interval = 0.0;
  // Virtual features carried by a photometric fit of the road region instead
  // of by per-corner flow. The road is a rapidly moving, low-textured surface
  // -- sparse corners on it die in a median of eight frames and their
  // aggregate carries a frame-wide error ten to twenty times what independent
  // point noise would give. One warp fitted over thousands of road pixels does
  // not have that: it is the same cue Song and Chandraker call plane-guided
  // dense stereo (CVPR 2014), reported there as the term that "vastly improves
  // camera self-localization".
  std::vector<cv::Point2f> road_points;
  std::vector<int64_t> road_identities;
  std::vector<int> road_ages;
  // Previous frame to current, in ROI coordinates. Kept as the warm start for
  // the next fit, which is most of why the iteration converges at all.
  cv::Mat road_warp;
  // (x, y, dx, dy) per grid cell: what the plane fit left behind this frame.
  std::vector<double> parallax;
  // This camera's own photometric step, in metres per nominal frame. Each
  // camera measures it separately: the two answers are what the estimator's
  // fusion is built on, and collapsing them to one would remove the binding
  // that the front/rear split provides.
  double road_step = std::numeric_limits<double>::quiet_NaN();
  // How well the region landed on itself at the answer. A road that aligns
  // reads 0.93 to 0.99; a frame that lost the surface reads far lower, and
  // that is the only thing separating a measurement from a tail event.
  double road_score = 0.0;
  // How far apart the across-track tiles' answers are, relative to the step.
  // Large where something that is not the road is in the region.
  double road_spread = 0.0;
  // The same step over the near and far halves of the band, and over its left
  // and right halves.
  RoadBands road_bands;
  // The four-parameter answer, when `road_step_esm` asked for one. Its step
  // replaces the search's; its three angles are what nothing else here emits.
  RoadSolve road_esm;
  // Where the next frame's search starts. Smoothed, and never published.
  double road_bracket = std::numeric_limits<double>::quiet_NaN();
  // Average |answer - bracket| / |answer| over recent frames. Negative until
  // there have been two answers to compare, which is what tells the search to
  // keep its fixed width.
  double road_predict_error = -1.0;
  // How many frames in a row the mirrored bracket has scored higher. The sign
  // only changes when this reaches the threshold; a single frame is noise.
  int road_reverse_votes = 0;

  // How many features this camera is currently trying to hold. Owned by the
  // one thread that runs this camera's callback, so it needs no guard.
  int target = 0;
  // The fraction of the frame, measured from the bottom edge, where this
  // camera must not detect.
  //
  // Ground approaches a forward camera and recedes from a rearward one, so one
  // edge of the image is where new ground enters and the other is where it
  // leaves. A feature born at the leaving edge has no future: measured on
  // straight_s8 at 7.5 m/s, of the 65,044 front-camera features born in the
  // bottom fifth, **97.9% die after exactly one frame** and 0.1% reach five,
  // against 30.2% and 49.0% for the top fifth. The detector spreads births
  // evenly over the image and cannot see the difference, so a fifth of every
  // frame's 532 new identities are minted onto ground that is about to leave --
  // and `found_after_observations` promotes each of them to an anchor.
  //
  // Per camera because the leaving edge is not the same edge for both, and
  // `detection_skip_top_fraction` is the other half of the same instrument.
  // The rear camera does not need it: its births already sit at its entering
  // edge and its lifetime is flat across the frame.
  double skip_bottom = 0.0;

#ifdef MONOSCALE_TRACKER_HAS_CUDA
  // The same two things the CPU path keeps, on the device. The pyramid is held
  // rather than rebuilt for the same reason: two flow calls a frame would
  // otherwise build four pyramids where two are needed.
  std::vector<cv::cuda::GpuMat> previous_device_pyramid;
  // One stream per camera, so a frame's whole conversation with the device --
  // the image up, both flow calls, every result back -- is queued once and
  // waited on once. It belongs to the camera and not to the node because the
  // two callbacks run at the same time on their own threads.
  cv::cuda::Stream stream;
  // Page-locked staging for everything that crosses the bus.
  cv::cuda::HostMem host_image;
  cv::cuda::HostMem host_source;
  cv::cuda::HostMem host_guess;
  cv::cuda::HostMem host_forward;
  cv::cuda::HostMem host_backward;
  cv::cuda::HostMem host_status;
  cv::cuda::HostMem host_backward_status;
  cv::cuda::HostMem host_error;
  // Held across frames so the device allocator is not asked for the same
  // handful of buffers on every hop.
  cv::cuda::GpuMat device_source;
  cv::cuda::GpuMat device_forward;
  cv::cuda::GpuMat device_status;
  cv::cuda::GpuMat device_error;
  cv::cuda::GpuMat device_backward;
  cv::cuda::GpuMat device_backward_status;
#endif
};

// What the flow did on one hop. Only the report reads it, but the survivors'
// mean hop on its own cannot tell a camera that is not moving apart from one
// whose moving features are all being filtered out.
struct FollowStats
{
  double raw = 0.0;     // mean hop over every point the flow kept
  size_t lost = 0;      // the flow gave up, forward or backward
  size_t noisy = 0;     // patch error above the threshold
  size_t drifted = 0;   // failed the forward-backward check
};

// Where a frame's time went. Which stage grows with the image and which with
// the feature count is not obvious from the code: the flow is charged per
// point, but the pyramid it builds first is charged per pixel.
struct StageTimes
{
  double prep = 0.0;      // decode and resize
  double follow = 0.0;    // the two flow calls and the checks
  double trim = 0.0;
  double detect = 0.0;
  double publish = 0.0;
  // Inside `follow`. Sweeping the flow's own drivers -- levels, window,
  // feature count -- moved the total by under 10%, which says the cost is not
  // where the arithmetic says it should be. These say where it is.
  double scale = 0.0;     // resize to the processing width
  double pyramid = 0.0;   // buildOpticalFlowPyramid
  double flow = 0.0;      // the two calcOpticalFlowPyrLK calls and their gates
  double road = 0.0;      // the photometric step, bands, ESM and parallax
  double step = 0.0;      // the corner-derived step
};

}  // namespace

class FeatureTracker : public rclcpp::Node
{
public:
  // Totals at teardown, because the periodic line is a snapshot: the
  // estimator reported receiving more messages than the last status said had
  // been published, which is only possible if the counter was read early.
  ~FeatureTracker()
  {
    std::string totals;
    for (const auto & entry : published_) {
      totals += " " + entry.first + "=" + std::to_string(entry.second);
    }
    RCLCPP_INFO(get_logger(), "published total:%s", totals.c_str());
    RCLCPP_INFO(
      get_logger(), "step sign: frames=%ld negative=%ld flips=%ld",
      sign_frames_, sign_negative_, sign_flips_);
    if (esm_compare_count_[0] > 0) {
      for (int k = 0; k < 4; ++k) {
        if (esm_compare_count_[k] == 0) {
          continue;
        }
        RCLCPP_INFO(
          get_logger(), "jacobian column %d: mean %.4f worst %.4f over %ld",
          k, esm_compare_sum_[k] / esm_compare_count_[k], esm_compare_worst_[k],
          esm_compare_count_[k]);
      }
    }
    RCLCPP_INFO(
      get_logger(), "fit: solves=%ld warps=%ld jacobians=%ld warps/solve=%.1f",
      fit_solves_, patch_warps_, fit_jacobians_,
      fit_solves_ > 0 ? double(patch_warps_) / fit_solves_ : 0.0);
  }

  FeatureTracker()
  : Node("feature_tracker")
  {
    // Negative leaves OpenCV's own default, which is what to use. Pinning the
    // flow to a single thread on the theory that two cameras plus the
    // estimator were oversubscribing the machine made the frame cost four
    // times worse -- 60-79 ms against 15-21 -- and the drive scored 20.9 m
    // against 4.0. The internal parallelism is earning its keep.
    cv::setNumThreads(declare_parameter<int>("opencv_threads", -1));
    processing_width_ = declare_parameter<int>("processing_width", 640);
    // 600 was set at parking speed and starves the ground registration at a
    // steady 8 m/s: a third of solves returned nothing, 3.478 m against 0.250.
    // 2000 fixed that, but once detection spread the features over the frame
    // instead of letting them pile up, 1200 turned out to be both cheaper and
    // more accurate -- 0.041 m against 0.076 at 8 m/s, 0.096 against 0.138 at
    // 2.5 m/s. Well placed features are worth more than many of them.
    //
    // 700 is where the frame fits the budget. At 1200 the front camera needs
    // 21 ms against the 16.7 a 60 Hz frame allows, so it runs at 50 Hz, and
    // everything downstream reads as a different bug: a sixth of the pairs
    // rejected on the translation gate, the two cameras disagreeing, the
    // trajectory finishing a fifth short, 2.8 m of position error. None of it
    // survives the frame fitting. At 700 the cameras hold 59 Hz, one pair in
    // the drive is rejected instead of thirty, and the real-time score is the
    // offline score -- 0.041 m either way.
    max_features_ = declare_parameter<int>("max_features_per_camera", 700);
    // What one frame is allowed to cost, in milliseconds, and how far the
    // feature count may be cut to stay under it. A fixed count cannot serve
    // both ends of the range: 2000 scores 0.095 m at 2.5 m/s and 2.8 m at
    // 8 m/s, 700 scores 0.041 m at 8 m/s and 0.131 at 2.5. The budget is the
    // thing that actually differs -- at parking speed the hops are small and
    // the frame is cheap, at road speed it is not -- so let the measurement
    // set the count instead of guessing it. 0 holds max_features_ fixed.
    //
    // Off by default, because a steady count turns out to be worth more than
    // a well chosen one: letting it move scored 0.059-0.064 m at 8 m/s where
    // a fixed 700 scored 0.038-0.042, even though the controller settled on
    // more features than that. Features that come and go have short lives,
    // and the anchor map is built out of long ones. Worth turning on where
    // the speed range is wider than these drives cover -- it evens the two
    // ends out, 0.060 and 0.109 against 0.040 and 0.131.
    frame_budget_ms_ = declare_parameter<double>("frame_budget_ms", 0.0);
    min_features_ = declare_parameter<int>("min_features_per_camera", 500);
    quality_level_ = declare_parameter<double>("quality_level", 0.01);
    // Publish how distinct each feature is against its surroundings.
    //
    // The estimator has never had this. It receives identities and pixels, so
    // when it chooses which candidates get the anchor map's scarce free slots
    // it can only rank them by where they are, not by whether they are really
    // landmarks -- and ranking by position was measured to be worse than not
    // ranking at all. The corner response is the quantity `goodFeaturesToTrack`
    // already thresholds on and then throws away.
    publish_clarity_ = declare_parameter<bool>("publish_clarity", false);

    // Publish the parallax the plane fit leaves behind, on a coarse grid.
    //
    // The road warp puts every point that lies on the plane back where it came
    // from. What does not lie on the plane does not go back, and how far it
    // misses is its height: this is the same relation the sparse obstacle path
    // solves from feature slip, but read densely and from a fit that already
    // ran. The warp is computed for the scoring anyway; only the second flow
    // and the sampling are new.
    // Run the coarse pass of the step search on a copy of the pair this many
    // times smaller. Locating the peak needs range and resolving it needs
    // precision, and only the second needs the pixels; at 2 the coarse pass
    // costs a quarter of the area. 1 keeps both passes at the frame's own size,
    // which is what every recorded number came from.
    road_step_coarse_divisor_ = declare_parameter<int>("road_step_coarse_divisor", 1);
    // Size the search bracket by the measured prediction error rather than the
    // fixed 35% of the step. The bracket is this many times the recent average
    // |answer - prediction|, clamped to [2%, 35%], so the same sample count
    // resolves a good prediction finer and a bad one still reaches. 0 keeps the
    // fixed width.
    road_step_bracket_k_ = declare_parameter<double>("road_step_bracket_k", 0.0);
    // How many of the four-parameter fit's parameters may move; see the note in
    // `solve_step_esm`. 4 is what every recorded number came from.
    road_step_esm_dof_ = declare_parameter<int>("road_step_esm_dof", 4);
    // Let the step search cross zero. Off is what every recorded number came
    // from; see the note in `measure_step_photometric`.
    road_step_reverse_ = declare_parameter<bool>("road_step_reverse", false);
    // Sample the warp inside the correlation instead of calling cv::remap, to
    // get past its 1/32-pixel map quantisation. See the note in `road_scores`.
    road_step_fused_ = declare_parameter<bool>("road_step_fused", false);
    // The lattice the four-parameter fit warps on, when it should be finer than
    // the search's. 0 keeps `road_step_stride`.
    road_step_fit_stride_ = declare_parameter<int>("road_step_fit_stride", 0);
    // Rebuild the fit's numeric Jacobian every this many iterations. 1 is every
    // one, which is what every recorded number came from.
    road_step_esm_rebuild_ = declare_parameter<int>("road_step_esm_rebuild", 1);
    // Build the fit's Jacobian from the model's derivatives instead of from
    // warps. Off is what every recorded number came from.
    road_step_esm_analytic_ = declare_parameter<bool>("road_step_esm_analytic", false);
    band_samples_ = declare_parameter<int>("road_step_band_samples", 9);
    arc_hop_ = declare_parameter<bool>("road_step_arc", false);
    esm_compare_ = declare_parameter<bool>("road_step_esm_compare", false);
    // How far the winning step has to clear a step of nothing before the frame
    // is believed. Zero support is what the tracks that died contribute, so
    // this is the margin between "the vehicle did not move" and "the corners
    // stopped saying anything". 1.0 asks only to tie it.
    step_observable_margin_ =
      declare_parameter<double>("step_observable_margin", 1.0);
    parallax_grid_ = declare_parameter<int>("parallax_grid", 0);
    parallax_flow_scale_ = declare_parameter<double>("parallax_flow_scale", 0.5);
    min_distance_ = declare_parameter<double>("min_feature_distance_px", 8.0);
    // 15, not 21: the search costs the square of this and 21 bought nothing
    // measurable. Together with the feature count it halves the frame cost,
    // which is the difference between holding 60 Hz and not.
    window_ = declare_parameter<int>("lk_window_px", 15);
    levels_ = declare_parameter<int>("lk_pyramid_levels", 4);
    // The backward check earns its cost: dropping it, or even running it on
    // every other hop, let mistracked points survive to the solve and moved
    // position error from 0.10 m to 0.73 and 0.29 respectively.
    backward_threshold_ = declare_parameter<double>(
      "lk_forward_backward_threshold_px", 1.0);
    error_threshold_ = declare_parameter<double>("lk_error_threshold", 30.0);
    // Applied per grid cell, not to the frame as a whole -- see detect().
    refill_ratio_ = declare_parameter<double>("feature_refill_ratio", 0.7);
    // 4x3 is enough to keep the horizon from crowding out the near ground,
    // and no finer: 8x5 tracked just as well and cost 50 ms a frame against
    // 15, all of it per-cell detection overhead rather than pixels.
    grid_columns_ = declare_parameter<int>("detection_grid_columns", 4);
    grid_rows_ = declare_parameter<int>("detection_grid_rows", 3);
    warm_start_ = declare_parameter<bool>("lk_warm_start", true);
    predict_by_plane_ =
      declare_parameter<bool>("lk_predict_by_plane", false);
    seed_new_from_cell_ =
      declare_parameter<bool>("lk_seed_new_from_cell", false);
    seed_min_flow_ =
      static_cast<float>(declare_parameter<double>("lk_seed_min_flow_px", 8.0));
    // Photometric alignment of the road region, emitted as virtual features.
    road_alignment_ = declare_parameter<bool>("road_alignment", false);
    road_roi_ = {
      declare_parameter<double>("road_roi_x0", 0.20),
      declare_parameter<double>("road_roi_y0", 0.50),
      declare_parameter<double>("road_roi_x1", 0.80),
      declare_parameter<double>("road_roi_y1", 1.00)};
    road_columns_ = declare_parameter<int>("road_grid_columns", 3);
    road_rows_ = declare_parameter<int>("road_grid_rows", 3);
    road_iterations_ = declare_parameter<int>("road_ecc_iterations", 30);
    road_homography_ = declare_parameter<bool>("road_homography", false);
    // How many frames a grid point may be carried before it is dropped and
    // reseeded. Each frame multiplies one more warp onto its position, so a
    // long-lived point accumulates the fit's error; on a curve that error has
    // a consistent direction and does not average out. 0 keeps them until they
    // leave the region.
    road_max_age_ = declare_parameter<int>("road_max_age", 0);
    road_epsilon_ = declare_parameter<double>("road_ecc_epsilon", 1.0e-4);
    // Score the step by photometric alignment of the far road instead of by
    // counting corners that agree. Off leaves `measure_step` in charge.
    road_step_photometric_ = declare_parameter<bool>("road_step_photometric", false);
    // Measure this camera's step photometrically and carry the road grid by the
    // homography that step induces, instead of by a free affine fit. Needs
    // `road_alignment` for the grid and a ground model for the geometry.
    road_from_step_ = declare_parameter<bool>("road_from_step", false);
    // Also measure the band in halves, so the plane offset and the two
    // mounting pitches can be solved for without truth.
    road_step_calibrate_ = declare_parameter<bool>("road_step_calibrate", false);
    // Solve the step and the three body angles together, over the same
    // photometric cost, instead of searching the step alone. Off leaves the
    // one-dimensional bracket in sole charge and emits no angles at all.
    road_step_esm_ = declare_parameter<bool>("road_step_esm", false);
    // Half-width of the band sweep, in pixels of image motion.
    road_step_band_window_px_ =
      declare_parameter<double>("road_step_band_window_px", 3.0);
    // The near band, in fractions of the frame, and the default for a camera
    // that does not name its own. Ground from 0.34 m -- the closest row the
    // frame holds -- out to about 1.5 m, one lane wide.
    //
    // The two cameras do not want the same band, and the reason is the
    // direction the ground flows. The front's ground approaches, so it enters
    // high and leaves at the bottom: every row was in the previous frame. The
    // rear's recedes, so the bottom rows are ground that has just appeared --
    // at 8 m/s a 29-row strip there moves about 47 px in one frame, which is
    // more than the strip is tall, and it measures -35% against truth while the
    // same strip at 1.6 m/s measures -0.7%. Masking the fabricated border does
    // not rescue it (it reads -63%); the band has to be left out. Hence
    // `rear.road_step_roi_y1` at 0.92 while the front keeps 1.00.
    //
    // The far band was tried first, on the reasoning that it is where the
    // corner path has nothing. It loses by two orders of magnitude, and the
    // reason is that both facts have one cause: a step moves a ground point's
    // bearing by h/(R^2+h^2), which at 5 m is a fifteenth of what it is at 1 m.
    // Where a corner cannot be identified is where the step cannot be seen.
    // Doubling the resolution does not rescue it, and widening this region
    // sideways or outwards makes it worse, not better, because the road leaves
    // the plane the warp assumes.
    road_step_roi_ = {
      declare_parameter<double>("road_step_roi_x0", 0.25),
      declare_parameter<double>("road_step_roi_y0", 0.60),
      declare_parameter<double>("road_step_roi_x1", 0.75),
      declare_parameter<double>("road_step_roi_y1", 1.00)};
    // Coarse candidates, then the same count again over one coarse spacing.
    road_step_samples_ = declare_parameter<int>("road_step_samples", 13);
    road_step_stride_ = declare_parameter<int>("road_step_stride", 8);
    // Across-track tiles, each answering on its own; the median is taken. 1
    // keeps the single-region search.
    road_step_tiles_ = declare_parameter<int>("road_step_tiles", 1);
    road_step_dump_ = declare_parameter<std::string>("road_step_dump", "");
    // Run the optical flow on the GPU.
    //
    // This is the one stage in the whole stack where a GPU has a case to make.
    // Measured on an Orin Nano at 25 W, a 640 px frame with 1600 features costs
    // 42-43 ms, of which the two flow calls are 35-38; the 30 Hz budget is
    // 33.3. Everything else -- detection, the trim, the publish -- is under
    // 5 ms together, and the estimator downstream is under 3.
    //
    // Off by default, and it stays off until it has been measured on the board
    // it is meant to help. The CPU path is what every recorded number came
    // from, and a GPU path that is merely plausible is worth less than a CPU
    // path that is known.
    use_cuda_ = declare_parameter<bool>("use_cuda", false);
    // How many iterations the GPU flow runs per pyramid level.
    //
    // The CPU stops at whichever comes first, thirty iterations or a step under
    // 0.01 px. OpenCV's CUDA kernel has no epsilon and always runs the count,
    // so on an ambiguous patch it keeps walking after the CPU would have
    // stopped. Before the gates, the GPU's mean hop reads 18.5 px against the
    // CPU's 8.6 on the same frames, which is what that looks like.
    //
    // Which makes lowering it look obvious, and it is a trap. Position RMSE
    // over the same two drives, tracks recorded from each arm and replayed
    // through the same estimator:
    //
    //                    CPU      GPU 10    GPU 30
    //   clean60_release  0.0989   0.0773    0.1106
    //   fish8            0.0618   0.1750    0.0593
    //
    // Ten iterations is a 22% win on one drive and a 2.8x loss on the other.
    // Thirty -- OpenCV's default, and the count the CPU would reach if its
    // epsilon never fired -- sits in the same band as the CPU on both. So the
    // default stays where it is, and the drop to ten stays here as a warning
    // rather than as a setting anyone should reach for.
    cuda_iterations_ = declare_parameter<int>("lk_cuda_iterations", 30);
    if (use_cuda_) {
#ifdef MONOSCALE_TRACKER_HAS_CUDA
      if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cuda_active_ = true;
        forward_flow_ = cv::cuda::SparsePyrLKOpticalFlow::create(
          cv::Size(window_, window_), levels_, cuda_iterations_, true);
        backward_flow_ = cv::cuda::SparsePyrLKOpticalFlow::create(
          cv::Size(window_, window_), levels_, cuda_iterations_, false);
        RCLCPP_INFO(
          get_logger(), "optical flow on the GPU: %s",
          cv::cuda::DeviceInfo(0).name());
      } else {
        RCLCPP_WARN(get_logger(), "use_cuda asked for, but no CUDA device; staying on the CPU");
      }
#else
      RCLCPP_WARN(
        get_logger(),
        "use_cuda asked for, but this OpenCV has no cudaoptflow; staying on the CPU");
#endif
    }
    // Fraction of the image, measured from the top, to leave out of feature
    // detection. The estimator keeps only what projects onto the ground
    // between ground_min_distance_m and ground_max_distance_m, so anything
    // above the horizon -- sky, buildings, the road far ahead -- is tracked
    // at full cost and then discarded.
    //
    // This was once justified by the front camera costing 29-34 ms a frame
    // against the rear's 12-14. That comparison was worthless: the rear was
    // holding a stationary band of horizon features and doing no work at all.
    // With both cameras tracking, the two cost the same. 0 keeps the whole
    // image, which is what has been measured for accuracy.
    skip_top_ = declare_parameter<double>("detection_skip_top_fraction", 0.0);
    skip_bottom_ = declare_parameter<double>("detection_skip_bottom_fraction", 0.0);
    dump_dir_ = declare_parameter<std::string>("debug_dump_directory", "");
    // Best effort with a shallow queue is right against a live camera, where
    // a late frame is worth less than the one behind it. It is wrong when the
    // input is a bag being replayed for measurement: the rear camera kept
    // under half its frames that way, and a configuration cannot be compared
    // against another on a different subset of the drive. Reliable with a
    // deep queue makes the replay lossless and the simulator wait instead.
    const auto reliability = declare_parameter<std::string>(
      "input_reliability", "best_effort");
    // Half a second at 60 Hz, where five frames was 83 ms. A frame costs
    // 9-10 ms here, so this is never a backlog being worked through -- it is
    // room for the scheduler to be late without the frame being lost. Five
    // was not enough room: the number of pairs the estimator managed to form
    // from the same recording swung by a tenth between runs, and with it the
    // score.
    const int depth = static_cast<int>(declare_parameter<int>("input_queue_depth", 30));
    rclcpp::QoS input_qos(static_cast<size_t>(std::max(depth, 1)));
    if (reliability == "reliable") {
      input_qos.reliable();
    } else {
      input_qos.best_effort();
    }

    // The input comes from a camera and best effort is right for it: a late
    // frame is worth less than the one behind it, and nothing can be done to
    // make a camera wait. The output goes to the estimator, which is ours,
    // and there is no reason to lose any of it -- every track dropped on that
    // link is a pair the estimator never forms, and it showed up as the same
    // recording scoring differently run to run.
    const auto output_reliability = declare_parameter<std::string>(
      "output_reliability", "reliable");
    const int output_depth =
      static_cast<int>(declare_parameter<int>("output_queue_depth", 200));
    // KEEP_ALL on the writer is the other half of a lossless link: the reader
    // asking for every sample cannot get them if the writer has already
    // dropped its own history. Offline only -- on the vehicle a writer that
    // holds frames for a slow reader is a writer that goes on holding them.
    const auto output_history =
      declare_parameter<std::string>("output_history", "keep_last");
    rclcpp::QoS output_qos = output_history == "keep_all"
      ? rclcpp::QoS(rclcpp::KeepAll())
      : rclcpp::QoS(static_cast<size_t>(std::max(output_depth, 1)));
    const auto output_durability =
      declare_parameter<std::string>("output_durability", "volatile");
    if (output_durability == "transient_local") {
      output_qos.transient_local();
    }
    if (output_reliability == "best_effort") {
      output_qos.best_effort();
    } else {
      output_qos.reliable();
    }

    // Predicting the flow from the vehicle's own motion instead of from each
    // feature's last hop. Off until asked for.
    predict_from_motion_ = declare_parameter<bool>("motion_prediction", false);
    // Which camera is trusted to measure the step. Under forward motion the
    // rear camera's ground enters its field of view while the front camera's
    // leaves, so the rear keeps a population that moved and the front does not:
    // measured over nine drives the rear recovers the step to 1.3-7.5 mm, and
    // the front fails outright at 8 m/s, reporting zero with 64% support.
    step_reference_ = declare_parameter<std::string>("motion_reference", "rear");
    step_search_slack_ = declare_parameter<double>("motion_search_slack_m", 0.08);
    step_tolerance_deg_ = declare_parameter<double>("motion_inlier_deg", 0.3);
    // Resample the previous frame through the prediction before matching, so
    // the flow sees a patch that has already been deformed the way the motion
    // deforms it.
    motion_warp_ = declare_parameter<bool>("motion_warp", false);
    motion_warp_min_step_ =
      declare_parameter<double>("motion_warp_min_step_m", 0.12);

    const auto cameras = declare_parameter<std::vector<std::string>>(
      "cameras", std::vector<std::string>{"front", "rear"});
    const auto topics = declare_parameter<std::vector<std::string>>(
      "image_topics",
      std::vector<std::string>{
        "/sensing/camera/front/vio/image_raw",
        "/sensing/camera/rear/vio/image_raw"});

    if (cameras.size() != topics.size()) {
      RCLCPP_FATAL(get_logger(), "cameras and image_topics must be the same length");
      throw std::runtime_error("camera configuration mismatch");
    }

    camera_names_ = cameras;
    image_topics_ = topics;
    for (size_t index = 0; index < cameras.size(); ++index) {
      const std::string & name = cameras[index];
      states_[name].target = max_features_;
      states_[name].skip_bottom = declare_parameter<double>(
        name + ".detection_skip_bottom_fraction", skip_bottom_);
      if (predict_from_motion_ || road_from_step_) {
        models_[name] = load_ground_model(name);
        road_bands_[name] = load_road_band(name);
      }
      publish_topic_[name] = "/vision/tracks/" + name;
      publishers_[name] = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/vision/tracks/" + name, output_qos);
      // Each camera needs its own callback group. Subscriptions left in the
      // node's default group are mutually exclusive with each other, so a
      // multi-threaded executor runs them one at a time no matter how many
      // threads it has -- the rear camera kept 36% of its frames against the
      // front's 97% because it was always the one waiting.
      groups_.push_back(
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));
      rclcpp::SubscriptionOptions options;
      options.callback_group = groups_.back();
      subscriptions_.push_back(
        create_subscription<sensor_msgs::msg::Image>(
          topics[index], input_qos,
          [this, name](sensor_msgs::msg::Image::ConstSharedPtr message) {
            onImage(name, *message);
          },
          options));
      RCLCPP_INFO(
        get_logger(), "tracking %s on %s", name.c_str(), topics[index].c_str());
    }

    if (predict_from_motion_ || road_from_step_) {
      // The turn in the induced homography is measured, not solved for. This
      // rig's gyro yaw is exact -- correlation 0.985 to 0.9997 against the
      // truth pose, scale 0.97 to 1.004 -- so the only thing left to find is
      // the forward step.
      const auto imu_topic =
        declare_parameter<std::string>("imu_topic", "/sensing/imu/imu_data");
      imu_topic_ = imu_topic;
      rclcpp::QoS imu_qos(rclcpp::KeepLast(400));
      imu_qos.best_effort();
      groups_.push_back(
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));
      rclcpp::SubscriptionOptions imu_options;
      imu_options.callback_group = groups_.back();
      imu_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, imu_qos,
        [this](sensor_msgs::msg::Imu::ConstSharedPtr message) {on_imu(*message);},
        imu_options);
      RCLCPP_INFO(
        get_logger(), "motion prediction on, heading from %s, step from %s",
        imu_topic.c_str(), step_reference_.c_str());
    }
  }

  // The whole recording, in stamp order, through the same handlers the live
  // node uses -- but called directly rather than delivered.
  //
  // Live, the frames arrive over the middleware: two camera callbacks in their
  // own groups, an executor with four threads, and a queue that fills at
  // whatever rate the player and the tracker happen to run. Which frame is
  // processed when is then a property of that run, and the same recording
  // extracted twice gives different tracks -- measured on str_1.5 at 1.70x
  // between the best and worst of three repeats, which is larger than most of
  // the differences anyone has tried to measure with it.
  //
  // Here there is no player, no middleware and no queue: the bag is read in
  // order and each message is handed to its handler. The result is the same
  // every time, and it is also much faster, because nothing waits for real
  // time to pass.
  // Analytic against central difference, on the model the node was configured
  // with. A derivation that is wrong in one term still descends, just to the
  // wrong place, so it has to be checked against the thing it replaces before
  // it is allowed to replace it.
  int check_jacobian()
  {
    if (models_.empty()) {
      RCLCPP_ERROR(get_logger(), "no ground model; needs road_from_step");
      return 2;
    }
    const auto & model = models_.begin()->second;
    const int w = 640;
    const int h = 360;
    double worst_h = 0.0;
    double worst_p = 0.0;
    unsigned seed = 12345;
    const auto next = [&seed]() {
        seed = seed * 1103515245u + 12345u;
        return static_cast<double>((seed >> 16) & 0x7fff) / 32767.0;
      };
    for (int trial = 0; trial < 200; ++trial) {
      const double p[4] = {
        0.05 + 0.4 * next(), -0.05 + 0.1 * next(),
        -0.03 + 0.06 * next(), -0.03 + 0.06 * next()};
      cv::Matx33d analytic[4];
      model.homography_jacobian(p[0], p[1], p[2], p[3], analytic);
      for (int k = 0; k < 4; ++k) {
        const double e = 1e-6;
        double up[4] = {p[0], p[1], p[2], p[3]};
        double down[4] = {p[0], p[1], p[2], p[3]};
        up[k] += e;
        down[k] -= e;
        const cv::Matx33d numeric =
          (model.homography(up[0], up[1], up[2], up[3], arc_hop_) -
          model.homography(down[0], down[1], down[2], down[3], arc_hop_)) *
          (1.0 / (2.0 * e));
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            const double scale = std::max(std::abs(numeric(i, j)), 1e-6);
            worst_h = std::max(worst_h, std::abs(analytic[k](i, j) - numeric(i, j)) / scale);
          }
        }
      }
      // The projection, at a bearing that came from somewhere in the image.
      const cv::Point2f q(
        static_cast<float>(0.2 * w + 0.6 * w * next()),
        static_cast<float>(0.5 * h + 0.5 * h * next()));
      const cv::Vec3d b = model.bearing(q, w, h);
      double jac[2][3];
      model.pixel_jacobian(b, w, h, jac);
      for (int c = 0; c < 3; ++c) {
        // `pixel` returns a Point2f, so the step has to be large enough that
        // the difference clears float32 resolution -- at pixel values in the
        // hundreds that is about 3e-5, and 1e-7 gives nothing but rounding.
        const double e = 1e-4;
        cv::Vec3d up = b;
        cv::Vec3d down = b;
        up[c] += e;
        down[c] -= e;
        const cv::Point2f a = model.pixel(up, w, h);
        const cv::Point2f d = model.pixel(down, w, h);
        const double nx = (a.x - d.x) / (2.0 * e);
        const double ny = (a.y - d.y) / (2.0 * e);
        const double sx = std::max(std::abs(nx), std::abs(jac[0][c]));
        const double sy = std::max(std::abs(ny), std::abs(jac[1][c]));
        const double ex = sx > 1e-6 ? std::abs(jac[0][c] - nx) / sx : 0.0;
        const double ey = sy > 1e-6 ? std::abs(jac[1][c] - ny) / sy : 0.0;
        if (std::max(ex, ey) > worst_p) {
          worst_p = std::max(ex, ey);
          RCLCPP_INFO(
            get_logger(),
            "  worst so far c=%d  analytic (%.4f, %.4f)  numeric (%.4f, %.4f)  b=(%.4f,%.4f,%.4f)",
            c, jac[0][c], jac[1][c], nx, ny, b[0], b[1], b[2]);
        }
      }
    }
    RCLCPP_INFO(
      get_logger(), "jacobian check: homography %.2e  projection %.2e (relative)",
      worst_h, worst_p);
    return (worst_h < 1e-4 && worst_p < 1e-4) ? 0 : 1;
  }

  int run_offline(const std::string & input, const std::string & output)
  {
    rosbag2_cpp::Reader reader;
    reader.open(input);
    rosbag2_cpp::Writer writer;
    writer.open(output);
    offline_ = &writer;

    rclcpp::Serialization<sensor_msgs::msg::Image> image_codec;
    rclcpp::Serialization<sensor_msgs::msg::Imu> imu_codec;
    rclcpp::Serialization<sensor_msgs::msg::CameraInfo> info_codec;
    rclcpp::Serialization<geometry_msgs::msg::PoseWithCovarianceStamped> pose_codec;

    std::map<std::string, std::string> camera_of;
    for (size_t i = 0; i < camera_names_.size() && i < image_topics_.size(); ++i) {
      camera_of[image_topics_[i]] = camera_names_[i];
    }
    // A frame waits until the instrument has passed it.
    //
    // `yaw_at` interpolates between the samples either side of a stamp, so a
    // frame handed over before the sample after it exists gets no heading and
    // the turn is unknown for that hop. Live that never happens -- the queue is
    // 400 deep and the tracker is always behind -- but reading the bag in order
    // hands each frame over the moment it is met. Holding frames until an
    // inertial sample past their stamp has been ingested reproduces the live
    // ordering without depending on how fast anything runs.
    std::vector<std::pair<std::string, sensor_msgs::msg::Image>> pending;
    double instrument = 0.0;
    long images = 0;
    const auto drain = [&](bool flush) {
        size_t kept = 0;
        for (size_t i = 0; i < pending.size(); ++i) {
          const double stamp = rclcpp::Time(pending[i].second.header.stamp).seconds();
          if (flush || stamp < instrument) {
            onImage(pending[i].first, pending[i].second);
            ++images;
          } else {
            pending[kept++] = std::move(pending[i]);
          }
        }
        pending.resize(kept);
      };
    while (reader.has_next()) {
      const auto message = reader.read_next();
      rclcpp::SerializedMessage serialized(*message->serialized_data);
      const auto camera = camera_of.find(message->topic_name);
      if (camera != camera_of.end()) {
        sensor_msgs::msg::Image image;
        image_codec.deserialize_message(&serialized, &image);
        pending.emplace_back(camera->second, std::move(image));
        continue;
      }
      if (message->topic_name == imu_topic_) {
        sensor_msgs::msg::Imu imu;
        imu_codec.deserialize_message(&serialized, &imu);
        on_imu(imu);
        instrument = rclcpp::Time(imu.header.stamp).seconds();
        drain(false);
        // Carried through so the output bag is what the estimator needs.
        writer.write(imu, message->topic_name, rclcpp::Time(imu.header.stamp));
        continue;
      }
      // Everything the estimator reads but the tracker does not: copied over
      // unchanged so one bag is enough to score with.
      if (message->topic_name.find("camera_info") != std::string::npos) {
        sensor_msgs::msg::CameraInfo info;
        info_codec.deserialize_message(&serialized, &info);
        writer.write(info, message->topic_name, rclcpp::Time(info.header.stamp));
        continue;
      }
      if (message->topic_name.find("ground_truth") != std::string::npos) {
        geometry_msgs::msg::PoseWithCovarianceStamped pose;
        pose_codec.deserialize_message(&serialized, &pose);
        writer.write(pose, message->topic_name, rclcpp::Time(pose.header.stamp));
        continue;
      }
    }
    drain(true);
    offline_ = nullptr;
    RCLCPP_INFO(get_logger(), "offline: %ld images", images);
    return 0;
  }

private:
  void onImage(const std::string & name, const sensor_msgs::msg::Image & message)
  {
    out_stamp_ = rclcpp::Time(message.header.stamp);
    const auto entered = std::chrono::steady_clock::now();
    auto mark = entered;
    const auto lap = [&mark]() {
      const auto now = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(now - mark).count();
      mark = now;
      return seconds;
    };
    StageTimes stage;
    cv::Mat gray;
    if (!to_gray(message, gray)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "unsupported image encoding: %s",
        message.encoding.c_str());
      return;
    }
    if (processing_width_ > 0 && gray.cols > processing_width_) {
      const double scale = static_cast<double>(processing_width_) / gray.cols;
      // Rounded, not truncated: 540 * (640/960) lands on 359.99999999999994 in
      // double, so a cast gives 359 rows where the image path gives 360.
      cv::resize(
        gray, gray, cv::Size(processing_width_, cvRound(gray.rows * scale)),
        0, 0, cv::INTER_AREA);
    }

    TrackState & state = states_[name];
    std::vector<cv::Mat> pyramid;
#ifdef MONOSCALE_TRACKER_HAS_CUDA
    // Only declared where the type is complete: without cudaoptflow, OpenCV
    // forward declares GpuMat and nothing can be held in a vector of them.
    std::vector<cv::cuda::GpuMat> device_pyramid;
    if (cuda_active_) {
      build_device_pyramid(state, gray, device_pyramid);
    } else {
    stage.scale = lap();
      cv::buildOpticalFlowPyramid(gray, pyramid, cv::Size(window_, window_), levels_);
    }
#else
    cv::buildOpticalFlowPyramid(gray, pyramid, cv::Size(window_, window_), levels_);
#endif
    stage.prep = lap();

    std::vector<cv::Point2f> previous_points;
    std::vector<cv::Point2f> current_points;
    std::vector<cv::Point2f> velocities;
    std::vector<int64_t> identities;
    FollowStats stats;
    state.predicted.clear();
    state.warped_pyramid.clear();

    // How much longer this hop is than the one the stored velocities measure.
    // Frames get dropped under load, and when one is missed the next hop
    // covers twice the ground. Predicting one frame of motion into a two
    // frame gap leaves the search half a hop short, which costs iterations,
    // fails the backward check, and drops more features -- so the next frame
    // is dearer still and the camera never catches back up. Scaling by the
    // gap that actually elapsed is what stops that from running away.
    const double stamp = static_cast<double>(message.header.stamp.sec) +
      static_cast<double>(message.header.stamp.nanosec) * 1e-9;
    const double elapsed = state.stamp > 0.0 ? stamp - state.stamp : 0.0;
    const double reach = state.interval > 0.0 && elapsed > 0.0
      ? std::clamp(elapsed / state.interval, 0.25, 4.0)
      : 1.0;

    const bool have_previous = !state.points.empty() &&
      state.previous_gray.size() == gray.size();

    // Where the vehicle's own motion says the held points will be. Built before
    // the flow so the search starts there rather than where the feature was.
    double turn = 0.0;
    bool turn_known = false;
    if ((predict_from_motion_ || road_from_step_) && have_previous &&
      !state.points.empty())
    {
      double yaw_now = 0.0;
      double yaw_then = 0.0;
      if (yaw_at(stamp, yaw_now) && yaw_at(state.stamp, yaw_then)) {
        turn = std::remainder(yaw_now - yaw_then, 2.0 * M_PI);
        turn_known = true;
      }
      double step = 0.0;
      bool have_step = false;
      {
        std::lock_guard<std::mutex> guard(step_lock_);
        step = shared_step_;
        have_step = step_ready_;
      }
      const auto found = models_.find(name);
      if (turn_known && have_step && found != models_.end() && found->second.ready) {
        const double hop = step * reach;
        const cv::Matx33d h = found->second.homography(hop, turn, 0.0, 0.0, arc_hop_);
        state.predicted.resize(state.points.size());
        for (size_t i = 0; i < state.points.size(); ++i) {
          const cv::Vec3d b =
            h * found->second.bearing(state.points[i], gray.cols, gray.rows);
          state.predicted[i] = found->second.pixel(b, gray.cols, gray.rows);
        }
        // Only where the deformation is worth a resample. The warp costs one
        // bilinear interpolation of the whole frame, and at 1.5 m/s the flow is
        // four pixels -- the signal is sub-pixel and the interpolation noise
        // sits on top of it. At 8 m/s the near ground sweeps 68 px a frame and
        // removing that deformation is worth far more than the resample costs.
        if (motion_warp_ && std::abs(hop) >= motion_warp_min_step_ &&
          !state.previous_gray.empty() &&
          state.previous_gray.size() == gray.size())
        {
          cv::Mat map_x;
          cv::Mat map_y;
          build_warp(found->second, hop, turn, gray.cols, gray.rows, map_x, map_y);
          cv::Mat warped;
          cv::remap(
            state.previous_gray, warped, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_REPLICATE);
          // Same depth as the frame it will be matched against: the flow
          // takes one maxLevel for both sides.
          cv::buildOpticalFlowPyramid(
            warped, state.warped_pyramid, cv::Size(window_, window_), levels_);
        }
      }
    }

    if (cuda_active_) {
#ifdef MONOSCALE_TRACKER_HAS_CUDA
      if (have_previous && !state.previous_device_pyramid.empty()) {
    stage.pyramid = lap();
        follow_cuda(
          state, gray, device_pyramid, previous_points, current_points, velocities,
          identities, stats, reach);
      }
#endif
    } else if (have_previous && !state.previous_pyramid.empty()) {
      follow(
        state, gray, pyramid, previous_points, current_points, velocities,
        identities, stats, reach);
    }
    // And what this frame says the step was, if this is the camera trusted to
    // say it. Searched as one scalar over the induced family: the features that
    // failed to track all agree on zero, so the vote is bimodal, and a window
    // around the last answer keeps the zero out of reach.
    stage.flow = lap();
    if (road_from_step_ && turn_known && !state.previous_gray.empty() &&
      state.previous_gray.size() == gray.size())
    {
      const auto model = models_.find(name);
      if (model != models_.end() && model->second.ready) {
        const bool held = std::isfinite(state.road_bracket);
        double peak = 0.0;
        double spread = 0.0;
        const double found = measure_step_photometric(
          model->second, state.previous_gray, gray, turn, reach,
          held ? state.road_bracket : 0.0, held, band_for(name),
          state.road_predict_error, &state.road_reverse_votes, &peak, &spread);
        state.road_score = peak;
        state.road_spread = spread;
        if (road_step_calibrate_ && std::isfinite(found)) {
          measure_step_bands(
            model->second, state.previous_gray, gray, turn, reach, found,
            band_for(name), state.road_bands);
        }
        // Then the same region again, with the three angles freed. The bands
        // above are left where they are: they seed nothing here and they stay
        // comparable frame to frame whether this runs or not.
        double answer = found;
        state.road_esm = RoadSolve();
        if (road_step_esm_ && std::isfinite(found)) {
          const double span = std::max(reach, 1e-3);
          state.road_esm = solve_step_esm(
            model->second, state.previous_gray, gray, turn, found * span,
            band_for(name));
          if (state.road_esm.ok) {
            answer = state.road_esm.step / span;
          }
        }
        // What that fit could not explain. Uses the answer it just settled on,
        // so the plane part is removed with the best motion available.
        if (parallax_grid_ > 0 && std::isfinite(answer)) {
          const double span = std::max(reach, 1e-3);
          measure_parallax(
            model->second, state.previous_gray, gray, answer * span, turn,
            state.road_esm.ok ? state.road_esm.pitch : 0.0,
            state.road_esm.ok ? state.road_esm.roll : 0.0,
            band_for(name), state.parallax);
        } else {
          state.parallax.clear();
        }
        if (std::isfinite(answer)) {
          // Two values, deliberately. The smoothed one brackets the next
          // frame's search, because a bracket wants the best guess available.
          // The raw one is what leaves this node, because smoothing an output
          // correlates consecutive measurements by construction, and correlated
          // hops accumulate as n instead of root n -- measured as the lag-1
          // autocorrelation rising from 0.66 to 0.90 while the bias fell.
          // How wrong the bracket that just ran turned out to be, as a fraction
          // of the answer. This is the quantity the next frame's bracket should
          // be sized by, so it is measured rather than assumed. Same gain as
          // the bracket itself, and it is updated before the bracket moves so
          // it prices the prediction that was actually used.
          if (std::isfinite(state.road_bracket) && std::abs(answer) > 1e-6) {
            const double miss = std::abs(answer - state.road_bracket) / std::abs(answer);
            state.road_predict_error = state.road_predict_error >= 0.0
              ? state.road_predict_error + 0.3 * (miss - state.road_predict_error) : miss;
          }
          // Same three counts `measure_step` keeps, on the path that actually
          // runs when the step comes from the road. One camera only, so the
          // numbers mean the same thing in both places.
          if (name == step_reference_ && std::isfinite(answer)) {
            if (std::isfinite(state.road_step) && (state.road_step < 0.0) != (answer < 0.0)) {
              ++sign_flips_;
            }
            ++sign_frames_;
            if (answer < 0.0) {
              ++sign_negative_;
            }
          }
          state.road_step = answer;
          state.road_bracket = std::isfinite(state.road_bracket)
            ? state.road_bracket + 0.7 * (answer - state.road_bracket) : answer;
        }
      }
    }
    stage.road = lap();
    if (predict_from_motion_ && turn_known && name == step_reference_ &&
      previous_points.size() >= 60)
    {
      const auto found = models_.find(name);
      if (found != models_.end() && found->second.ready) {
        measure_step(found->second, previous_points, current_points,
          gray.cols, gray.rows, turn, reach);
        if ((road_step_photometric_ || !road_step_dump_.empty()) &&
          !state.previous_gray.empty() && state.previous_gray.size() == gray.size())
        {
          double held = 0.0;
          bool ready = false;
          {
            std::lock_guard<std::mutex> guard(step_lock_);
            held = shared_step_;
            ready = step_ready_;
          }
          const double photometric = measure_step_photometric(
            found->second, state.previous_gray, gray, turn, reach, held, ready,
            band_for(name));
          if (!road_step_dump_.empty()) {
            std::lock_guard<std::mutex> guard(road_step_lock_);
            if (road_step_file_ == nullptr) {
              road_step_file_ = std::fopen(road_step_dump_.c_str(), "w");
              if (road_step_file_ != nullptr) {
                std::fprintf(road_step_file_, "stamp,votes,photometric,turn,reach\n");
              }
            }
            if (road_step_file_ != nullptr) {
              std::fprintf(
                road_step_file_, "%.6f,%.6f,%.6f,%.6f,%.4f\n", stamp, held,
                photometric, turn, reach);
              std::fflush(road_step_file_);
            }
          }
          if (road_step_photometric_ && std::isfinite(photometric)) {
            std::lock_guard<std::mutex> guard(step_lock_);
            shared_step_ = photometric;
            step_ready_ = true;
          }
        }
      }
    }
    if (elapsed > 0.0) {
      state.interval = elapsed;
    }
    state.stamp = stamp;

    stage.step = lap();
    stage.follow = stage.scale + stage.pyramid + stage.flow +
      stage.road + stage.step;
    trim(
      gray, previous_points, current_points, velocities, identities,
      state.target);
    stage.trim = lap();

    // How far the surviving features moved on this hop. If the cost per
    // point differs between the cameras, this is what would explain it: a
    // longer jump makes the pyramid search descend further and iterate more.
    double shift = 0.0;
    for (const auto & step : velocities) {
      shift += std::hypot(step.x, step.y);
    }
    if (!velocities.empty()) {
      shift /= static_cast<double>(velocities.size());
    }

    // How different this frame is from the one it was just compared against.
    // The rear camera reported a 0.03 px hop where the very same recorded
    // frames give 15 px when the flow is run over them offline, and the only
    // way to get that is to compare a picture with itself.
    double change = 0.0;
    if (!state.previous_gray.empty() && state.previous_gray.size() == gray.size()) {
      change = cv::norm(state.previous_gray, gray, cv::NORM_L1) /
        static_cast<double>(gray.total());
    }

    if (predict_by_plane_ && previous_points.size() >= 12 &&
      previous_points.size() == current_points.size())
    {
      // RANSAC because the horizon and anything standing up are not on the
      // plane; they are the minority and the ground is the consensus.
      cv::Mat fitted = cv::findHomography(previous_points, current_points, cv::RANSAC, 3.0);
      if (!fitted.empty() && fitted.rows == 3 && fitted.cols == 3) {
        state.plane_motion = fitted;
      }
    }
    state.points = current_points;
    state.velocities = velocities;
    state.identities = identities;
    const size_t followed = state.points.size();
    detect(state, gray);
    stage.detect = lap();
    // Held before it is replaced: the road fit needs the frame the flow just
    // came from, and detection has already overwritten the state's copy.
    const cv::Mat road_previous = state.previous_gray;
    state.previous_gray = gray;
    if (cuda_active_) {
#ifdef MONOSCALE_TRACKER_HAS_CUDA
      state.previous_device_pyramid = device_pyramid;
#endif
    } else {
      state.previous_pyramid = pyramid;
    }

    // Features detected on this frame go out too, standing still: previous
    // position equal to current, because they have not moved yet. The image
    // path puts them into the tracked set the moment it finds them, so they
    // are in the snapshot a solve compares against and survive the whole of
    // the next interval. Holding them back for one frame left this path
    // solving over a pool that was always a generation older -- median
    // baseline 0.272 m against 0.389 -- and the shorter the baseline the
    // noisier the geometry.
    //
    // Appended to the followed set in place. Copying the three vectors whole
    // to do this was copying the entire frame's output for the sake of the
    // handful of points detection had just added.
    for (size_t i = followed; i < state.points.size(); ++i) {
      previous_points.push_back(state.points[i]);
      current_points.push_back(state.points[i]);
      identities.push_back(state.identities[i]);
    }

    if (road_alignment_ && !road_previous.empty() && road_previous.size() == gray.size()) {
      const auto model = models_.find(name);
      const bool physical = road_from_step_ && turn_known &&
        std::isfinite(state.road_step) && model != models_.end() && model->second.ready;
      if (physical) {
        const cv::Matx33d plane =
          model->second.homography(state.road_step * reach, turn, 0.0, 0.0, arc_hop_);
        align_road(
          state, road_previous, gray, previous_points, current_points, identities,
          &model->second, &plane);
      } else {
        align_road(state, road_previous, gray, previous_points, current_points, identities);
      }
    }

    if (!dump_dir_.empty() && ++dumped_[name] == 60) {
      cv::Mat canvas;
      cv::cvtColor(gray, canvas, cv::COLOR_GRAY2BGR);
      for (const auto & point : state.points) {
        cv::circle(canvas, point, 2, cv::Scalar(0, 0, 255), -1);
      }
      cv::imwrite(dump_dir_ + "/dump_" + name + ".png", canvas);
    }

    // How distinct each surviving feature is, sampled from the same minimum
    // eigenvalue `goodFeaturesToTrack` scores with, at the same block size.
    std::vector<double> clarity;
    if (publish_clarity_ && !current_points.empty()) {
      cv::Mat response;
      cv::cornerMinEigenVal(gray, response, 7);
      clarity.reserve(current_points.size());
      for (const auto & point : current_points) {
        const int x = std::clamp(cvRound(point.x), 0, response.cols - 1);
        const int y = std::clamp(cvRound(point.y), 0, response.rows - 1);
        clarity.push_back(static_cast<double>(response.at<float>(y, x)));
      }
    }
    publish(
      name, message, gray.size(), previous_points, current_points, identities,
      road_from_step_ && std::isfinite(state.road_step)
      ? state.road_step * reach : std::numeric_limits<double>::quiet_NaN(),
      state.road_score, state.road_spread, state.road_bands, state.road_esm, reach,
      clarity, state.parallax);

    stage.publish = lap();
    const double spent = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - entered).count() * 1000.0;
    if (frame_budget_ms_ > 0.0) {
      // Cut hard when over, creep back when there is room to spare. A frame
      // that overruns costs the next one too -- it arrives late, its hop is
      // twice as long and dearer still -- so the way out has to be quicker
      // than the way in.
      if (spent > frame_budget_ms_) {
        state.target = std::max(min_features_, state.target - state.target / 8);
      } else if (spent < 0.7 * frame_budget_ms_) {
        state.target = std::min(max_features_, state.target + state.target / 32 + 1);
      }
    }

    report(
      name,
      spent / 1000.0,
      state.points.size(),
      shift,
      velocities.size(),
      change,
      stats,
      stage);
  }

  void report(
    const std::string & name, double seconds, size_t held, double shift,
    size_t followed, double change, const FollowStats & stats,
    const StageTimes & stage)
  {
    std::lock_guard<std::mutex> guard(count_mutex_);
    ++received_[name];
    spent_[name] += seconds;
    held_[name] += static_cast<double>(held);
    shift_[name] += shift;
    followed_[name] += static_cast<double>(followed);
    change_[name] += change;
    prep_[name] += stage.prep;
    follow_[name] += stage.follow;
    scale_[name] += stage.scale;
    pyramid_[name] += stage.pyramid;
    flow_[name] += stage.flow;
    road_[name] += stage.road;
    step_[name] += stage.step;
    trim_[name] += stage.trim;
    detect_[name] += stage.detect;
    pub_[name] += stage.publish;
    raw_[name] += stats.raw;
    lost_[name] += static_cast<double>(stats.lost);
    noisy_[name] += static_cast<double>(stats.noisy);
    drifted_[name] += static_cast<double>(stats.drifted);
    const rclcpp::Time now = get_clock()->now();
    if (!counting_) {
      counted_since_ = now;
      counting_ = true;
      return;
    }
    const double span = (now - counted_since_).seconds();
    if (span < 2.0) {
      return;
    }
    std::string line;
    for (const auto & entry : received_) {
      const int hz = static_cast<int>(entry.second / span + 0.5);
      const int ms = entry.second > 0
        ? static_cast<int>(1000.0 * spent_[entry.first] / entry.second + 0.5)
        : 0;
      const int points = entry.second > 0
        ? static_cast<int>(held_[entry.first] / entry.second + 0.5)
        : 0;
      line += entry.first + "=" + std::to_string(hz) + "Hz/" +
        std::to_string(ms) + "ms/" + std::to_string(points) + "pts ";
      const double px = entry.second > 0
        ? shift_[entry.first] / entry.second
        : 0.0;
      const int kept = entry.second > 0
        ? static_cast<int>(followed_[entry.first] / entry.second + 0.5)
        : 0;
      const double changed = entry.second > 0
        ? change_[entry.first] / entry.second
        : 0.0;
      const auto stage_ms = [&](std::map<std::string, double> & bucket) {
        return std::to_string(
          1000.0 * bucket[entry.first] / entry.second).substr(0, 4);
      };
      line += "[prep=" + stage_ms(prep_) + " follow=" + stage_ms(follow_) +
        "(scale=" + stage_ms(scale_) + " pyr=" + stage_ms(pyramid_) +
        " flow=" + stage_ms(flow_) + " road=" + stage_ms(road_) +
        " step=" + stage_ms(step_) + ")" +
        " trim=" + stage_ms(trim_) + " detect=" + stage_ms(detect_) +
        " pub=" + stage_ms(pub_) + "] ";
      line += "(" + std::to_string(px).substr(0, 4) + "px, " +
        std::to_string(kept) + " followed, d=" +
        std::to_string(changed).substr(0, 5) + ", raw=" +
        std::to_string(raw_[entry.first] / entry.second).substr(0, 5) +
        "px, lost/noisy/drifted=" +
        std::to_string(static_cast<int>(lost_[entry.first] / entry.second + 0.5)) + "/" +
        std::to_string(static_cast<int>(noisy_[entry.first] / entry.second + 0.5)) + "/" +
        std::to_string(static_cast<int>(drifted_[entry.first] / entry.second + 0.5)) + ") ";
      followed_[entry.first] = 0.0;
      spent_[entry.first] = 0.0;
      held_[entry.first] = 0.0;
      shift_[entry.first] = 0.0;
      change_[entry.first] = 0.0;
      prep_[entry.first] = 0.0;
      scale_[entry.first] = 0.0;
      pyramid_[entry.first] = 0.0;
      flow_[entry.first] = 0.0;
      road_[entry.first] = 0.0;
      step_[entry.first] = 0.0;
      follow_[entry.first] = 0.0;
      trim_[entry.first] = 0.0;
      detect_[entry.first] = 0.0;
      pub_[entry.first] = 0.0;
      raw_[entry.first] = 0.0;
      lost_[entry.first] = 0.0;
      noisy_[entry.first] = 0.0;
      drifted_[entry.first] = 0.0;
    }
    for (auto & entry : received_) {
      entry.second = 0;
    }
    counted_since_ = now;
    // The enclosing report() already holds count_mutex_; taking it again on
    // a non-recursive mutex froze the tracker outright, which is a reminder
    // that instrumentation is code and can break what it measures.
    std::string totals;
    for (const auto & entry : published_) {
      totals += " pub_" + entry.first + "=" + std::to_string(entry.second);
    }
    RCLCPP_INFO(
      get_logger(), "front end in: %s%s%s", line.c_str(), totals.c_str(),
      predict_from_motion_
      ? (" step=" + std::to_string(shared_step_) + (step_ready_ ? "" : " (none)") +
      " models=" + std::to_string(std::count_if(
        models_.begin(), models_.end(),
        [](const auto & e) {return e.second.ready;}))).c_str()
      : "");
  }


  // Photometric alignment of the road region, emitted as virtual features.
  //
  // A grid of points is carried across frames by one affine warp fitted to the
  // whole road ROI rather than by tracking each of them. Everything downstream
  // sees ordinary features with ordinary identities; what is different is where
  // their positions come from. Per-corner flow on a low-textured road is
  // aperture limited and its errors are shared across the frame -- measured on
  // this rig, the hop those corners produce is eight to twenty times noisier
  // than independent point noise would allow. A warp fitted over the whole
  // region is not aperture limited.
  // The grid the road region carries, and what carries it.
  //
  // `plane` is the frame's motion written as the homography a plane induces --
  // the vehicle's step and turn, and nothing else, because the mount, the lens
  // and the height are calibration. When it is supplied the ECC fit is not run
  // at all: six or eight free parameters are fitted where three exist, and the
  // spare ones absorb noise with a direction, which is why the region fit has
  // never survived a curve.
  void align_road(
    TrackState & state, const cv::Mat & previous, const cv::Mat & current,
    std::vector<cv::Point2f> & previous_points, std::vector<cv::Point2f> & current_points,
    std::vector<int64_t> & identities, const GroundModel * model = nullptr,
    const cv::Matx33d * plane = nullptr)
  {
    const cv::Rect roi(
      cv::Point(
        cvRound(road_roi_[0] * current.cols), cvRound(road_roi_[1] * current.rows)),
      cv::Point(
        cvRound(road_roi_[2] * current.cols), cvRound(road_roi_[3] * current.rows)));
    const cv::Rect bounded = roi & cv::Rect(0, 0, current.cols, current.rows);
    if (bounded.width < 32 || bounded.height < 32) {
      return;
    }

    const bool from_plane = model != nullptr && plane != nullptr && model->ready;
    const int motion = road_homography_ ? cv::MOTION_HOMOGRAPHY : cv::MOTION_AFFINE;
    const int rows = road_homography_ ? 3 : 2;
    if (state.road_warp.empty() || state.road_warp.rows != rows) {
      state.road_warp = cv::Mat::eye(rows, 3, CV_32F);
    }
    // Warm started from the last frame's warp. Starting from identity every
    // time asks the iteration to cross the whole hop, which at road speed is
    // further than it can see.
    cv::Mat warp = state.road_warp.clone();
    if (!from_plane) {
    try {
      cv::findTransformECC(
        previous(bounded), current(bounded), warp, motion,
        cv::TermCriteria(
          cv::TermCriteria::COUNT + cv::TermCriteria::EPS, road_iterations_, road_epsilon_),
        cv::noArray(), 5);
    } catch (const cv::Exception &) {
      // Did not converge. The grid is dropped rather than carried on a warp
      // nobody vouched for; it is reseeded next frame.
      state.road_points.clear();
      state.road_identities.clear();
      state.road_ages.clear();
      state.road_warp = cv::Mat::eye(rows, 3, CV_32F);
      return;
    }
    if (!warp.empty() && cv::checkRange(warp)) {
      state.road_warp = warp;
    } else {
      return;
    }
    }

    const cv::Mat_<float> w = state.road_warp;
    const cv::Point2f origin(
      static_cast<float>(bounded.x), static_cast<float>(bounded.y));
    // Carry whoever is already there, then emit the pair. Coordinates are the
    // ROI's for the warp and the frame's for the message.
    std::vector<cv::Point2f> carried;
    std::vector<int64_t> kept;
    carried.reserve(state.road_points.size());
    kept.reserve(state.road_points.size());
    std::vector<int> ages;
    ages.reserve(state.road_points.size());
    for (size_t i = 0; i < state.road_points.size(); ++i) {
      if (road_max_age_ > 0 && i < state.road_ages.size() &&
        state.road_ages[i] >= road_max_age_)
      {
        continue;
      }
      const cv::Point2f & from = state.road_points[i];
      cv::Point2f to;
      if (from_plane) {
        // Straight through the geometry: a bearing in the previous frame, the
        // plane's homography, and back to a pixel. The pair this emits is
        // exactly consistent with the step that was measured, so what the
        // estimator recovers from it is that step and nothing else.
        const cv::Vec3d b = (*plane) *
          model->bearing(from + origin, current.cols, current.rows);
        to = model->pixel(b, current.cols, current.rows) - origin;
      } else {
        const float scale = road_homography_
          ? w(2, 0) * from.x + w(2, 1) * from.y + w(2, 2) : 1.0f;
        if (std::abs(scale) < 1e-6f) {
          continue;
        }
        to = cv::Point2f(
          (w(0, 0) * from.x + w(0, 1) * from.y + w(0, 2)) / scale,
          (w(1, 0) * from.x + w(1, 1) * from.y + w(1, 2)) / scale);
      }
      if (to.x < 0.0f || to.y < 0.0f ||
        to.x > static_cast<float>(bounded.width - 1) ||
        to.y > static_cast<float>(bounded.height - 1))
      {
        continue;
      }
      previous_points.push_back(from + origin);
      current_points.push_back(to + origin);
      identities.push_back(state.road_identities[i]);
      carried.push_back(to);
      kept.push_back(state.road_identities[i]);
      ages.push_back(i < state.road_ages.size() ? state.road_ages[i] + 1 : 1);
    }

    // Refill the grid wherever it has emptied. A cell is free when nothing
    // carried lands within half a cell of its centre.
    const float step_x = static_cast<float>(bounded.width) / road_columns_;
    const float step_y = static_cast<float>(bounded.height) / road_rows_;
    for (int row = 0; row < road_rows_; ++row) {
      for (int column = 0; column < road_columns_; ++column) {
        const cv::Point2f centre(
          (column + 0.5f) * step_x, (row + 0.5f) * step_y);
        bool taken = false;
        for (const auto & point : carried) {
          if (std::abs(point.x - centre.x) < 0.5f * step_x &&
            std::abs(point.y - centre.y) < 0.5f * step_y)
          {
            taken = true;
            break;
          }
        }
        if (taken) {
          continue;
        }
        carried.push_back(centre);
        kept.push_back(road_next_identity_++);
        ages.push_back(0);
      }
    }
    state.road_points = carried;
    state.road_identities = kept;
    state.road_ages = ages;
  }

  void follow(
    TrackState & state, const cv::Mat & gray,
    const std::vector<cv::Mat> & pyramid,
    std::vector<cv::Point2f> & previous_points,
    std::vector<cv::Point2f> & current_points,
    std::vector<cv::Point2f> & velocities,
    std::vector<int64_t> & identities, FollowStats & stats,
    double reach) const
  {
    // Match against the warped previous frame when there is one. The points
    // start where the warp put them, so what the flow finds is the residual
    // the motion model did not predict.
    const bool warped = !state.warped_pyramid.empty() &&
      state.predicted.size() == state.points.size() && !state.points.empty();
    const std::vector<cv::Mat> & source_pyramid =
      warped ? state.warped_pyramid : state.previous_pyramid;
    const std::vector<cv::Point2f> & source_points =
      warped ? state.predicted : state.points;

    std::vector<cv::Point2f> forward;
    std::vector<uchar> status;
    std::vector<float> error;
    const cv::Size window(window_, window_);
    const cv::TermCriteria criteria(
      cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);
    // Start each search where the feature was heading. The near ground sweeps
    // fastest and carries most of the metric information, so those are the
    // points that fall into a wrong local minimum without a guess -- which
    // biases displacement short and the whole trajectory with it. The image
    // path predicts from the estimated motion; the front end has no access to
    // that, so it carries each feature's own last hop forward instead.
    int flags = 0;
    if (state.predicted.size() == state.points.size() && !state.points.empty()) {
      forward = state.predicted;
      flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    } else if (predict_by_plane_ && !state.plane_motion.empty() && !state.points.empty()) {
      // One prediction for the whole frame, from the plane's last hop.
      std::vector<cv::Point2f> warped;
      cv::perspectiveTransform(state.points, warped, state.plane_motion);
      forward.resize(state.points.size());
      for (size_t i = 0; i < state.points.size(); ++i) {
        const cv::Point2f step = warped[i] - state.points[i];
        forward[i] = state.points[i] + step * static_cast<float>(reach);
      }
      flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    } else if (warm_start_ && state.velocities.size() == state.points.size()) {
      forward.resize(state.points.size());
      for (size_t i = 0; i < state.points.size(); ++i) {
        forward[i] = state.points[i] +
          state.velocities[i] * static_cast<float>(reach);
      }
      flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    }
    cv::calcOpticalFlowPyrLK(
      source_pyramid, pyramid, source_points, forward, status, error,
      window, levels_, criteria, flags);

    // What the flow returned before anything was thrown away, and why each
    // rejection happened. The survivors' mean hop alone cannot tell a camera
    // that is not moving apart from one whose moving features are all being
    // filtered out.
    double raw = 0.0;
    size_t raw_count = 0;
    size_t lost = 0;
    size_t noisy = 0;
    size_t drifted = 0;

    // Settle everything the forward pass decides on its own, then run the
    // backward pass over the survivors alone.
    //
    // Both flow calls are charged per point and together they are most of the
    // frame. A point the forward pass lost, or whose patch error is already
    // over the threshold, or that landed off the image, is going to be dropped
    // whatever comes back -- so the second call was paying full price for
    // points on their way to the bin. The gates below are the same gates in
    // the same conjunction, and the flow treats each point independently of
    // the others in the call, so the set that survives is unchanged to the
    // bit. Only which counter a doomed point lands in can move: one that fails
    // two gates is now charged to the first gate tested rather than to
    // whichever the old order happened to reach first.
    std::vector<cv::Point2f> candidates;
    std::vector<size_t> source;
    candidates.reserve(forward.size());
    source.reserve(forward.size());
    for (size_t i = 0; i < forward.size(); ++i) {
      if (status[i]) {
        raw += std::hypot(
          forward[i].x - state.points[i].x, forward[i].y - state.points[i].y);
        ++raw_count;
      }
      if (!status[i]) {
        ++lost;
        continue;
      }
      if (error[i] > error_threshold_) {
        ++noisy;
        continue;
      }
      if (forward[i].x < 0 || forward[i].y < 0 ||
        forward[i].x >= gray.cols || forward[i].y >= gray.rows)
      {
        continue;
      }
      candidates.push_back(forward[i]);
      source.push_back(i);
    }

    std::vector<cv::Point2f> backward;
    std::vector<uchar> backward_status;
    if (!candidates.empty()) {
      // No error vector out of this one. Nothing ever read it, and OpenCV
      // skips the patch sum altogether when it is not asked for -- a window's
      // worth of arithmetic per point, computed and discarded on every hop.
      cv::calcOpticalFlowPyrLK(
        pyramid, source_pyramid, candidates, backward, backward_status,
        cv::noArray(), window, levels_, criteria);
    }

    for (size_t k = 0; k < candidates.size(); ++k) {
      const size_t i = source[k];
      if (!backward_status[k]) {
        ++lost;
        continue;
      }
      const cv::Point2f drift = backward[k] - source_points[i];
      if (std::hypot(drift.x, drift.y) > backward_threshold_) {
        ++drifted;
        continue;
      }
      previous_points.push_back(state.points[i]);
      current_points.push_back(candidates[k]);
      velocities.push_back(candidates[k] - state.points[i]);
      identities.push_back(state.identities[i]);
    }
    stats.raw = raw_count > 0 ? raw / static_cast<double>(raw_count) : 0.0;
    stats.lost = lost;
    stats.noisy = noisy;
    stats.drifted = drifted;
  }

#ifdef MONOSCALE_TRACKER_HAS_CUDA
  void build_device_pyramid(
    TrackState & state, const cv::Mat & gray,
    std::vector<cv::cuda::GpuMat> & pyramid) const
  {
    // The same ladder the flow builds for itself, kept so that two calls a
    // frame do not build four pyramids where two are needed.
    //
    // Staged through page-locked memory and queued on the camera's stream. The
    // pageable upload it replaces blocked until the copy had landed and could
    // overlap with nothing, which was half the cost of building the ladder.
    pyramid.resize(static_cast<size_t>(levels_) + 1);
    if (state.host_image.empty() || state.host_image.size() != gray.size() ||
      state.host_image.type() != gray.type())
    {
      state.host_image.create(gray.rows, gray.cols, gray.type());
    }
    cv::Mat staged = state.host_image.createMatHeader();
    gray.copyTo(staged);
    pyramid[0].upload(staged, state.stream);
    for (int level = 1; level <= levels_; ++level) {
      cv::cuda::pyrDown(
        pyramid[static_cast<size_t>(level - 1)],
        pyramid[static_cast<size_t>(level)], state.stream);
    }
  }

  // The GPU twin of follow(). Every gate below is the one above it, applied to
  // the same quantities: OpenCV's CUDA kernel reports the patch error as the
  // mean absolute intensity difference over the window, which is what the CPU
  // reports and what `lk_error_threshold` was measured against.
  //
  // The two do not agree to the bit and cannot: the pyramids are built
  // differently -- cv::buildOpticalFlowPyramid pads each level by the window
  // where cuda::pyrDown does not -- and the flow itself iterates in floats
  // where the CPU uses fixed point. What has to hold is that they track the
  // same features to within a fraction of a pixel, which is what the
  // comparison in tools measures.
  void follow_cuda(
    TrackState & state, const cv::Mat & gray,
    const std::vector<cv::cuda::GpuMat> & pyramid,
    std::vector<cv::Point2f> & previous_points,
    std::vector<cv::Point2f> & current_points,
    std::vector<cv::Point2f> & velocities,
    std::vector<int64_t> & identities, FollowStats & stats, double reach)
  {
    const size_t count = state.points.size();
    // The points already sit in a contiguous buffer of exactly this layout, so
    // staging them is one memcpy rather than a loop of element writes through
    // a Mat accessor.
    cv::Mat source = pinned(state.host_source, static_cast<int>(count), CV_32FC2);
    std::memcpy(source.ptr(), state.points.data(), count * sizeof(cv::Point2f));

    // Start each search where the feature was heading, the same warm start the
    // CPU path uses: the near ground sweeps fastest and carries most of the
    // metric information, so those are the points that fall into a wrong local
    // minimum without a guess.
    cv::Mat guess = pinned(state.host_guess, static_cast<int>(count), CV_32FC2);
    const bool warm = warm_start_ && state.velocities.size() == count;
    cv::Point2f * heading = guess.ptr<cv::Point2f>();
    for (size_t i = 0; i < count; ++i) {
      heading[i] = warm
        ? state.points[i] + state.velocities[i] * static_cast<float>(reach)
        : state.points[i];
    }

    // The whole hop is queued on the camera's stream and waited on once. Every
    // upload, flow call and download used to synchronise on its own -- five
    // stalls a frame where one will do, and on a 640 px frame that plumbing
    // cost more than the flow it was carrying.
    state.device_source.upload(source, state.stream);
    state.device_forward.upload(guess, state.stream);
    (warm ? forward_flow_ : backward_flow_)
    ->calc(
      state.previous_device_pyramid, pyramid, state.device_source,
      state.device_forward, state.device_status, state.device_error,
      state.stream);

    backward_flow_->calc(
      pyramid, state.previous_device_pyramid, state.device_forward,
      state.device_backward, state.device_backward_status, cv::noArray(),
      state.stream);

    cv::Mat forward = pinned(state.host_forward, static_cast<int>(count), CV_32FC2);
    cv::Mat backward = pinned(state.host_backward, static_cast<int>(count), CV_32FC2);
    cv::Mat status = pinned(state.host_status, static_cast<int>(count), CV_8UC1);
    cv::Mat backward_status =
      pinned(state.host_backward_status, static_cast<int>(count), CV_8UC1);
    state.device_forward.download(forward, state.stream);
    state.device_backward.download(backward, state.stream);
    state.device_status.download(status, state.stream);
    state.device_backward_status.download(backward_status, state.stream);
    cv::Mat error;
    if (!state.device_error.empty()) {
      error = pinned(state.host_error, static_cast<int>(count), CV_32FC1);
      state.device_error.download(error, state.stream);
    }
    state.stream.waitForCompletion();

    double raw = 0.0;
    size_t raw_count = 0;
    size_t lost = 0;
    size_t noisy = 0;
    size_t drifted = 0;

    for (size_t i = 0; i < count; ++i) {
      const int at = static_cast<int>(i);
      const cv::Point2f moved = forward.at<cv::Point2f>(0, at);
      const uchar kept = status.at<uchar>(0, at);
      if (kept) {
        raw += std::hypot(moved.x - state.points[i].x, moved.y - state.points[i].y);
        ++raw_count;
      }
      // The same gates in the same order as the CPU arm, so a rejection is
      // charged to the same counter on both. Compacting before the backward
      // call is what the CPU does with this order; here the flow has already
      // run over every point, since compacting on the device would cost a
      // pass of its own to save a pass that is already cheap.
      if (!kept) {
        ++lost;
        continue;
      }
      if (!error.empty() && error.at<float>(0, at) > error_threshold_) {
        ++noisy;
        continue;
      }
      if (moved.x < 0 || moved.y < 0 || moved.x >= gray.cols || moved.y >= gray.rows) {
        continue;
      }
      if (!backward_status.at<uchar>(0, at)) {
        ++lost;
        continue;
      }
      const cv::Point2f drift = backward.at<cv::Point2f>(0, at) - state.points[i];
      if (std::hypot(drift.x, drift.y) > backward_threshold_) {
        ++drifted;
        continue;
      }
      previous_points.push_back(state.points[i]);
      current_points.push_back(moved);
      velocities.push_back(moved - state.points[i]);
      identities.push_back(state.identities[i]);
    }
    stats.raw = raw_count > 0 ? raw / static_cast<double>(raw_count) : 0.0;
    stats.lost = lost;
    stats.noisy = noisy;
    stats.drifted = drifted;
  }
#endif

  // The same quota, applied to what survives rather than to what is added.
  //
  // Detection alone only bounds a cell that is being filled. Nothing bounds
  // one that is merely holding: the horizon's features survive every hop, so
  // the cells that filled at the start stayed full for the rest of the drive
  // while the emptying ground cells were topped up every frame. The set grew
  // to 52949 points and 211 ms a frame with no cap in sight. Trimming keeps
  // the oldest track in each cell -- the one the anchor map has the longest
  // history for, since follow preserves order -- and drops the surplus.
  void trim(
    const cv::Mat & gray,
    std::vector<cv::Point2f> & previous_points,
    std::vector<cv::Point2f> & current_points,
    std::vector<cv::Point2f> & velocities,
    std::vector<int64_t> & identities, int target) const
  {
    const int columns = std::max(grid_columns_, 1);
    const int rows = std::max(grid_rows_, 1);
    const int quota = std::max(target, 1) / (columns * rows) + 1;
    const double cell_width = static_cast<double>(gray.cols) / columns;
    const double cell_height = static_cast<double>(gray.rows) / rows;

    std::vector<int> occupancy(static_cast<size_t>(columns * rows), 0);
    size_t kept = 0;
    for (size_t i = 0; i < current_points.size(); ++i) {
      const int cx = std::clamp(
        static_cast<int>(current_points[i].x / cell_width), 0, columns - 1);
      const int cy = std::clamp(
        static_cast<int>(current_points[i].y / cell_height), 0, rows - 1);
      if (++occupancy[static_cast<size_t>(cy * columns + cx)] > quota) {
        continue;
      }
      previous_points[kept] = previous_points[i];
      current_points[kept] = current_points[i];
      velocities[kept] = velocities[i];
      identities[kept] = identities[i];
      ++kept;
    }
    previous_points.resize(kept);
    current_points.resize(kept);
    velocities.resize(kept);
    identities.resize(kept);
  }

  // Detection has to be told where to look, not only how many to find.
  //
  // A single global count as the refill gate let the rear camera collapse.
  // Its near ground sweeps about 28 px a frame against the front's 15; the
  // warm start here carries each feature's own last hop forward, which does
  // not predict a jump that size, so those points fail the backward check and
  // die. The horizon's points do not move and never die. Once the surviving
  // horizon band alone stood above refill_ratio_ * max_features_ detection
  // never ran again: 1548 points, every one of them within 28 rows of the
  // top, mean hop 0.03 px, for the rest of the drive. The camera was reported
  // as keeping 60 Hz and holding a full feature set while contributing
  // nothing to the ground registration.
  //
  // So the quota is per cell, and so is the refill gate. A cell that has lost
  // its features is refilled whatever the rest of the frame is holding, and
  // the quality threshold is measured against its own cell rather than the
  // strongest corner in the image, which is what lets the near ground compete
  // with the horizon at all. This is what ov_core's Grider_GRID does.
  void detect(TrackState & state, const cv::Mat & gray) const
  {
    // Per-cell quota and a per-cell refill gate, which is ov_core's
    // Grider_GRID: keeping the strongest responses in each cell rather than
    // the strongest in the frame is what stops the near ground losing to the
    // horizon, and a global count as the gate let the rear camera freeze on
    // a stationary band for a whole drive.
    //
    // The detector itself is not ov_core's. Grider_FAST was ported and
    // measured: it is cheaper -- 2.2 ms a frame against 3.5, and 6 ms for the
    // whole frame against 9 -- and less accurate here, 0.036-0.094 m against
    // 0.038-0.042 over the same drive, with the scale reading long. FAST
    // takes corners above an absolute intensity threshold, and on a low
    // contrast road surface those are weaker and less repeatable than
    // minimum-eigenvalue corners; this estimate is a ground plane fitted to
    // exactly these pixels, so their placement is the measurement. Adding
    // cornerSubPix on top, as TrackKLT does, cost 5 ms and bought nothing.
    const int columns = std::max(grid_columns_, 1);
    const int rows = std::max(grid_rows_, 1);
    const int quota = std::max(state.target, 1) / (columns * rows) + 1;
    const double cell_width = static_cast<double>(gray.cols) / columns;
    const double cell_height = static_cast<double>(gray.rows) / rows;

    std::vector<int> occupancy(static_cast<size_t>(columns * rows), 0);
    for (const auto & point : state.points) {
      const int cx = std::clamp(static_cast<int>(point.x / cell_width), 0, columns - 1);
      const int cy = std::clamp(static_cast<int>(point.y / cell_height), 0, rows - 1);
      ++occupancy[static_cast<size_t>(cy * columns + cx)];
    }

    // Which cells are being refilled, settled before the mask is built.
    //
    // The mask is only ever read through a refilled cell's rectangle. Painting
    // it over the whole frame and stamping every held feature into it is work
    // the skipped cells never look at, and on a frame where nothing needs
    // refilling it is the entire cost of this stage, paid to be thrown away.
    // Once the grid is holding its quota -- which is the steady state, since
    // the point of the quota is to reach it -- that is every frame.
    std::vector<cv::Rect> refill;
    std::vector<int> room;
    for (int cy = 0; cy < rows; ++cy) {
      for (int cx = 0; cx < columns; ++cx) {
        const int held = occupancy[static_cast<size_t>(cy * columns + cx)];
        if (held > refill_ratio_ * quota) {
          continue;
        }
        const int x0 = static_cast<int>(cx * cell_width);
        const int y0 = static_cast<int>(cy * cell_height);
        const int x1 = cx + 1 == columns
          ? gray.cols : static_cast<int>((cx + 1) * cell_width);
        const int y1 = cy + 1 == rows
          ? gray.rows : static_cast<int>((cy + 1) * cell_height);
        const cv::Rect cell(x0, y0, x1 - x0, y1 - y0);
        if (cell.width < 7 || cell.height < 7) {
          continue;
        }
        refill.push_back(cell);
        room.push_back(quota - held);
      }
    }
    if (refill.empty()) {
      return;
    }

    // Left uninitialised on purpose: every pixel that will be read is written
    // below, first by the cell fill and then by the exclusion discs.
    cv::Mat mask(gray.size(), CV_8UC1);
    for (const auto & cell : refill) {
      mask(cell).setTo(255);
    }
    if (skip_top_ > 0.0) {
      const int cut = std::min(
        static_cast<int>(skip_top_ * gray.rows), gray.rows - 1);
      if (cut > 0) {
        const cv::Rect band(0, 0, gray.cols, cut);
        for (const auto & cell : refill) {
          const cv::Rect hidden = cell & band;
          if (hidden.area() > 0) {
            mask(hidden).setTo(0);
          }
        }
      }
    }
    if (state.skip_bottom > 0.0) {
      const int cut = std::min(
        static_cast<int>(state.skip_bottom * gray.rows), gray.rows - 1);
      if (cut > 0) {
        const cv::Rect band(0, gray.rows - cut, gray.cols, cut);
        for (const auto & cell : refill) {
          const cv::Rect hidden = cell & band;
          if (hidden.area() > 0) {
            mask(hidden).setTo(0);
          }
        }
      }
    }
    // A held feature only has to be stamped where it can actually block a
    // detection, which is the refilled cells its exclusion disc reaches. The
    // horizon band is what makes this worth doing: those features survive
    // every hop, so their cells stop refilling, and they are exactly the ones
    // that were being drawn every frame into a mask nobody read.
    const int radius = static_cast<int>(min_distance_);
    for (const auto & point : state.points) {
      const cv::Rect disc(
        cvRound(point.x) - radius, cvRound(point.y) - radius,
        2 * radius + 1, 2 * radius + 1);
      for (const auto & cell : refill) {
        if ((disc & cell).area() > 0) {
          cv::circle(mask, point, radius, 0, -1);
          break;
        }
      }
    }

    // What the flow is doing where each refilled cell sits, taken from the
    // features that survived into it. A feature seen once has no hop of its
    // own, and at road speed most features have been seen once: the ground
    // sweeps 15 to 22 px a frame at 8 m/s and identities last one or two
    // frames, so the warm start it was meant to get never exists. Its
    // neighbours' does, and the ground flow field is smooth enough to borrow.
    // What the flow is doing where each refilled cell sits, taken from the
    // features that survived into it. A feature seen once has no hop of its
    // own, and at road speed most features have been seen once: the ground
    // sweeps 15 to 22 px a frame at 8 m/s and identities last one or two
    // frames, so the warm start it was meant to get never exists. Its
    // neighbours' does, and the ground flow field is smooth enough to borrow.
    std::vector<cv::Point2f> seed(refill.size(), cv::Point2f(0.0f, 0.0f));
    if (seed_new_from_cell_ && state.velocities.size() == state.points.size() &&
      !state.points.empty())
    {
      const auto middle = [](std::vector<float> & v) {
          if (v.empty()) {
            return 0.0f;
          }
          const size_t half = v.size() / 2;
          std::nth_element(v.begin(), v.begin() + half, v.end());
          return v[half];
        };
      std::vector<float> xs;
      std::vector<float> ys;
      xs.reserve(state.points.size());
      ys.reserve(state.points.size());
      for (const auto & step : state.velocities) {
        xs.push_back(std::hypot(step.x, step.y));
      }
      // Decided for the whole frame, not per cell. Seeding only the fast cells
      // was measured worse at 8 m/s than seeding all of them or none: what the
      // consensus behind the solve wants is one population, and half a frame
      // carrying a neighbour's hop while the other half starts from zero is
      // two. Below a few pixels the search finds the right minimum from where
      // the feature already is anyway.
      if (middle(xs) >= seed_min_flow_) {
        xs.clear();
        ys.clear();
        for (const auto & step : state.velocities) {
          xs.push_back(step.x);
          ys.push_back(step.y);
        }
        const cv::Point2f overall(middle(xs), middle(ys));
        for (size_t c = 0; c < refill.size(); ++c) {
          xs.clear();
          ys.clear();
          for (size_t i = 0; i < state.points.size(); ++i) {
            if (refill[c].contains(state.points[i])) {
              xs.push_back(state.velocities[i].x);
              ys.push_back(state.velocities[i].y);
            }
          }
          // A cell that kept nothing borrows the frame rather than guess zero.
          seed[c] = xs.empty() ? overall : cv::Point2f(middle(xs), middle(ys));
        }
      }
    }

    std::vector<cv::Point2f> found;
    for (size_t c = 0; c < refill.size(); ++c) {
      const cv::Rect & cell = refill[c];
      found.clear();
      cv::goodFeaturesToTrack(
        gray(cell), found, room[c], quality_level_, min_distance_,
        mask(cell), 7);
      for (const auto & point : found) {
        state.points.emplace_back(point.x + cell.x, point.y + cell.y);
        // The vectors have to stay the same length or the warm start silently
        // switches itself off.
        state.velocities.push_back(seed[c]);
        state.identities.push_back(state.next_identity++);
      }
    }
  }


  // Where each pixel of one region of the predicted frame came from, rather
  // than of the whole frame. Same coarse lattice as `build_warp`; the point is
  // that a candidate step costs one small region instead of 3.7 megapixels,
  // which is what makes scanning over steps affordable.
  void build_warp_roi(
    const GroundModel & model, double step, double turn, int width, int height,
    const cv::Rect & roi, cv::Mat & map_x, cv::Mat & map_y,
    double pitch = 0.0, double roll = 0.0, int stride_override = 0) const
  {
    const cv::Matx33d inverse = model.homography(step, turn, pitch, roll, arc_hop_).inv();
    const int stride = std::max(
      stride_override > 0 ? stride_override : road_step_stride_, 1);
    const int cols = std::max(roi.width / stride, 2);
    const int rows = std::max(roi.height / stride, 2);
    cv::Mat sx(rows, cols, CV_32F);
    cv::Mat sy(rows, cols, CV_32F);
    // Same rule as `build_warp`: the lattice sits where cv::resize will read
    // it, not on a linspace across the region. Getting this wrong shifts the
    // whole map by half a lattice cell, which reads as several per cent of
    // step -- measured at +3.9% against +0.33% before the sampling was fixed.
    const double sx_step = static_cast<double>(roi.width) / cols;
    const double sy_step = static_cast<double>(roi.height) / rows;
    for (int r = 0; r < rows; ++r) {
      const double y = roi.y + (r + 0.5) * sy_step - 0.5;
      for (int c = 0; c < cols; ++c) {
        const double x = roi.x + (c + 0.5) * sx_step - 0.5;
        const cv::Vec3d b = inverse *
          model.bearing(cv::Point2f(static_cast<float>(x), static_cast<float>(y)), width, height);
        const cv::Point2f p = model.pixel(b, width, height);
        sx.at<float>(r, c) = p.x;
        sy.at<float>(r, c) = p.y;
      }
    }
    cv::resize(sx, map_x, roi.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(sy, map_y, roi.size(), 0, 0, cv::INTER_LINEAR);
  }

  // How well the road lands on itself if the vehicle stepped this far, once per
  // across-track tile.
  //
  // One number for the whole region is not robust to anything that is not the
  // road. A kerb entering from the side is off the plane, so the induced
  // homography cannot describe it; it sits closer to the camera, so it sweeps
  // faster and the fit that matches it reports a step 75 to 130 per cent too
  // long; and it is a strong straight edge, so it aligns crisply and *raises*
  // the score -- on curve_s10 the band's standard deviation goes from 5 to 24
  // where it enters, and those frames score 0.987 against the drive's 0.956. No
  // threshold on the score can reject a wrong answer that scores better than
  // the right one. Tiles can: the intruder contaminates the tile it is in, and
  // the median of the tiles is the road's answer.
  // Zero-mean normalised cross correlation from its five sums.
  //
  // `n` is the pixel count, `a` the target and `b` the warped candidate. The
  // form is the textbook one, written over sums so a region and its parts can
  // share one pass: the numerator is the covariance and the denominator the
  // product of the two variances, all times n and left unnormalised because
  // the scale cancels.
  static double zncc_from_sums(
    double n, double sa, double sb, double saa, double sbb, double sab)
  {
    if (n <= 0.0) {
      return -2.0;
    }
    const double variance = (n * saa - sa * sa) * (n * sbb - sb * sb);
    if (!(variance > 0.0)) {
      // A flat patch on either side. There is no correlation to report and no
      // reason to prefer this candidate, which is what zero says.
      return 0.0;
    }
    return (n * sab - sa * sb) / std::sqrt(variance);
  }

  void road_scores(
    const GroundModel & model, const cv::Mat & previous, const cv::Mat & current,
    double hop, double turn, const cv::Rect & roi, std::vector<double> & out,
    double * whole = nullptr) const
  {
    // With `road_step_fused` the warp is not built at all: each pixel's source
    // is computed where it is used. The lattice is what sets the precision --
    // stride 8 to 1 moves the median hop error 0.278 mm to 0.174 and RPE5 0.240
    // to 0.162, because the fisheye composed with the induced homography is not
    // linear over eight pixels and `cv::resize` joins the lattice with straight
    // lines. Stride 1 costs 9.7x, so it is not a setting; computing the exact
    // source per pixel and summing it immediately is the same geometry without
    // the lattice, the two resizes, the remap and the buffer they write.
    const bool fused = road_step_fused_;
    cv::Matx33d inverse;
    cv::Mat map_x;
    cv::Mat map_y;
    if (fused) {
      inverse = model.homography(hop, turn, 0.0, 0.0, arc_hop_).inv();
    } else {
      build_warp_roi(model, hop, turn, previous.cols, previous.rows, roi, map_x, map_y);
    }
    const int tiles = std::max(road_step_tiles_, 1);
    out.assign(static_cast<size_t>(tiles), -2.0);
    const cv::Mat target = current(roi);
    // Sampling the warp here instead of calling `cv::remap`, which rounds the
    // source position to 1/32 of a pixel (`INTER_TAB_SIZE` is 32 and the
    // coefficient table is indexed by the rounded fraction, whatever type the
    // map is).
    //
    // That quantisation is 0.125 mm where the band runs 4 mm to the pixel, and
    // the median hop error is 0.28 mm, so it looked like the floor. It is not:
    // removing it entirely moves the hop error from 0.278 mm to 0.279 and hop%
    // from 0.311 to 0.312, and costs 8% of the frame. Measured on str_1.5
    // through the deterministic offline path, so the two runs are comparable to
    // the byte.
    //
    // Kept, default off, because it is the only way to ask that question again
    // if the floor moves for another reason -- and because at a higher width
    // the quantiser and the floor change places.
    cv::Mat warped;
    if (!fused) {
      cv::remap(previous, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }
    const int source_cols = previous.cols;
    const int source_rows = previous.rows;
    const auto sample = [&](double x, double y) {
        x = std::clamp(x, 0.0, static_cast<double>(source_cols - 1));
        y = std::clamp(y, 0.0, static_cast<double>(source_rows - 1));
        const int x0 = static_cast<int>(x);
        const int y0 = static_cast<int>(y);
        const int x1 = std::min(x0 + 1, source_cols - 1);
        const int y1 = std::min(y0 + 1, source_rows - 1);
        const double fx = x - x0;
        const double fy = y - y0;
        const uchar * r0 = previous.ptr<uchar>(y0);
        const uchar * r1 = previous.ptr<uchar>(y1);
        return (1.0 - fy) * ((1.0 - fx) * r0[x0] + fx * r0[x1]) +
               fy * ((1.0 - fx) * r1[x0] + fx * r1[x1]);
      };
    // One pass over the region, five sums per tile.
    //
    // This used to be a `matchTemplate` call for the region and one more per
    // tile -- two at the configured `road_step_tiles: 1`, four at the 3 the
    // sweeps have used -- each with the template the same size as its target,
    // so each produced a 1x1 result. At that template size OpenCV correlates
    // through a DFT, which is an enormous way to obtain one number, and the
    // region was walked once per call over inputs the calls share.
    //
    // `TM_CCOEFF_NORMED` evaluated at a single position is exactly zero-mean
    // NCC, so every one of those numbers is a function of Sa, Sb, Saa, Sbb and
    // Sab over its pixels -- and the region's sums are the tiles' sums added.
    // One pass, no transform, no per-tile allocation, and the region's score
    // comes out for free.
    //
    // Worth having because this is called about 64 times a frame per camera:
    // `measure_step_photometric` scans 13 coarse and 13 fine candidates plus
    // two parabola probes, and `measure_step_bands` sweeps 9 samples over each
    // of four sub-bands. The whole road stage measures 29.5 ms a frame, so a
    // call is about 0.46 ms.
    std::vector<int> edge(static_cast<size_t>(tiles) + 1);
    for (int i = 0; i <= tiles; ++i) {
      edge[static_cast<size_t>(i)] = i * roi.width / tiles;
    }
    std::vector<int64_t> sa(tiles, 0), sb(tiles, 0);
    std::vector<int64_t> saa(tiles, 0), sbb(tiles, 0), sab(tiles, 0);
    std::vector<double> fa(tiles, 0.0), fb(tiles, 0.0);
    std::vector<double> faa(tiles, 0.0), fbb(tiles, 0.0), fab(tiles, 0.0);
    for (int y = 0; y < roi.height; ++y) {
      const uchar * a = target.ptr<uchar>(y);
      if (fused) {
        const double image_y = roi.y + y;
        for (int i = 0; i < tiles; ++i) {
          double la = 0.0, lb = 0.0, laa = 0.0, lbb = 0.0, lab = 0.0;
          for (int x = edge[static_cast<size_t>(i)]; x < edge[static_cast<size_t>(i) + 1]; ++x) {
            const cv::Vec3d bearing = inverse * model.bearing(
              cv::Point2f(static_cast<float>(roi.x + x), static_cast<float>(image_y)),
              source_cols, source_rows);
            const cv::Point2f p = model.pixel(bearing, source_cols, source_rows);
            const double va = a[x];
            const double vb = sample(p.x, p.y);
            la += va;
            lb += vb;
            laa += va * va;
            lbb += vb * vb;
            lab += va * vb;
          }
          fa[static_cast<size_t>(i)] += la;
          fb[static_cast<size_t>(i)] += lb;
          faa[static_cast<size_t>(i)] += laa;
          fbb[static_cast<size_t>(i)] += lbb;
          fab[static_cast<size_t>(i)] += lab;
        }
        continue;
      }
      const uchar * b = warped.ptr<uchar>(y);
      for (int i = 0; i < tiles; ++i) {
        // Accumulated per row into locals first: the totals are int64 and the
        // row's contribution cannot overflow int32 at any region this code
        // will ever see, so the wide adds happen once a row rather than once a
        // pixel.
        int64_t la = 0, lb = 0, laa = 0, lbb = 0, lab = 0;
        for (int x = edge[static_cast<size_t>(i)]; x < edge[static_cast<size_t>(i) + 1]; ++x) {
          const int64_t va = a[x];
          const int64_t vb = b[x];
          la += va;
          lb += vb;
          laa += va * va;
          lbb += vb * vb;
          lab += va * vb;
        }
        sa[static_cast<size_t>(i)] += la;
        sb[static_cast<size_t>(i)] += lb;
        saa[static_cast<size_t>(i)] += laa;
        sbb[static_cast<size_t>(i)] += lbb;
        sab[static_cast<size_t>(i)] += lab;
      }
    }
    for (int i = 0; i < tiles; ++i) {
      const int width = edge[static_cast<size_t>(i) + 1] - edge[static_cast<size_t>(i)];
      // Too narrow to say anything, and left at the caller's sentinel rather
      // than scored -- same rule the four-call form used.
      if (width < 8) {
        continue;
      }
      const size_t t = static_cast<size_t>(i);
      out[t] = fused
        ? zncc_from_sums(
          static_cast<double>(width) * roi.height, fa[t], fb[t], faa[t], fbb[t], fab[t])
        : zncc_from_sums(
          static_cast<double>(width) * roi.height,
          static_cast<double>(sa[t]), static_cast<double>(sb[t]),
          static_cast<double>(saa[t]), static_cast<double>(sbb[t]),
          static_cast<double>(sab[t]));
    }
    if (whole != nullptr) {
      // Every column, including any the tile loop was too narrow to report.
      double ta = 0.0, tb = 0.0, taa = 0.0, tbb = 0.0, tab = 0.0;
      for (int i = 0; i < tiles; ++i) {
        const size_t t = static_cast<size_t>(i);
        ta += fused ? fa[t] : static_cast<double>(sa[t]);
        tb += fused ? fb[t] : static_cast<double>(sb[t]);
        taa += fused ? faa[t] : static_cast<double>(saa[t]);
        tbb += fused ? fbb[t] : static_cast<double>(sbb[t]);
        tab += fused ? fab[t] : static_cast<double>(sab[t]);
      }
      *whole = zncc_from_sums(
        static_cast<double>(roi.width) * roi.height, ta, tb, taa, tbb, tab);
    }
  }

  double road_score(
    const GroundModel & model, const cv::Mat & previous, const cv::Mat & current,
    double hop, double turn, const cv::Rect & roi) const
  {
    std::vector<double> tiles;
    road_scores(model, previous, current, hop, turn, roi, tiles);
    return tiles.empty() ? -2.0 : *std::max_element(tiles.begin(), tiles.end());
  }

  // The step the road itself says, scored photometrically rather than by vote.
  //
  // `measure_step` scans the same one-parameter family -- the plane-induced
  // homography, in which the vehicle's step is the only unknown once the mount
  // and the turn are known -- but scores each candidate by counting tracked
  // corners that agree. Past about 1.9 m the front camera has no corners to
  // count: the ground samples at 10-37 mm per pixel out there and a road corner
  // is not individually identifiable, which is what Lovegrove reported in 2011
  // and why that camera uses a third of the band it is given. A photometric
  // score needs no corner. It asks whether the whole region lands on itself,
  // which is the one question thousands of ambiguous pixels can answer together.
  double measure_step_photometric(
    const GroundModel & model, const cv::Mat & previous, const cv::Mat & current,
    double turn, double reach, double centre, bool have_centre,
    const std::array<double, 4> & band, double predicted_error = -1.0,
    int * votes = nullptr, double * peak_score = nullptr,
    double * spread = nullptr) const
  {
    const cv::Rect roi = cv::Rect(
      cv::Point(cvRound(band[0] * current.cols), cvRound(band[1] * current.rows)),
      cv::Point(cvRound(band[2] * current.cols), cvRound(band[3] * current.rows))) &
      cv::Rect(0, 0, current.cols, current.rows);
    if (roi.width < 32 || roi.height < 32) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double span = std::max(reach, 1e-3);
    // Each candidate is warped once; the whole-region score and the per-tile
    // scores both come out of that one warp, so keeping the tiles here costs
    // nothing and saves the separate spread sweep it used to need -- a third of
    // the search's warps.
    const int tiles = std::max(road_step_tiles_, 1);
    std::vector<double> tile_best(static_cast<size_t>(tiles), 0.0);
    std::vector<double> tile_value(static_cast<size_t>(tiles), -2.0);
    std::vector<double> scores;
    // The pair the scan runs against, and the region on it. The fine pass
    // always uses the frames as given; the coarse pass may use a smaller copy.
    const auto scan = [&](const cv::Mat & from, const cv::Mat & to,
        const cv::Rect & over, double low, double high, int samples,
        bool keep_tiles, double & best_score) {
        double best = low;
        best_score = -2.0;
        for (int i = 0; i < samples; ++i) {
          const double s = low + (high - low) * i / std::max(samples - 1, 1);
          double whole = -2.0;
          road_scores(model, from, to, s * span, turn, over, scores, &whole);
          if (whole > best_score) {
            best_score = whole;
            best = s;
          }
          if (!keep_tiles) {
            continue;
          }
          for (int j = 0; j < tiles && j < static_cast<int>(scores.size()); ++j) {
            if (scores[static_cast<size_t>(j)] > tile_value[static_cast<size_t>(j)]) {
              tile_value[static_cast<size_t>(j)] = scores[static_cast<size_t>(j)];
              tile_best[static_cast<size_t>(j)] = s;
            }
          }
        }
        return best;
      };
    // A bracket that follows the last answer, so the grid stays fine relative
    // to the step at any speed. A fixed span over every plausible speed makes
    // the spacing 2% of the step at 8 m/s and 9% at 1.6, and the coarser
    // spacing biases the parabola.
    //
    // 35% of the step is a fixed guess at how wrong the prediction can be. When
    // `road_step_bracket_k` is set the bracket is instead this many times the
    // *measured* prediction error, which the caller carries as an average of
    // |answer - prediction| over recent frames. A prediction that has been good
    // gets a narrow bracket and the same sample count resolves it finer; a
    // prediction that has been bad widens on its own. `predicted_error` is
    // negative when the caller has no estimate yet, which keeps the old width.
    double peak = 0.0;
    const double width_fraction =
      (road_step_bracket_k_ > 0.0 && predicted_error >= 0.0)
      ? std::min(std::max(road_step_bracket_k_ * predicted_error, 0.02), 0.35)
      : 0.35;
    // The bracket sits on the side the prediction is on, with zero at its near
    // edge rather than inside it. Zero has to stay out of reach: features that
    // failed to track all agree on it, so the vote is bimodal and a window that
    // contains zero finds the wrong mode.
    //
    // It used to be written as `max(0.0, centre - reachable)`, which keeps zero
    // out only while the step is positive. Reversing it cannot describe at all,
    // and `centre` is an average of this function's own answers, so once the
    // answer is positive it can never become negative again -- a lock, not a
    // prior. Measured on the obstacle park drive, whose path is 56% reverse:
    // the forward stretches read 1.000 of truth and the reverse ones 0.265,
    // which is the whole of that drive's 5 m final error.
    const double magnitude = std::abs(centre);
    const double reachable = have_centre ? std::max(width_fraction * magnitude, 0.01) : 0.0;
    // The direction comes from the corner vote, which is not sign-locked; this
    // search only resolves how far. Without `road_step_reverse` the vote never
    // reports a negative and this is the sign of `centre`, as before.
    double toward = centre < 0.0 ? -1.0 : 1.0;
    if (road_step_reverse_) {
      std::lock_guard<std::mutex> guard(step_lock_);
      if (step_ready_ && std::abs(shared_step_) > 1e-9) {
        toward = shared_step_ < 0.0 ? -1.0 : 1.0;
      }
    }
    const double near_edge = toward * std::max(magnitude - reachable, 0.0);
    const double far_edge = toward * (magnitude + reachable);
    const double low = have_centre ? std::min(near_edge, far_edge) : -0.05;
    const double high = have_centre ? std::max(near_edge, far_edge) : 0.60;
    // The coarse pass locates the peak and the fine pass resolves it, and those
    // are different jobs: locating needs range, resolving needs precision. Only
    // the second needs the pixels. `road_step_coarse_divisor` runs the first on
    // a smaller copy of the pair -- a quarter of the area at 2 -- which is what
    // makes a full-resolution fine pass affordable. The tiles are taken from the
    // fine pass only, so the kerb signal keeps the resolution it was measured at.
    const int divisor = std::max(road_step_coarse_divisor_, 1);
    double coarse = 0.0;
    if (divisor > 1 && previous.cols / divisor >= 64 && roi.width / divisor >= 32 &&
      roi.height / divisor >= 32)
    {
      cv::Mat small_from;
      cv::Mat small_to;
      const cv::Size size(previous.cols / divisor, previous.rows / divisor);
      cv::resize(previous, small_from, size, 0, 0, cv::INTER_AREA);
      cv::resize(current, small_to, size, 0, 0, cv::INTER_AREA);
      const cv::Rect small_roi =
        cv::Rect(roi.x / divisor, roi.y / divisor, roi.width / divisor, roi.height / divisor) &
        cv::Rect(0, 0, size.width, size.height);
      coarse = scan(
        small_from, small_to, small_roi, low, high, road_step_samples_, false, peak);
    } else {
      coarse = scan(previous, current, roi, low, high, road_step_samples_, true, peak);
    }
    const double spacing = (high - low) / std::max(road_step_samples_ - 1, 1);
    double fine_peak = 0.0;
    double found = scan(
      previous, current, roi, coarse - spacing, coarse + spacing,
      road_step_samples_, true, fine_peak);
    // The sign is not decided here. Three attempts were made to read it off the
    // photometric score -- take the mirrored bracket when it scores higher,
    // when it scores higher three frames running, and only when the bracket
    // already touches zero -- and all three flipped forward drives at random.
    // nv2/nv3/nv4 are three recordings of one condition and came out +59%,
    // +0.2%, +26% under the last of them, which is what a coin looks like.
    //
    // It cannot work. Near a stop the two warps are nearly the same picture, so
    // the score does not separate them -- and the score is what this search
    // maximises. What does separate them is the corner vote: a homography with
    // the wrong sign puts every tracked corner on the wrong side and the inlier
    // count collapses. So `measure_step` owns the direction and this owns the
    // magnitude, which is the division the rest of the stack already uses.
    // What each across-track tile would have answered, at no extra warp: the
    // region is warped once per candidate either way. The answer stays the
    // whole region's -- splitting it costs precision on the 92% of frames that
    // have nothing wrong with them, measured as ATE 0.0581 -> 0.0705. The
    // spread of the tiles is kept instead, as the one signal that separates a
    // frame with a kerb in it from a frame without: the score cannot, because
    // the intruder scores *higher* than the road.
    if (tiles > 1 && spread != nullptr) {
      const auto range = std::minmax_element(tile_best.begin(), tile_best.end());
      *spread = (*range.second - *range.first) / std::max(std::abs(found), 1e-6);
    }
    const double fine = 2.0 * spacing / std::max(road_step_samples_ - 1, 1);
    const double left = road_score(model, previous, current, (found - fine) * span, turn, roi);
    const double right = road_score(model, previous, current, (found + fine) * span, turn, roi);
    if (peak_score != nullptr) {
      *peak_score = fine_peak;
    }
    const double curve = left - 2.0 * fine_peak + right;
    if (std::abs(curve) > 1e-9) {
      const double shift = 0.5 * (left - right) / curve;
      if (std::abs(shift) <= 1.0) {
        found += shift * fine;
      }
    }
    return found;
  }

  // Zero mean, unit norm. The squared distance between two regions normalised
  // this way is exactly 2(1 - ZNCC), which turns the score the search maximises
  // into a residual a least-squares step can descend.
  static bool normalise(const cv::Mat & patch, cv::Mat & out)
  {
    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(patch, mean, deviation);
    const double norm = deviation[0] * std::sqrt(static_cast<double>(patch.total()));
    if (!(norm > 1e-9)) {
      return false;
    }
    patch.convertTo(out, CV_32F, 1.0 / norm, -mean[0] / norm);
    return true;
  }

  // The fit's Jacobian without warping for it.
  //
  // A column is d(patch)/d(parameter). The patch is the previous frame sampled
  // at v(p) = pixel(H(p)^-1 bearing(u)), so the chain is the image gradient at
  // v, times how v moves with the bearing, times how the bearing moves with the
  // matrix. Two of those three the model can state exactly; the third is a
  // Sobel over the region, taken once.
  //
  // Numerically this replaces two warps per freed parameter -- eight of the
  // 14.5 warps a solve spends, and a warp at the lattice the precision needs is
  // the expensive thing here.
  //
  // `normalise` divides the patch by its own norm, and that norm changes with
  // the parameters too. The derivative of the normalised patch carries a second
  // term for it, which is what keeps this equal to the difference of two
  // normalised patches rather than to the difference of two raw ones.
  bool analytic_columns(
    const GroundModel & model, const cv::Mat & previous, const cv::Rect & roi,
    const double * at, int freedom, cv::Mat * column) const
  {
    cv::Mat map_x;
    cv::Mat map_y;
    build_warp_roi(
      model, at[0], at[1], previous.cols, previous.rows, roi, map_x, map_y,
      at[2], at[3], road_step_fit_stride_);
    cv::Mat warped;
    cv::remap(previous, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    // The gradient wanted is the source image's, read where the warp lands --
    // not the gradient of the warped image, which is that same gradient already
    // multiplied by the warp's own spatial Jacobian. Taking the second is what
    // makes the columns disagree with the difference of two warps by 66 to 78%.
    // Computed here rather than cached. It was cached, keyed on `previous.data`
    // and held in one pair of buffers for the whole node -- which is two
    // cameras sharing them, and an allocator free to hand the same address to a
    // different frame. That is the same defect `road_cuda` was removed for, and
    // it is not worth the two Sobels it saves against the warps this replaces.
    cv::Mat source_x;
    cv::Mat source_y;
    cv::Sobel(previous, source_x, CV_32F, 1, 0, 3, 1.0 / 8.0);
    cv::Sobel(previous, source_y, CV_32F, 0, 1, 3, 1.0 / 8.0);
    cv::Mat gx;
    cv::Mat gy;
    cv::remap(source_x, gx, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::remap(source_y, gy, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(warped, mean, deviation);
    const double norm = deviation[0] * std::sqrt(static_cast<double>(warped.total()));
    if (!(norm > 1e-9)) {
      return false;
    }
    cv::Mat patch;
    warped.convertTo(patch, CV_64F, 1.0 / norm, -mean[0] / norm);

    const cv::Matx33d h = model.homography(at[0], at[1], at[2], at[3], arc_hop_);
    const cv::Matx33d inverse = h.inv();
    cv::Matx33d derivative[4];
    model.homography_jacobian(at[0], at[1], at[2], at[3], derivative);

    const int rows = roi.height;
    const int cols = roi.width;
    std::vector<cv::Mat> raw(static_cast<size_t>(freedom));
    for (int k = 0; k < freedom; ++k) {
      raw[static_cast<size_t>(k)] = cv::Mat::zeros(rows, cols, CV_64F);
    }
    for (int y = 0; y < rows; ++y) {
      const float * rx = gx.ptr<float>(y);
      const float * ry = gy.ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        const cv::Vec3d b = model.bearing(
          cv::Point2f(static_cast<float>(roi.x + x), static_cast<float>(roi.y + y)),
          previous.cols, previous.rows);
        const cv::Vec3d w = inverse * b;
        double pj[2][3];
        model.pixel_jacobian(w, previous.cols, previous.rows, pj);
        for (int k = 0; k < freedom; ++k) {
          // d(H^-1)/dp = -H^-1 (dH/dp) H^-1, so d(w)/dp = -H^-1 (dH/dp) w.
          const cv::Vec3d dw = -(inverse * (derivative[k] * w));
          const double dx = pj[0][0] * dw[0] + pj[0][1] * dw[1] + pj[0][2] * dw[2];
          const double dy = pj[1][0] * dw[0] + pj[1][1] * dw[1] + pj[1][2] * dw[2];
          raw[static_cast<size_t>(k)].at<double>(y, x) = rx[x] * dx + ry[x] * dy;
        }
      }
    }
    // The normalisation's own derivative. With p = (I - mean)/norm, moving the
    // patch by dI changes both, and dp = (dI - mean(dI))/norm - p * <p, dI>/norm.
    for (int k = 0; k < freedom; ++k) {
      cv::Mat & r = raw[static_cast<size_t>(k)];
      const double average = cv::mean(r)[0];
      const double along = patch.dot(r);
      cv::Mat out = (r - average) / norm - patch * (along / norm);
      out.convertTo(column[k], CV_32F);
    }
    return true;
  }

  // The region warped by one candidate (step, yaw, pitch, roll), normalised.
  bool road_patch(
    const GroundModel & model, const cv::Mat & previous, const cv::Rect & roi,
    const double * candidate, cv::Mat & out) const
  {
    // The fit's own lattice. The search only has to find which candidate is
    // nearest the peak, and a coarse warp does that; this is where the answer
    // that leaves the node is actually formed, so it is where the lattice
    // error lands. Measured over the whole tracker, stride 8 to 1 moves the
    // median hop error 0.278 mm to 0.174 -- and fusing the search's warp alone,
    // lattice and all removed, moved nothing, because the ESM overwrites the
    // search's answer with its own.
    ++patch_warps_;
    cv::Mat map_x;
    cv::Mat map_y;
    build_warp_roi(
      model, candidate[0], candidate[1], previous.cols, previous.rows, roi,
      map_x, map_y, candidate[2], candidate[3], road_step_fit_stride_);
    cv::Mat warped;
    cv::remap(previous, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return normalise(warped, out);
  }

  // What the plane fit could not explain, on a coarse grid.
  //
  // Warp the previous frame forward by the motion the fit settled on, then ask
  // dense flow what is still moving. A point on the road is put back exactly
  // and contributes nothing; a point at height z above it is left short by its
  // parallax, and that residual is what carries the height. Output rows are
  // (x, y, dx, dy) in pixels of the full frame.
  void measure_parallax(
    const GroundModel & model, const cv::Mat & previous, const cv::Mat & current,
    double step, double turn, double pitch, double roll,
    const std::array<double, 4> & band, std::vector<double> & out) const
  {
    out.clear();
    if (parallax_grid_ <= 0 || previous.empty() || current.empty()) {
      return;
    }
    // The same band the step was solved on, so the plane part removed here is
    // the plane part that was fitted there.
    const cv::Rect roi = cv::Rect(
      cv::Point(cvRound(band[0] * current.cols), cvRound(band[1] * current.rows)),
      cv::Point(cvRound(band[2] * current.cols), cvRound(band[3] * current.rows))) &
      cv::Rect(0, 0, current.cols, current.rows);
    if (roi.width < 16 || roi.height < 16) {
      return;
    }
    cv::Mat map_x;
    cv::Mat map_y;
    build_warp_roi(
      model, step, turn, previous.cols, previous.rows, roi, map_x, map_y, pitch, roll);
    cv::Mat warped;
    cv::remap(previous, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    const cv::Mat target = current(roi);
    // Half size by default: the residual is a smooth field over a region the
    // fit has already flattened, and the flow is the expensive part.
    cv::Mat small_warped;
    cv::Mat small_target;
    const double scale = std::clamp(parallax_flow_scale_, 0.1, 1.0);
    cv::resize(warped, small_warped, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::resize(target, small_target, cv::Size(), scale, scale, cv::INTER_AREA);
    if (small_warped.size() != small_target.size() ||
      small_warped.cols < 8 || small_warped.rows < 8)
    {
      return;
    }
    if (!parallax_flow_) {
      parallax_flow_ = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
    }
    cv::Mat flow;
    parallax_flow_->calc(small_warped, small_target, flow);
    // DIS returns two-channel float, but a failed call leaves whatever was
    // there. Reading `.at<Point2f>` out of that is how a residual of 1e14 gets
    // into the message.
    if (flow.type() != CV_32FC2 || flow.size() != small_warped.size()) {
      return;
    }
    const int cells = parallax_grid_;
    out.reserve(static_cast<size_t>(cells) * cells * 4);
    for (int row = 0; row < cells; ++row) {
      for (int column = 0; column < cells; ++column) {
        const int fx = std::min(
          flow.cols - 1, static_cast<int>((column + 0.5) * flow.cols / cells));
        const int fy = std::min(
          flow.rows - 1, static_cast<int>((row + 0.5) * flow.rows / cells));
        const cv::Point2f d = flow.at<cv::Point2f>(fy, fx);
        if (!std::isfinite(d.x) || !std::isfinite(d.y)) {
          continue;
        }
        // Back to full-frame pixels, and to where in the frame this cell sits.
        out.push_back(roi.x + (fx + 0.5) / scale);
        out.push_back(roi.y + (fy + 0.5) / scale);
        out.push_back(d.x / scale);
        out.push_back(d.y / scale);
      }
    }
  }

  // The same photometric cost, with three more unknowns freed.
  //
  // The search above holds the rotation at what the gyro said and the plane at
  // where the mount says it is, and asks only how far the vehicle stepped. The
  // induced homography H = R + t n^T/d has more in it than that: R is a full
  // body rotation over the hop, and freeing its pitch and roll alongside its
  // yaw gives one algorithm that measures all four from the same pixels -- the
  // step, and the three angles the band split only reaches two of. The camera
  // height is not among them, and not for want of asking: a scale change of the
  // whole scene about the lens leaves every bearing exactly where it was, so
  // no photometric cost anywhere can see `d`.
  //
  // Levenberg-Marquardt rather than a bare Gauss-Newton, because the four are
  // correlated by construction. A nose-down pitch and a longer step move the
  // near ground the same way and only the far ground tells them apart, so the
  // normal matrix is near-singular in that direction and an undamped step walks
  // out of the basin on the first iteration.
  //
  // Seeded from the search's own answer and the gyro's turn, and it keeps them
  // if it cannot beat them: the caller falls back to the one-dimensional
  // answer, which is the measurement this stack already trusts.
  RoadSolve solve_step_esm(
    const GroundModel & model, const cv::Mat & previous, const cv::Mat & current,
    double turn, double seed_step, const std::array<double, 4> & band) const
  {
    RoadSolve out;
    ++fit_solves_;
    const cv::Rect roi = cv::Rect(
      cv::Point(cvRound(band[0] * current.cols), cvRound(band[1] * current.rows)),
      cv::Point(cvRound(band[2] * current.cols), cvRound(band[3] * current.rows))) &
      cv::Rect(0, 0, current.cols, current.rows);
    if (roi.width < 32 || roi.height < 32) {
      return out;
    }
    cv::Mat target;
    if (!normalise(current(roi), target)) {
      return out;
    }
    // The geometry that turns a parameter into pixels, taken exactly as the
    // band sweep takes it.
    const double lens = std::abs(model.translation_base_from_camera[2]);
    const double focal = model.fx * current.cols /
      std::max(model.calibration_width, 1);
    const double rbar = band_range(model, band, current.cols, current.rows);
    if (!std::isfinite(rbar) || !(lens > 1e-6) || !(focal > 1.0)) {
      return out;
    }
    const double px_per_m = focal * lens / (rbar * rbar + lens * lens);
    if (!(px_per_m > 1e-6)) {
      return out;
    }
    // Perturbations for the numerical Jacobian, sized in pixels of image motion
    // and never as a fraction of the parameter. A metre of step at one metre of
    // range moves the picture f h/(R^2+h^2) = 131 px and at five metres a
    // fifteenth of that, so a percentage asks a different question at every
    // speed and every range -- the trap the band sweep already documents. Half
    // a pixel is the floor worth using: cv::remap quantises its map to 1/32 of
    // a pixel, so half a pixel spans sixteen of those steps and the central
    // difference is measuring the image rather than the quantiser. An angle
    // moves an equidistant fisheye by at most f d(angle), which is where the
    // three angular probes come from.
    const double probe_px = 0.5;
    const double probe[4] = {
      probe_px / px_per_m, probe_px / focal, probe_px / focal, probe_px / focal};
    // How many of the four the fit is allowed to move. The Jacobian is numeric
    // and central, so each freed parameter costs two warps an iteration -- four
    // of them is eight warps against the one the trial step needs, and that
    // ratio is the whole cost of this routine.
    //
    // The three angles are not consumed anywhere: `esm_attitude` and
    // `esm_yaw_source` are both false by default and neither is set in the
    // deployed file, so yaw, pitch and roll are solved for, published, ingested
    // and accumulated by the estimator, and then discarded. Only `step` leaves
    // this function into the answer. That does NOT make them free to drop --
    // they may be absorbing model error the step would otherwise take as scale
    // -- so this is a switch to be measured, not a cleanup. 1 solves the step
    // alone at a quarter of the Jacobian; 4 is what every recorded number came
    // from.
    const int freedom = std::min(std::max(road_step_esm_dof_, 1), 4);
    // How far the fit may travel from where the search left it. The search's
    // own bracket is 35% of the step, and an answer outside it is not a
    // refinement of that answer; the angles are bounded by what one hop can
    // physically hold, about three degrees at 30 ms.
    const double step_limit = 0.35 * std::max(std::abs(seed_step), 0.01);
    const double angle_limit = 0.05;

    double at[4] = {seed_step, turn, 0.0, 0.0};
    cv::Mat patch;
    if (!road_patch(model, previous, roi, at, patch)) {
      return out;
    }
    cv::Mat residual = patch - target;
    double cost = residual.dot(residual);
    const double seeded = cost;
    double lambda = 1e-3;
    int taken = 0;
    cv::Mat column[4];
    cv::Matx44d held_normal;
    bool have_jacobian = false;
    for (int iteration = 0; iteration < 10; ++iteration) {
      // How often the Jacobian is rebuilt.
      //
      // It is numeric and central, so each freed parameter costs two warps and
      // four of them cost eight -- against the one warp a trial step needs.
      // That is the whole cost of this routine, and it is spent re-deriving
      // something that barely moves: the fit travels at most 35% of the step
      // and three hundredths of a radian, over which the derivative of the
      // warp is nearly constant. Holding it is what inverse-compositional does
      // structurally, without needing the analytic form.
      //
      // The warps themselves have to be exact -- see `road_patch` -- and an
      // exact warp is expensive, so doing an eighth as many of them is where
      // the precision becomes affordable.
      const bool rebuild = !have_jacobian ||
        road_step_esm_rebuild_ <= 1 ||
        (iteration % road_step_esm_rebuild_) == 0;
      bool built = true;
      if (rebuild && (road_step_esm_analytic_ || esm_compare_)) {
        cv::Mat exact[4];
        built = analytic_columns(model, previous, roi, at, freedom, exact);
        if (built && esm_compare_) {
          // Both, once, so the derivation is checked against the thing it
          // replaces on the quantity that is actually used.
          for (int k = 0; k < freedom; ++k) {
            double up[4] = {at[0], at[1], at[2], at[3]};
            double down[4] = {at[0], at[1], at[2], at[3]};
            up[k] += probe[k];
            down[k] -= probe[k];
            cv::Mat ahead;
            cv::Mat behind;
            if (!road_patch(model, previous, roi, up, ahead) ||
              !road_patch(model, previous, roi, down, behind))
            {
              continue;
            }
            const cv::Mat numeric = (ahead - behind) / (2.0 * probe[k]);
            const double na = cv::norm(numeric);
            const double diff = cv::norm(numeric, exact[k]);
            if (na > 1e-12) {
              esm_compare_worst_[k] = std::max(esm_compare_worst_[k], diff / na);
              esm_compare_sum_[k] += diff / na;
              ++esm_compare_count_[k];
            }
          }
        }
        if (built && road_step_esm_analytic_) {
          for (int k = 0; k < freedom; ++k) {
            column[k] = exact[k];
          }
        } else if (built) {
          built = false;  // fall through to the numeric loop
        }
      }
      for (int k = 0; k < freedom && built && rebuild && !road_step_esm_analytic_; ++k) {
        double up[4] = {at[0], at[1], at[2], at[3]};
        double down[4] = {at[0], at[1], at[2], at[3]};
        up[k] += probe[k];
        down[k] -= probe[k];
        cv::Mat ahead;
        cv::Mat behind;
        built = road_patch(model, previous, roi, up, ahead) &&
          road_patch(model, previous, roi, down, behind);
        if (built) {
          column[k] = (ahead - behind) / (2.0 * probe[k]);
        }
      }
      if (!built) {
        break;
      }
      if (rebuild) {
        have_jacobian = true;
        ++fit_jacobians_;
      }
      if (!have_jacobian) {
        break;
      }
      cv::Matx44d normal = cv::Matx44d::zeros();
      cv::Vec4d gradient(0.0, 0.0, 0.0, 0.0);
      for (int i = 0; i < freedom; ++i) {
        gradient[i] = -column[i].dot(residual);
        for (int j = i; j < freedom; ++j) {
          normal(i, j) = column[i].dot(column[j]);
          normal(j, i) = normal(i, j);
        }
      }
      // A frozen parameter is held by an identity row rather than by shrinking
      // the system: its gradient is zero and its diagonal is one, so Cholesky
      // returns exactly zero for it and the 4x4 algebra below is untouched.
      for (int i = freedom; i < 4; ++i) {
        normal(i, i) = 1.0;
      }
      cv::Vec4d delta(0.0, 0.0, 0.0, 0.0);
      bool accepted = false;
      for (int attempt = 0; attempt < 6 && !accepted; ++attempt) {
        cv::Matx44d damped = normal;
        for (int i = 0; i < 4; ++i) {
          damped(i, i) += lambda * std::max(normal(i, i), 1e-12);
        }
        cv::Vec4d proposal;
        if (cv::solve(damped, gradient, proposal, cv::DECOMP_CHOLESKY)) {
          const double trial[4] = {
            at[0] + proposal[0], at[1] + proposal[1],
            at[2] + proposal[2], at[3] + proposal[3]};
          const bool inside =
            std::abs(trial[0] - seed_step) <= step_limit &&
            std::abs(trial[1] - turn) <= angle_limit &&
            std::abs(trial[2]) <= angle_limit && std::abs(trial[3]) <= angle_limit;
          cv::Mat moved_patch;
          if (inside && road_patch(model, previous, roi, trial, moved_patch)) {
            cv::Mat next = moved_patch - target;
            const double trial_cost = next.dot(next);
            if (trial_cost < cost) {
              std::copy(trial, trial + 4, at);
              residual = next;
              cost = trial_cost;
              delta = proposal;
              accepted = true;
              ++taken;
            }
          }
        }
        if (!accepted) {
          lambda = std::min(lambda * 10.0, 1e6);
        }
      }
      if (!accepted) {
        break;
      }
      lambda = std::max(lambda * 0.3, 1e-9);
      // Converged when the last accepted step moved the picture by a hundredth
      // of a pixel, which is well under what the map's own quantisation can
      // represent -- past there the iteration is polishing rounding.
      const double moved = std::abs(delta[0]) * px_per_m +
        (std::abs(delta[1]) + std::abs(delta[2]) + std::abs(delta[3])) * focal;
      if (moved < 0.01) {
        break;
      }
    }
    if (taken == 0 || !(cost < seeded)) {
      return out;
    }
    out.step = at[0];
    out.yaw = at[1];
    out.pitch = at[2];
    out.roll = at[3];
    out.score = 1.0 - 0.5 * cost;
    out.iterations = taken;
    out.ok = true;
    return out;
  }

  // The same one-parameter family, but each across-track tile answers on its
  // own and the median of their answers is taken. Coarse then fine, as the
  // single-region path does, so precision is not traded for the robustness.
  double measure_step_tiled(
    const GroundModel & model, const cv::Mat & previous, const cv::Mat & current,
    double turn, double span, double centre, bool have_centre, const cv::Rect & roi,
    double * peak_score) const
  {
    const int tiles = road_step_tiles_;
    const double reachable = have_centre ? std::max(0.35 * centre, 0.01) : 0.0;
    const double low = have_centre ? std::max(0.0, centre - reachable) : -0.05;
    const double high = have_centre ? centre + reachable : 0.60;
    const auto sweep = [&](double from, double to, int samples,
        std::vector<double> & best, std::vector<double> & value) {
        best.assign(static_cast<size_t>(tiles), from);
        value.assign(static_cast<size_t>(tiles), -2.0);
        std::vector<double> scores;
        for (int i = 0; i < samples; ++i) {
          const double s = from + (to - from) * i / std::max(samples - 1, 1);
          road_scores(model, previous, current, s * span, turn, roi, scores);
          for (int j = 0; j < tiles && j < static_cast<int>(scores.size()); ++j) {
            if (scores[static_cast<size_t>(j)] > value[static_cast<size_t>(j)]) {
              value[static_cast<size_t>(j)] = scores[static_cast<size_t>(j)];
              best[static_cast<size_t>(j)] = s;
            }
          }
        }
      };
    std::vector<double> coarse;
    std::vector<double> ignored;
    sweep(low, high, road_step_samples_, coarse, ignored);
    const double spacing = (high - low) / std::max(road_step_samples_ - 1, 1);
    std::vector<double> answers;
    answers.reserve(static_cast<size_t>(tiles));
    double peak = -2.0;
    for (int j = 0; j < tiles; ++j) {
      std::vector<double> fine;
      std::vector<double> value;
      const double at = coarse[static_cast<size_t>(j)];
      sweep(at - spacing, at + spacing, road_step_samples_, fine, value);
      answers.push_back(fine[static_cast<size_t>(j)]);
      peak = std::max(peak, value[static_cast<size_t>(j)]);
    }
    if (peak_score != nullptr) {
      *peak_score = peak;
    }
    std::sort(answers.begin(), answers.end());
    const size_t half = answers.size() / 2;
    return answers.size() % 2 == 1
           ? answers[half] : 0.5 * (answers[half - 1] + answers[half]);
  }

  // Horizontal range from the lens to the ground under one pixel.
  double ground_range(
    const GroundModel & model, const cv::Point2f & at, int width, int height) const
  {
    const cv::Vec3d b = model.rotation_base_from_camera * model.bearing(at, width, height);
    const double z = model.translation_base_from_camera[2];
    if (!(z > 1e-6) || b[2] >= -1e-9) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double t = -z / b[2];
    return t * std::hypot(b[0], b[1]);
  }

  // Mean range over a band's rows, at its centre column.
  double band_range(
    const GroundModel & model, const std::array<double, 4> & band, int width,
    int height) const
  {
    const double x = 0.5 * (band[0] + band[2]) * width;
    double total = 0.0;
    int kept = 0;
    for (int i = 0; i < 5; ++i) {
      const double y = (band[1] + (band[3] - band[1]) * (i + 0.5) / 5.0) * height;
      const double r = ground_range(
        model, cv::Point2f(static_cast<float>(x), static_cast<float>(y)), width, height);
      if (std::isfinite(r)) {
        total += r;
        ++kept;
      }
    }
    return kept > 0 ? total / kept : std::numeric_limits<double>::quiet_NaN();
  }

  // Mean lateral offset over a band's columns, at its centre row, in base_link.
  // Roll tilts the road sideways, so the effective height at lateral offset y
  // is h + y phi and every range there scales by h / (h + y phi). What
  // separates the two sides is therefore where they sit across the vehicle,
  // exactly as range separates near from far. Taken in the body frame, not the
  // image, so the rear mount's yaw of 180 degrees comes out in the sign.
  // `axis` 0 is forward, 1 is lateral, both in base_link.
  double band_offset(
    const GroundModel & model, const std::array<double, 4> & band, int width,
    int height, int axis) const
  {
    const double y = 0.5 * (band[1] + band[3]) * height;
    const double z = model.translation_base_from_camera[2];
    double total = 0.0;
    int kept = 0;
    for (int i = 0; i < 5; ++i) {
      const double x = (band[0] + (band[2] - band[0]) * (i + 0.5) / 5.0) * width;
      const cv::Vec3d b = model.rotation_base_from_camera * model.bearing(
        cv::Point2f(static_cast<float>(x), static_cast<float>(y)), width, height);
      if (!(z > 1e-6) || b[2] >= -1e-9) {
        continue;
      }
      total += (-z / b[2]) * b[axis];
      ++kept;
    }
    return kept > 0 ? total / kept : std::numeric_limits<double>::quiet_NaN();
  }

  // The same step, but over the near half of the band and over the far half.
  //
  // Three unknowns set this camera chain's scale: one plane offset shared by
  // both cameras -- the road is a single surface -- and one mounting pitch
  // each. Three observations reach them without truth. A pitch error moves a
  // point at range R by (R^2+h^2)/h and a height error by R, so within one
  // camera the *near against far* disagreement is the pitch:
  //
  //     dtheta = h (s_far/s_near - 1) / (2 (R_far - R_near))
  //
  // and between the two cameras, which measure the same motion from different
  // heights, what is left after removing each one's pitch is the plane offset:
  //
  //     delta = [ (s_f/s_r - 1) - 2 R_f dtheta_f/h_f + 2 R_r dtheta_r/h_r ]
  //             / (1/h_f - 1/h_r)
  //
  // Checked offline against truth on straight_s8_t01: this recovers 3.39 mm
  // where the value fitted to truth was 3.5, and pitches of 0.041 and 0.059
  // degrees. What it cannot reach is a scale error common to both cameras --
  // the same lens model and the same assembly are used for each, so a focal
  // length that is wrong scales every bearing alike and cancels in the ratio.
  // That residual measures -0.133%.
  void measure_step_bands(
    const GroundModel & model, const cv::Mat & previous, const cv::Mat & current,
    double turn, double reach, double centre, const std::array<double, 4> & band,
    RoadBands & out) const
  {
    const double split = 0.5 * (band[1] + band[3]);
    std::array<double, 4> far_band{band[0], band[1], band[2], split};
    std::array<double, 4> near_band{band[0], split, band[2], band[3]};
    // The across-track split, for roll. It cuts the same region the other way,
    // so the two measurements are of one region and share its texture.
    const double middle = 0.5 * (band[0] + band[2]);
    std::array<double, 4> left_band{band[0], band[1], middle, band[3]};
    std::array<double, 4> right_band{middle, band[1], band[2], band[3]};
    // A short sweep either side of the answer the whole band gave; the two
    // halves cannot disagree by much or the warp would not have fitted at all.
    const double span = std::max(reach, 1e-3);
    // The peak's curvature comes out with it. Two bands are being compared by
    // their interpolated peaks, and a parabola's bias depends on how sharp the
    // peak is; sharpness depends on the texture, which differs between the
    // bands and between road surfaces. If the two curvatures differ, part of
    // what reads as pitch is the interpolation and not the geometry.
    double curve_out = 0.0;
    const auto refine = [&](const std::array<double, 4> & use) {
        const cv::Rect roi = cv::Rect(
          cv::Point(cvRound(use[0] * current.cols), cvRound(use[1] * current.rows)),
          cv::Point(cvRound(use[2] * current.cols), cvRound(use[3] * current.rows))) &
          cv::Rect(0, 0, current.cols, current.rows);
        if (roi.width < 16 || roi.height < 8) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        // The disagreement being measured is 0.05 to 0.16 per cent of the
        // step. A grid alone cannot see it -- at nine points across two per
        // cent the spacing is half a per cent and the two halves come back
        // bit-identical, which is what the first version did. The peak has to
        // be interpolated, exactly as the main search does.
        // The window is set by the image, not by the step.
        //
        // A percentage of the step was wrong: the alignment stops matching when
        // the warp has moved the picture by a texture correlation length, which
        // is a few pixels, and that is a fixed distance on the ground whatever
        // the speed. At one metre of range a metre of step moves the image
        // f h/(R^2+h^2) = 131 px, so three pixels is 23 mm -- while one per cent
        // of the step is 2.5 mm at 8 m/s and 0.6 at 2. Sampling a twenty-fifth
        // of the peak puts every sample on the apex: the fitted curvature fell
        // from -0.0066 to -0.0004 between those two speeds, and with it the
        // band comparison that the calibration rests on.
        // How many points across the peak before the parabola is fitted.
        //
        // Nine is what every recorded band number came from. It is not enough
        // at low speed and the failure is a bias, not noise: at 1.8 m/s the six
        // strips disagree by forty standard errors of their own medians while
        // running *non-monotonically* in range, where at 7.5 m/s the same
        // measurement is monotone at 19-50 sigma. A parabola through nine
        // points inherits the peak's asymmetry, and the asymmetry follows each
        // strip's texture.
        //
        // More points cannot fix the underlying ratio -- the quantity is 0.05
        // to 0.16 per cent of the step, which is 0.13-0.4 mm at 7.5 m/s and
        // 0.03-0.1 at 1.8, against a peak about 23 mm wide either way -- but it
        // does reduce the interpolation's own bias, which is what the low-speed
        // profile is made of.
        const int samples = std::max(band_samples_, 3);
        const double rbar = band_range(model, use, current.cols, current.rows);
        const double lens = std::abs(model.translation_base_from_camera[2]);
        const double focal = model.fx * current.cols /
          std::max(model.calibration_width, 1);
        double width = 0.01 * std::max(centre, 0.01);
        if (std::isfinite(rbar) && lens > 1e-6 && focal > 1.0) {
          const double px_per_m = focal * lens / (rbar * rbar + lens * lens);
          if (px_per_m > 1e-6) {
            width = road_step_band_window_px_ / px_per_m / span;
          }
        }
        const double spacing = 2.0 * width / (samples - 1);
        std::vector<double> value(static_cast<size_t>(samples), -2.0);
        int at = 0;
        for (int i = 0; i < samples; ++i) {
          const double s = centre - width + spacing * i;
          std::vector<double> tiles;
          double whole = -2.0;
          road_scores(model, previous, current, s * span, turn, roi, tiles, &whole);
          value[static_cast<size_t>(i)] = whole;
          if (whole > value[static_cast<size_t>(at)]) {
            at = i;
          }
        }
        double best = centre - width + spacing * at;
        curve_out = 0.0;
        if (at > 0 && at + 1 < samples) {
          const double a = value[static_cast<size_t>(at - 1)];
          const double b = value[static_cast<size_t>(at)];
          const double c = value[static_cast<size_t>(at + 1)];
          const double curve = a - 2.0 * b + c;
          curve_out = curve;
          if (std::abs(curve) > 1e-12) {
            const double shift = 0.5 * (a - c) / curve;
            if (std::abs(shift) <= 1.0) {
              best += shift * spacing;
            }
          }
        }
        return best;
      };
    out.far_step = refine(far_band);
    out.far_curve = curve_out;
    out.near_step = refine(near_band);
    out.near_curve = curve_out;
    out.left_step = refine(left_band);
    out.right_step = refine(right_band);
    out.near_range = band_range(model, near_band, current.cols, current.rows);
    out.far_range = band_range(model, far_band, current.cols, current.rows);
    out.left_lateral = band_offset(model, left_band, current.cols, current.rows, 1);
    out.right_lateral = band_offset(model, right_band, current.cols, current.rows, 1);
    // The pitch is set by how far along the vehicle a band sits, not by how far
    // away: a body pitched nose-down meets the ground ahead sooner and the
    // ground astern later, so a rear mount's bands answer with the opposite
    // sign. Taking the longitudinal offset in base_link puts that sign in the
    // geometry instead of in a special case.
    out.near_forward = band_offset(model, near_band, current.cols, current.rows, 0);
    out.far_forward = band_offset(model, far_band, current.cols, current.rows, 0);

    // Diagnostic only, off unless the environment names a file. The two-band
    // split gives one number per camera and cannot separate what that number is
    // made of: a body pitch is common to both cameras and enters with the sign
    // of the band's longitudinal offset, which is +0.39 m on the front and
    // -0.55 on the rear -- so a pitch must show as *opposite* signs. Measured,
    // both cameras read the far half long, so it is not a pitch, and one scale
    // error does not fit either (the two cameras want 0.00051 and 0.00591).
    // What is left is a function of the field angle, and two points per camera
    // cannot draw a curve. This sweeps the region in strips so it can be drawn.
    if (const char * path = std::getenv("MONOSCALE_BAND_PROFILE")) {
      static std::FILE * dump = nullptr;
      static bool opened = false;
      static int frame = 0;
      if (!opened) {
        opened = true;
        dump = std::fopen(path, "w");
        if (dump != nullptr) {
          std::fprintf(dump, "frame,strip,range,forward,step,whole\n");
        }
      }
      if (dump != nullptr) {
        ++frame;
        const int strips = 6;
        for (int i = 0; i < strips; ++i) {
          const double y0 = band[1] + (band[3] - band[1]) * i / strips;
          const double y1 = band[1] + (band[3] - band[1]) * (i + 1) / strips;
          const std::array<double, 4> use{band[0], y0, band[2], y1};
          const double value = refine(use);
          if (!std::isfinite(value)) {
            continue;
          }
          std::fprintf(
            dump, "%d,%d,%.5f,%.5f,%.8f,%.8f\n", frame, i,
            band_range(model, use, current.cols, current.rows),
            band_offset(model, use, current.cols, current.rows, 0),
            value, centre);
        }
        std::fflush(dump);
      }
    }
  }

  // One scalar, voted on by every correspondence this camera kept.
  void measure_step(
    const GroundModel & model, const std::vector<cv::Point2f> & previous,
    const std::vector<cv::Point2f> & current, int width, int height,
    double turn, double reach)
  {
    const size_t count = previous.size();
    std::vector<cv::Vec3d> from(count);
    std::vector<cv::Vec3d> to(count);
    for (size_t i = 0; i < count; ++i) {
      from[i] = model.bearing(previous[i], width, height);
      to[i] = model.bearing(current[i], width, height);
    }
    const double tolerance = std::cos(step_tolerance_deg_ * M_PI / 180.0);
    const auto support = [&](double step) {
        const cv::Matx33d h =
          model.homography(step * std::max(reach, 1e-3), turn, 0.0, 0.0, arc_hop_);
        int agree = 0;
        for (size_t i = 0; i < count; ++i) {
          cv::Vec3d p = h * from[i];
          const double n = cv::norm(p);
          if (n < 1e-12) {
            continue;
          }
          p *= 1.0 / n;
          if (p.dot(to[i]) > tolerance) {
            ++agree;
          }
        }
        return agree;
      };
    // The same sweep, but every candidate has to clear a step of nothing before
    // it may win, and the winner reports how much support it had.
    //
    // The floor is applied per candidate and not only to the winner. A step too
    // small to move a corner past the agreement tolerance collects the same
    // votes zero does -- the tracks that died -- and if only the winner is
    // tested, a candidate a hair above zero passes on one extra vote and drags
    // the average to a stop. Testing each one excludes zero and everything
    // indistinguishable from it by the same rule, without needing to know how
    // wide that neighbourhood is: the support count says.
    const auto scan_counting =
      [&](double low, double high, int samples, int floor, int & best_count) {
        double best = low;
        best_count = -1;
        for (int i = 0; i < samples; ++i) {
          const double s = low + (high - low) * i / std::max(samples - 1, 1);
          const int c = support(s);
          if (c > floor && c > best_count) {
            best = s;
            best_count = c;
          }
        }
        return best;
      };
    const auto scan = [&](double low, double high, int samples) {
        double best = low;
        int best_count = -1;
        for (int i = 0; i < samples; ++i) {
          const double s = low + (high - low) * i / std::max(samples - 1, 1);
          const int c = support(s);
          if (c > best_count) {
            best = s;
            best_count = c;
          }
        }
        return best;
      };
    // Zero is not a candidate. It is the baseline.
    //
    // Every step search in this file used to keep zero out by putting it at the
    // edge of the window -- `max(0.0, centre - slack)` -- which makes one window
    // answer three different questions at once: which way, how far, and whether
    // the answer means anything. They conflict. A window that excludes zero
    // cannot see reverse; one that includes it lets zero win, because features
    // that failed to track all agree on zero and the vote is bimodal. Clamping
    // made reverse unreportable (the obstacle park read its reverse stretches at
    // 0.265 of truth, ATE 2.37 m); unclamping it cost str_1.5 22%, which is the
    // same bimodality returning.
    //
    // Asked separately the three do not conflict:
    //
    //   best_forward   the most support any positive step gets
    //   best_reverse   the most support any negative step gets
    //   zero_support   what a step of nothing gets -- the failed tracks
    //
    // The direction is whichever side has more support, decided on a count and
    // not on a photometric score, because a wrong-signed homography puts every
    // corner on the wrong side while the two warps look nearly alike near a
    // stop. And when neither side clears `zero_support` by
    // `step_observable_margin` the step is not observable this frame: the
    // corners are agreeing that nothing moved because they stopped tracking,
    // not because the vehicle stopped. Holding the last answer is then the
    // honest output, and it is what the window's edge used to accomplish by
    // accident.
    const double slack = step_ready_
      ? std::max(0.45 * std::abs(shared_step_), step_search_slack_)
      : 0.60;
    const double centre = step_ready_ ? std::abs(shared_step_) : 0.30;
    const double reach_low = std::max(centre - slack, 0.0);
    const double reach_high = centre + slack;
    const int samples = step_ready_ ? 49 : 66;
    const int zero_support = support(0.0);
    const int floor = road_step_reverse_
      ? static_cast<int>(std::lround(step_observable_margin_ * zero_support)) : -1;
    int forward_count = -1;
    int reverse_count = -1;
    const double forward =
      scan_counting(reach_low, reach_high, samples, floor, forward_count);
    double reverse = 0.0;
    if (road_step_reverse_) {
      reverse = scan_counting(-reach_high, -reach_low, samples, floor, reverse_count);
    }
    if (forward_count < 0 && reverse_count < 0) {
      // Neither sign beat a step of nothing. The corners are agreeing that the
      // picture did not move because they stopped tracking it, which is not the
      // same as the vehicle standing still and must not be reported as one.
      // The held answer stands.
      return;
    }
    double found = reverse_count > forward_count ? reverse : forward;
    found = scan(found - 0.012, found + 0.012, 25);
    std::lock_guard<std::mutex> guard(step_lock_);
    // How often the direction changes, and how much of the drive is spent going
    // backwards. A drive that never reverses and reports either is misreading
    // the sign, and the counts say how badly.
    const double before = shared_step_;
    const bool had = step_ready_;
    shared_step_ = step_ready_ ? shared_step_ + 0.5 * (found - shared_step_) : found;
    if (had && (before < 0.0) != (shared_step_ < 0.0)) {
      ++sign_flips_;
    }
    ++sign_frames_;
    if (shared_step_ < 0.0) {
      ++sign_negative_;
    }
    step_ready_ = true;
  }

  // Where each pixel of the predicted frame came from in the previous one.
  //
  // Built on a coarse lattice and stretched: the map is smooth -- it is a
  // homography seen through one lens -- so sampling it every eighth pixel and
  // interpolating is well under a tenth of a pixel wrong, and it costs a few
  // thousand trigonometric evaluations instead of a quarter of a million.
  void build_warp(
    const GroundModel & model, double step, double turn, int width, int height,
    cv::Mat & map_x, cv::Mat & map_y) const
  {
    const cv::Matx33d inverse = model.homography(step, turn, 0.0, 0.0, arc_hop_).inv();
    const int stride = 8;
    const int cols = std::max(width / stride, 2);
    const int rows = std::max(height / stride, 2);
    cv::Mat sx(rows, cols, CV_32F);
    cv::Mat sy(rows, cols, CV_32F);
    // Sampled where cv::resize will read them: output pixel i comes from input
    // coordinate (i + 0.5) * cols / width - 0.5, so the lattice has to sit at
    // the matching places or the whole map is stretched by the ratio between
    // the grid's span and the image's -- which at this stride is a percent, and
    // a percent of 640 px is eight pixels of systematic error at the edge.
    const double sx_step = static_cast<double>(width) / cols;
    const double sy_step = static_cast<double>(height) / rows;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const cv::Point2f q(
          static_cast<float>((c + 0.5) * sx_step - 0.5),
          static_cast<float>((r + 0.5) * sy_step - 0.5));
        const cv::Vec3d b = inverse * model.bearing(q, width, height);
        const cv::Point2f p = model.pixel(b, width, height);
        sx.at<float>(r, c) = p.x;
        sy.at<float>(r, c) = p.y;
      }
    }
    cv::resize(sx, map_x, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    cv::resize(sy, map_y, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  }

  // The band this camera measures its step over. Falls back to the node-level
  // one where a camera does not name its own.
  const std::array<double, 4> & band_for(const std::string & name)
  {
    const auto found = road_bands_.find(name);
    return found != road_bands_.end() ? found->second : road_step_roi_;
  }

  std::array<double, 4> load_road_band(const std::string & name)
  {
    return {
      declare_parameter<double>(name + ".road_step_roi_x0", road_step_roi_[0]),
      declare_parameter<double>(name + ".road_step_roi_y0", road_step_roi_[1]),
      declare_parameter<double>(name + ".road_step_roi_x1", road_step_roi_[2]),
      declare_parameter<double>(name + ".road_step_roi_y1", road_step_roi_[3])};
  }

  GroundModel load_ground_model(const std::string & name)
  {
    GroundModel model;
    const auto k = declare_parameter<std::vector<double>>(
      name + ".k", std::vector<double>{});
    const auto rotation = declare_parameter<std::vector<double>>(
      name + ".rotation_base_from_camera", std::vector<double>{});
    const auto translation = declare_parameter<std::vector<double>>(
      name + ".translation_base_from_camera", std::vector<double>{});
    const auto lens = declare_parameter<std::string>(name + ".distortion_model", "equidistant");
    model.calibration_width = declare_parameter<int>(name + ".calibration_width", 0);
    model.calibration_height = declare_parameter<int>(name + ".calibration_height", 0);
    if (k.size() != 9 || rotation.size() != 9 || translation.size() != 3) {
      RCLCPP_WARN(
        get_logger(),
        "motion_prediction is on but %s has no calibration; it will fall back to "
        "the per-feature warm start", name.c_str());
      return model;
    }
    model.fx = k[0]; model.fy = k[4]; model.cx = k[2]; model.cy = k[5];
    model.rotation_base_from_camera = cv::Matx33d(
      rotation[0], rotation[1], rotation[2],
      rotation[3], rotation[4], rotation[5],
      rotation[6], rotation[7], rotation[8]);
    model.translation_base_from_camera =
      cv::Vec3d(translation[0], translation[1], translation[2]);
    model.equidistant = lens != "plumb_bob" && lens != "pinhole";
    // Correction to the mounting pitch, applied to the calibration.
    //
    // The road itself measures this: warp the near band and the far band
    // separately and the two disagree about the step by 2 R dtheta / h, so the
    // ratio of their answers is the pitch error and needs no truth. On the
    // deployed front camera it reads 0.06 to 0.09 degrees, and removing it
    // flattens the bias across range and takes the drive's scale error from
    // 0.254% to 0.077%. Positive is nose down, as in REP-103.
    const double pitch = declare_parameter<double>(name + ".mount_pitch_offset_deg", 0.0);
    if (pitch != 0.0) {
      const double a = pitch * M_PI / 180.0;
      const cv::Matx33d about_y(
        std::cos(a), 0.0, std::sin(a),
        0.0, 1.0, 0.0,
        -std::sin(a), 0.0, std::cos(a));
      model.rotation_base_from_camera = about_y * model.rotation_base_from_camera;
    }
    model.ready = std::abs(model.translation_base_from_camera[2]) > 1e-3 && model.fx > 0.0;
    return model;
  }

  // Heading at an image stamp, interpolated from the instrument. Returns false
  // rather than guessing when the instrument has not covered the interval.
  bool yaw_at(double stamp, double & out)
  {
    std::lock_guard<std::mutex> guard(imu_lock_);
    if (imu_yaw_.size() < 2) {
      return false;
    }
    const auto * before = static_cast<const std::pair<double, double> *>(nullptr);
    const auto * after = static_cast<const std::pair<double, double> *>(nullptr);
    for (const auto & sample : imu_yaw_) {
      if (sample.first <= stamp) {
        before = &sample;
      } else {
        after = &sample;
        break;
      }
    }
    if (before == nullptr && after != nullptr) {
      return false;
    }
    if (after == nullptr) {
      // The newest frame has no sample past it, and never will in time: the bag
      // and the live rig both publish in stamp order, so the image at t arrives
      // before the IMU sample at t + dt. Waiting for one costs every frame --
      // this gate is why an earlier run measured six frames out of six hundred.
      //
      // Extrapolated from the last two instead, and only across one sample
      // interval. At 60 Hz that is 17 ms, over which the yaw acceleration this
      // vehicle carries contributes well under a tenth of a milliradian.
      if (imu_yaw_.size() < 2) {
        return false;
      }
      const auto & last = imu_yaw_.back();
      const auto & prior = *(imu_yaw_.end() - 2);
      const double span = last.first - prior.first;
      const double ahead = stamp - last.first;
      if (!(span > 1e-9) || ahead > 0.05 || ahead < 0.0) {
        return false;
      }
      double step = last.second - prior.second;
      while (step > M_PI) {step -= 2.0 * M_PI;}
      while (step < -M_PI) {step += 2.0 * M_PI;}
      out = last.second + step * ahead / span;
      return true;
    }
    if (after->first - before->first > 0.12) {
      return false;
    }
    const double span = after->first - before->first;
    const double ratio = span > 1e-9 ? (stamp - before->first) / span : 0.0;
    double step = after->second - before->second;
    while (step > M_PI) {step -= 2.0 * M_PI;}
    while (step < -M_PI) {step += 2.0 * M_PI;}
    out = before->second + ratio * step;
    return true;
  }

  void on_imu(const sensor_msgs::msg::Imu & message)
  {
    const auto & q = message.orientation;
    const double yaw = std::atan2(
      2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    std::lock_guard<std::mutex> guard(imu_lock_);
    imu_yaw_.emplace_back(
      message.header.stamp.sec + message.header.stamp.nanosec * 1e-9, yaw);
    while (imu_yaw_.size() > 600) {
      imu_yaw_.pop_front();
    }
  }

  void publish(
    const std::string & name, const sensor_msgs::msg::Image & message,
    const cv::Size & frame,
    const std::vector<cv::Point2f> & previous_points,
    const std::vector<cv::Point2f> & current_points,
    const std::vector<int64_t> & identities, double step, double score,
    double spread, const RoadBands & bands, const RoadSolve & esm, double reach,
    const std::vector<double> & clarity, const std::vector<double> & parallax)
  {
    std_msgs::msg::Float64MultiArray out;
    out.data.reserve(4 + identities.size() * 5);
    // Double, not float. The array is a Float64MultiArray and the identities do
    // not fit a float: road-grid identities start at 1<<40, where the spacing
    // between representable floats is 1<<17, so **every road identity collapsed
    // to the same value** and the whole lattice arrived as one repeated
    // feature. Corner identities stay under 1<<24 and were exact, which is why
    // this hid. The stamp has the same fault waiting: these bags carry small
    // sim-time stamps where a float resolves to half a microsecond, but on
    // wall-clock stamps near 1.8e9 it resolves to 256 seconds.
    out.data.push_back(
      static_cast<double>(message.header.stamp.sec) +
      static_cast<double>(message.header.stamp.nanosec) * 1e-9);
    out.data.push_back(static_cast<double>(identities.size()));
    out.data.push_back(static_cast<double>(frame.width));
    out.data.push_back(static_cast<double>(frame.height));
    for (size_t i = 0; i < identities.size(); ++i) {
      out.data.push_back(static_cast<double>(identities[i]));
      out.data.push_back(previous_points[i].x);
      out.data.push_back(previous_points[i].y);
      out.data.push_back(current_points[i].x);
      out.data.push_back(current_points[i].y);
    }
    // The photometric step this frame, in metres, appended after the feature
    // block. It is a displacement, not a feature, and the estimator has to take
    // it as one -- carrying it as a grid of virtual points delivers nothing,
    // because those identities never enter the anchor map and the map path
    // answers most frames.
    if (std::isfinite(step)) {
      out.data.push_back(step);
      out.data.push_back(score);
      out.data.push_back(spread);
      // The band steps in metres, like the step above them: they leave here as
      // a fraction of the reach the search was scaled by, and a consumer that
      // took them raw would compare two quantities in different units.
      out.data.push_back(bands.near_step * reach);
      out.data.push_back(bands.far_step * reach);
      out.data.push_back(bands.near_curve);
      out.data.push_back(bands.far_curve);
      out.data.push_back(bands.left_step * reach);
      out.data.push_back(bands.right_step * reach);
      // Where those bands sat, so the estimator can turn a disagreement into
      // an angle without knowing this node's region of interest.
      out.data.push_back(bands.near_range);
      out.data.push_back(bands.far_range);
      out.data.push_back(bands.left_lateral);
      out.data.push_back(bands.right_lateral);
      out.data.push_back(bands.near_forward);
      out.data.push_back(bands.far_forward);
      // The three body angles the four-parameter solve freed, in radians over
      // this hop, appended after everything that was already here so the two
      // parsers that index this block positionally keep their offsets. Only
      // when the solve was asked for: with `road_step_esm` off the message is
      // the message it always was, byte for byte. NaN inside that when the fit
      // failed to beat its own seed and the search's answer was kept.
      if (road_step_esm_) {
        out.data.push_back(esm.yaw);
        out.data.push_back(esm.pitch);
        out.data.push_back(esm.roll);
      }
    }
    // The clarity block, last, with its own length after it. Everything before
    // this is unchanged, and a reader that does not know about it stops at the
    // photometric block and never looks here.
    if (clarity.size() == identities.size() && !clarity.empty()) {
      for (const double value : clarity) {
        out.data.push_back(value);
      }
      out.data.push_back(static_cast<double>(clarity.size()));
    }
    // The parallax block, last of all: four values a cell, then the cell count,
    // then a marker. The marker is what tells a reader this tail is parallax
    // and not a longer clarity block, since both describe themselves by length.
    if (!parallax.empty() && parallax.size() % 4 == 0) {
      for (const double value : parallax) {
        out.data.push_back(value);
      }
      out.data.push_back(static_cast<double>(parallax.size() / 4));
      out.data.push_back(kParallaxMarker);
    }
    if (offline_ != nullptr) {
      // Straight into the bag, in the order the frames were processed. The
      // publish path goes through the middleware, which is where the ordering
      // this mode exists to remove would come back in.
      offline_->write(out, publish_topic_[name], rclcpp::Time(out_stamp_));
    } else {
      publishers_[name]->publish(std::move(out));
    }
    {
      // Counted at the writer itself, so the estimator's tally can be
      // compared against it without a third subscriber in the middle
      // changing what the middleware does.
      std::lock_guard<std::mutex> guard(count_mutex_);
      published_[name] += 1;
    }
  }

  int processing_width_;
  bool road_alignment_ = false;
  std::array<double, 4> road_roi_{{0.2, 0.5, 0.8, 1.0}};
  int road_columns_ = 3;
  int road_rows_ = 3;
  int road_iterations_ = 30;
  bool road_homography_ = false;
  int road_max_age_ = 0;
  double road_epsilon_ = 1.0e-4;
  // Virtual road features live in their own identity range so they can never
  // collide with a corner's. 2^40 is far above anything the corner counter can
  // reach -- it mints of the order of a million an hour -- and far below the
  // 2^53 the float64 message carries exactly.
  // Distinctive enough that no pixel, step or identity can be mistaken for it.
  static constexpr double kParallaxMarker = -8.125e7;
  static constexpr int64_t kRoadIdentityBase = 1LL << 40;
  int64_t road_next_identity_ = kRoadIdentityBase;
  int max_features_;
  double quality_level_ = 0.01;
  bool publish_clarity_ = false;
  double skip_bottom_ = 0.0;
  int road_step_coarse_divisor_ = 1;
  double road_step_bracket_k_ = 0.0;
  int road_step_esm_dof_ = 4;
  bool road_step_reverse_ = false;
  bool road_step_fused_ = false;
  int road_step_fit_stride_ = 0;
  int road_step_esm_rebuild_ = 1;
  bool road_step_esm_analytic_ = false;
  int band_samples_ = 9;
  // See `GroundModel::homography`: follow the arc rather than the chord.
  bool arc_hop_ = false;
  bool esm_compare_ = false;
  mutable double esm_compare_worst_[4] = {0.0, 0.0, 0.0, 0.0};
  mutable double esm_compare_sum_[4] = {0.0, 0.0, 0.0, 0.0};
  mutable long esm_compare_count_[4] = {0, 0, 0, 0};
  double step_observable_margin_ = 1.0;
  int parallax_grid_ = 0;
  double parallax_flow_scale_ = 0.5;
  mutable cv::Ptr<cv::DISOpticalFlow> parallax_flow_;
  double min_distance_;
  int window_;
  int levels_;
  double backward_threshold_;
  double error_threshold_;
  double refill_ratio_;
  bool warm_start_;
  bool predict_by_plane_ = false;
  bool predict_from_motion_ = false;
  std::string step_reference_;
  double step_search_slack_ = 0.08;
  double step_tolerance_deg_ = 0.3;
  bool motion_warp_ = false;
  double motion_warp_min_step_ = 0.12;
  bool road_step_photometric_ = false;
  bool road_from_step_ = false;
  bool road_step_calibrate_ = false;
  bool road_step_esm_ = false;
  double road_step_band_window_px_ = 3.0;
  std::array<double, 4> road_step_roi_{0.25, 0.60, 0.75, 1.00};
  std::map<std::string, std::array<double, 4>> road_bands_;
  int road_step_samples_ = 13;
  int road_step_stride_ = 8;
  int road_step_tiles_ = 1;
  std::string road_step_dump_;
  std::mutex road_step_lock_;
  std::FILE * road_step_file_ = nullptr;
  std::unordered_map<std::string, GroundModel> models_;
  // The step the reference camera last measured, shared with the others. The
  // rig is rigid, so one number serves every camera on it.
  mutable std::mutex step_lock_;
  mutable long patch_warps_ = 0;
  mutable long fit_solves_ = 0;
  mutable long fit_jacobians_ = 0;
  long sign_flips_ = 0;
  long sign_frames_ = 0;
  long sign_negative_ = 0;
  double shared_step_ = 0.0;
  bool step_ready_ = false;
  // Heading from the instrument, so the turn in the homography is measured
  // rather than solved for.
  std::mutex imu_lock_;
  std::deque<std::pair<double, double>> imu_yaw_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_;
  bool seed_new_from_cell_ = false;
  float seed_min_flow_ = 8.0f;
  bool use_cuda_ = false;
  int cuda_iterations_ = 30;
  // Whether the GPU path is actually in use, which needs the parameter, an
  // OpenCV that has cudaoptflow, and a device to run on.
  bool cuda_active_ = false;
#ifdef MONOSCALE_TRACKER_HAS_CUDA
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> forward_flow_;
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> backward_flow_;
#endif
  double skip_top_;
  // How many frames each camera actually handed this node, and how many
  // tracked sets went back out. Counted here because a bag recorder running
  // beside a live stack competes for the same CPU and drops what it is meant
  // to be measuring.
  std::map<std::string, int> received_;
  std::map<std::string, double> spent_;
  std::map<std::string, double> held_;
  std::map<std::string, double> shift_;
  std::map<std::string, double> followed_;
  std::map<std::string, double> change_;
  std::map<std::string, double> scale_;
  std::map<std::string, double> pyramid_;
  std::map<std::string, double> flow_;
  std::map<std::string, double> road_;
  std::map<std::string, double> step_;
  std::map<std::string, double> prep_;
  std::map<std::string, double> follow_;
  std::map<std::string, double> trim_;
  std::map<std::string, double> detect_;
  std::map<std::string, double> pub_;
  std::map<std::string, double> raw_;
  std::map<std::string, double> lost_;
  std::map<std::string, double> noisy_;
  std::map<std::string, double> drifted_;
  double frame_budget_ms_ = 14.0;
  int min_features_ = 500;
  int grid_columns_ = 4;
  int grid_rows_ = 3;
  std::string dump_dir_;
  std::map<std::string, int> dumped_;
  rclcpp::Time counted_since_;
  bool counting_ = false;
  std::mutex count_mutex_;
  std::map<std::string, long> published_;
  std::vector<rclcpp::CallbackGroup::SharedPtr> groups_;
  std::map<std::string, TrackState> states_;
  // Set only in offline mode; see `run_offline`.
  rosbag2_cpp::Writer * offline_ = nullptr;
  std::vector<std::string> camera_names_;
  std::vector<std::string> image_topics_;
  std::string imu_topic_;
  rclcpp::Time out_stamp_{0, 0, RCL_ROS_TIME};
  std::map<std::string, std::string> publish_topic_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr>
  publishers_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // `--offline <input-bag> <output-bag>` reads the recording directly instead
  // of waiting for a player to deliver it. Same handlers, fixed order.
  std::string offline_in;
  std::string offline_out;
  for (int i = 1; i + 2 < argc; ++i) {
    if (std::string(argv[i]) == "--offline") {
      offline_in = argv[i + 1];
      offline_out = argv[i + 2];
    }
  }
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--check-jacobian") {
      check = true;
    }
  }
  auto node = std::make_shared<FeatureTracker>();
  if (check) {
    const int code = node->check_jacobian();
    rclcpp::shutdown();
    return code;
  }
  if (!offline_in.empty()) {
    const int code = node->run_offline(offline_in, offline_out);
    rclcpp::shutdown();
    return code;
  }
  // Settable so the count can follow the number of cameras. Four measured
  // the same as two against a 60 Hz replay (front 26-39 Hz, rear 57-60 either
  // way), so the asymmetry between the two cameras is not thread starvation.
  // Each camera already has its own callback group; more threads than that
  // buy nothing here.
  const int threads = static_cast<int>(
    node->declare_parameter<int>("executor_threads", 4));
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), static_cast<size_t>(std::max(threads, 1)));
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
