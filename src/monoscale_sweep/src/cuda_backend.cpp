// The CUDA path: hand the whole match-aggregate-reduce to the monoscale_fast
// kernel that plane_sweep.py already runs, so the C++ node and the python
// reference share the exact arithmetic rather than approximate each other.
//
// The kernel is `plane_sweep_cuda_impl`, declared here with a weak fallback so
// the library links whether or not monoscale_fast is on the link line: a build
// without CUDA gets a stub that reports unavailable, and the Sweep falls back
// to the CPU path in sweep.cpp.

#include "monoscale_sweep/sweep.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <Eigen/Dense>

// Provided by monoscale_fast (src/sweep_kernels.cu) in the GLOBAL namespace,
// with C++ linkage. Declaring these inside namespace monoscale_sweep made the
// compiler reference monoscale_sweep::plane_sweep_cuda_* instead, which have no
// definition -- the weak symbol stayed null and the backend silently fell back
// to CPU. They must be declared at global scope to bind the real kernel.
bool plane_sweep_cuda_available();
void plane_sweep_cuda_impl(
  const float * reference, const float * sources, int source_count,
  const float * homographies, int planes, int rows, int cols,
  int window_x, int window_y, int cost_kind, int road_plane,
  float small_step, float big_step,
  int * best_index, float * best_cost, float * road_cost, float * second_cost,
  float * volume_out,
  const float * geometry, const float * offsets);

namespace monoscale_sweep
{

bool cuda_available()
{
  return plane_sweep_cuda_available();
}

// The 30 floats one source needs, matching zncc_accumulate_fisheye's layout
// and plane_sweep.fisheye_geometry exactly:
//   focal, cx, cy | rotation_base_from_camera[9] | eye[3]
//   | normal[3] | rotation[9] | shift[3]
static void fisheye_geometry(
  const Lens & lens, const Pose5 & reference, const Pose5 & source,
  const Eigen::Vector3d & normal, float * out)
{
  const Eigen::Matrix3d forward = attitude(reference);
  const Eigen::Matrix3d backward = attitude(source);
  const Eigen::Matrix3d rotation = backward.transpose() * forward;
  const Eigen::Vector3d shift = backward.transpose() *
    Eigen::Vector3d(reference.x - source.x, reference.y - source.y, 0.0);
  int at = 0;
  out[at++] = static_cast<float>(lens.focal);
  out[at++] = static_cast<float>(lens.cx);
  out[at++] = static_cast<float>(lens.cy);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {out[at++] = static_cast<float>(lens.rotation_base_from_camera(r, c));}
  }
  for (int i = 0; i < 3; ++i) {out[at++] = static_cast<float>(lens.translation_base_from_camera(i));}
  for (int i = 0; i < 3; ++i) {out[at++] = static_cast<float>(normal(i));}
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {out[at++] = static_cast<float>(rotation(r, c));}
  }
  for (int i = 0; i < 3; ++i) {out[at++] = static_cast<float>(shift(i));}
}

// Run the kernel for one keyframe. Fills best/costs and, if requested, the
// aggregated volume (needed only for subplane). Returns false if CUDA is not
// available, so the caller can take the CPU path.
bool cuda_match(
  const Lens & lens, const SweepSettings & settings,
  const cv::Mat & reference32, const std::vector<cv::Mat> & source32,
  const Pose5 & reference_pose, const std::vector<Pose5> & source_poses,
  const std::vector<double> & heights, int road,
  cv::Mat & best, cv::Mat & best_cost, cv::Mat & road_cost, cv::Mat & second_cost,
  std::vector<cv::Mat> & volume, bool want_volume)
{
  static bool announced = false;
  if (!announced) {
    std::fprintf(stderr, "[monoscale_sweep] cuda backend: available=%d\n",
      plane_sweep_cuda_available() ? 1 : 0);
    announced = true;
  }
  if (!cuda_available()) {
    return false;
  }
  const int rows = reference32.rows;
  const int cols = reference32.cols;
  const int planes = static_cast<int>(heights.size());
  const int sources = static_cast<int>(source32.size());
  const Eigen::Vector3d normal = attitude(reference_pose).transpose() *
    Eigen::Vector3d(0.0, 0.0, 1.0);

  // Contiguous float32 reference and stacked sources.
  std::vector<float> ref(static_cast<size_t>(rows) * cols);
  std::memcpy(ref.data(), reference32.ptr<float>(0), ref.size() * sizeof(float));
  std::vector<float> src(static_cast<size_t>(rows) * cols * sources);
  for (int s = 0; s < sources; ++s) {
    std::memcpy(src.data() + static_cast<size_t>(s) * rows * cols,
      source32[s].ptr<float>(0), static_cast<size_t>(rows) * cols * sizeof(float));
  }
  std::vector<float> geometry(static_cast<size_t>(sources) * 30);
  for (int s = 0; s < sources; ++s) {
    fisheye_geometry(lens, reference_pose, source_poses[s], normal,
      geometry.data() + static_cast<size_t>(s) * 30);
  }
  std::vector<float> offsets(planes);
  for (int p = 0; p < planes; ++p) {offsets[p] = static_cast<float>(heights[p]);}

  best.create(rows, cols, CV_32S);
  best_cost.create(rows, cols, CV_32F);
  road_cost.create(rows, cols, CV_32F);
  second_cost.create(rows, cols, CV_32F);
  std::vector<float> vol;
  float * vol_ptr = nullptr;
  if (want_volume) {
    vol.assign(static_cast<size_t>(planes) * rows * cols, 0.0f);
    vol_ptr = vol.data();
  }

  // seen_fraction reaches the kernel only through the environment, the same
  // channel plane_sweep.py uses. Set it to our setting for this call.
  static bool set_env = false;
  if (!set_env) {
    std::string v = std::to_string(settings.seen_fraction);
    setenv("SWEEP_SEEN_FRACTION", v.c_str(), 1);
    set_env = true;
  }

  plane_sweep_cuda_impl(
    ref.data(), src.data(), sources, nullptr, planes, rows, cols,
    settings.window_x, settings.window_y, /*cost_kind=*/1, road,
    static_cast<float>(settings.small_step), static_cast<float>(settings.big_step),
    best.ptr<int>(0), best_cost.ptr<float>(0), road_cost.ptr<float>(0),
    second_cost.ptr<float>(0), vol_ptr, geometry.data(), offsets.data());

  if (want_volume) {
    volume.resize(planes);
    for (int p = 0; p < planes; ++p) {
      volume[p].create(rows, cols, CV_32F);
      std::memcpy(volume[p].ptr<float>(0),
        vol.data() + static_cast<size_t>(p) * rows * cols,
        static_cast<size_t>(rows) * cols * sizeof(float));
    }
  }
  return true;
}

}  // namespace monoscale_sweep
