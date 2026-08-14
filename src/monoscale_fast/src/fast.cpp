// The estimator's two hot functions, moved out of numpy.
//
// These are ports, not improvements. Every step is in the order the Python
// does it, including the ones that look redundant: the initial median, the
// three reweighting passes, the final reselection of inliers under the
// heading that was settled on. A port that is merely equivalent in exact
// arithmetic is not enough here, because the difference between the two would
// then show up as a difference in the drive's score and there would be no way
// to tell it from a real change.

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

namespace py = pybind11;

// Compiled by nvcc in sweep_kernels.cu, which keeps every CUDA header out of
// this translation unit.
bool plane_sweep_cuda_available();
void plane_sweep_cuda_impl(
  const float * reference, const float * sources, int source_count,
  const float * homographies, int planes, int rows, int cols,
  int window_x, int window_y, int cost_kind, int road_plane,
  float small_step, float big_step,
  int * best_index, float * best_cost, float * road_cost, float * second_cost,
  float * volume_out,
  const float * geometry, const float * offsets);
void plane_sweep_cuda_volume(float * out, size_t count);

namespace
{

using Points = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

double wrap_pi(double angle)
{
  const double two_pi = 2.0 * M_PI;
  double wrapped = std::fmod(angle + M_PI, two_pi);
  if (wrapped < 0.0) {
    wrapped += two_pi;
  }
  return wrapped - M_PI;
}

// numpy's median: the mean of the two middle elements for an even count.
double median(std::vector<double> & values)
{
  const size_t n = values.size();
  if (n == 0) {
    return 0.0;
  }
  const size_t middle = n / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (n % 2 == 1) {
    return upper;
  }
  const double lower =
    *std::max_element(values.begin(), values.begin() + middle);
  return 0.5 * (lower + upper);
}

}  // namespace

// Ground intersections of the rays through `pixels`, and which are usable.
//
// `tilt` is the vehicle's roll and pitch as a rotation from the body into a
// level frame; an empty array keeps the level assumption the estimate was
// built on.
py::object pixels_to_ground(
  const Eigen::Ref<const Points> & pixels,
  const Eigen::Ref<const Eigen::Matrix3d> & k_inverse,
  const Eigen::Ref<const Eigen::Matrix3d> & rotation_base_from_camera,
  const Eigen::Ref<const Eigen::Vector3d> & origin,
  double max_distance,
  double min_distance,
  const std::optional<Eigen::Matrix3d> & tilt)
{
  const Eigen::Index count = pixels.rows();
  py::array_t<double> points_out({count, static_cast<Eigen::Index>(2)});
  py::array_t<bool> valid_out(count);
  auto points_view = points_out.mutable_unchecked<2>();
  auto valid_view = valid_out.mutable_unchecked<1>();

  Eigen::Vector3d normal(0.0, 0.0, 1.0);
  if (tilt) {
    // The ground stays level in the world while the body tilts, so in body
    // coordinates the plane's normal is the world vertical seen from here.
    normal = tilt->transpose() * Eigen::Vector3d(0.0, 0.0, 1.0);
  }
  const Eigen::Matrix3d ray_from_pixel = rotation_base_from_camera * k_inverse;
  const double height = origin.dot(normal);

  for (Eigen::Index i = 0; i < count; ++i) {
    const Eigen::Vector3d homogeneous(pixels(i, 0), pixels(i, 1), 1.0);
    const Eigen::Vector3d ray = ray_from_pixel * homogeneous;
    const double denominator = ray.dot(normal);
    bool usable = std::abs(denominator) > 1e-8;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    double lambda = std::numeric_limits<double>::quiet_NaN();
    if (usable) {
      lambda = -height / denominator;
      point = origin + lambda * ray;
      if (tilt) {
        // Into the level frame, so the two coordinates that come out are the
        // ones the planar solve is entitled to treat as a plane.
        point = (*tilt) * point;
      }
    }
    const double dx = point.x() - origin.x();
    const double dy = point.y() - origin.y();
    const double distance = std::hypot(dx, dy);
    const bool finite = std::isfinite(point.x()) && std::isfinite(point.y()) &&
      std::isfinite(point.z());
    points_view(i, 0) = point.x();
    points_view(i, 1) = point.y();
    valid_view(i) = usable && finite && lambda > 0.0 &&
      distance <= max_distance && distance >= min_distance;
  }
  return py::make_tuple(points_out, valid_out);
}

// Translation that places the current ground view onto its world anchors.
py::object align_to_anchors(
  const Eigen::Ref<const Points> & body_points,
  const Eigen::Ref<const Points> & world_points,
  const Eigen::Ref<const Vector> & weights_in,
  double yaw,
  double threshold,
  int min_inliers,
  bool refine_yaw)
{
  const Eigen::Index count = body_points.rows();
  if (count < std::max<Eigen::Index>(2, min_inliers)) {
    return py::none();
  }

  Vector weights = weights_in;
  if (weights.size() != count) {
    weights = Vector::Ones(count);
  }

  const double given_yaw = yaw;
  double applied = std::numeric_limits<double>::infinity();
  Eigen::Vector2d centre = Eigen::Vector2d::Zero();
  bool have_centre = false;
  std::vector<char> inliers(static_cast<size_t>(count), 0);
  Points votes(count, 2);

  for (int iteration = 0; iteration < 4; ++iteration) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    for (Eigen::Index i = 0; i < count; ++i) {
      const double bx = body_points(i, 0);
      const double by = body_points(i, 1);
      votes(i, 0) = world_points(i, 0) - (c * bx - s * by);
      votes(i, 1) = world_points(i, 1) - (s * bx + c * by);
    }
    if (!have_centre) {
      std::vector<double> xs(static_cast<size_t>(count));
      std::vector<double> ys(static_cast<size_t>(count));
      for (Eigen::Index i = 0; i < count; ++i) {
        xs[static_cast<size_t>(i)] = votes(i, 0);
        ys[static_cast<size_t>(i)] = votes(i, 1);
      }
      centre = Eigen::Vector2d(median(xs), median(ys));
      have_centre = true;
    }

    for (int pass = 0; pass < 3; ++pass) {
      Eigen::Index kept = 0;
      for (Eigen::Index i = 0; i < count; ++i) {
        const double residual = std::hypot(
          votes(i, 0) - centre.x(), votes(i, 1) - centre.y());
        inliers[static_cast<size_t>(i)] = residual <= threshold ? 1 : 0;
        kept += inliers[static_cast<size_t>(i)];
      }
      if (kept < min_inliers) {
        return py::none();
      }
      double total = 0.0;
      Eigen::Vector2d sum = Eigen::Vector2d::Zero();
      for (Eigen::Index i = 0; i < count; ++i) {
        if (!inliers[static_cast<size_t>(i)]) {
          continue;
        }
        const double w = weights(i);
        total += w;
        sum.x() += w * votes(i, 0);
        sum.y() += w * votes(i, 1);
      }
      centre = sum / total;
    }

    if (!refine_yaw || iteration == 3) {
      break;
    }

    double total = 0.0;
    Eigen::Vector2d body_centre = Eigen::Vector2d::Zero();
    Eigen::Vector2d world_centre = Eigen::Vector2d::Zero();
    for (Eigen::Index i = 0; i < count; ++i) {
      if (!inliers[static_cast<size_t>(i)]) {
        continue;
      }
      const double w = weights(i);
      total += w;
      body_centre += w * Eigen::Vector2d(body_points(i, 0), body_points(i, 1));
      world_centre += w * Eigen::Vector2d(world_points(i, 0), world_points(i, 1));
    }
    body_centre /= total;
    world_centre /= total;

    double cross = 0.0;
    double dot = 0.0;
    double lever_sum = 0.0;
    double fit_sum = 0.0;
    double kept = 0.0;
    for (Eigen::Index i = 0; i < count; ++i) {
      if (!inliers[static_cast<size_t>(i)]) {
        continue;
      }
      const double w = weights(i);
      const double bx = body_points(i, 0) - body_centre.x();
      const double by = body_points(i, 1) - body_centre.y();
      const double tx = world_points(i, 0) - world_centre.x();
      const double ty = world_points(i, 1) - world_centre.y();
      cross += w * (bx * ty - by * tx);
      dot += w * (bx * tx + by * ty);
      lever_sum += w * (bx * bx + by * by);
      const double rx = votes(i, 0) - centre.x();
      const double ry = votes(i, 1) - centre.y();
      fit_sum += rx * rx + ry * ry;
      kept += 1.0;
    }
    if (std::abs(cross) < 1e-12 && std::abs(dot) < 1e-12) {
      break;
    }
    const double proposed = wrap_pi(std::atan2(cross, dot));
    const double lever = std::sqrt(lever_sum / std::max(total, 1e-9));
    if (lever < 1e-6) {
      break;
    }
    const double fit = std::sqrt(fit_sum / std::max(kept, 1.0));
    applied = fit / (lever * std::sqrt(std::max(kept, 1.0)));
    if (std::abs(wrap_pi(proposed - yaw)) < 1e-6) {
      yaw = proposed;
      break;
    }
    yaw = proposed;
  }
  (void)given_yaw;

  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  py::array_t<bool> mask(count);
  auto mask_view = mask.mutable_unchecked<1>();
  Eigen::Index kept = 0;
  double squared = 0.0;
  for (Eigen::Index i = 0; i < count; ++i) {
    const double bx = body_points(i, 0);
    const double by = body_points(i, 1);
    const double rx = world_points(i, 0) - (c * bx - s * by) - centre.x();
    const double ry = world_points(i, 1) - (s * bx + c * by) - centre.y();
    const double residual = std::hypot(rx, ry);
    const bool inlier = residual <= threshold;
    mask_view(i) = inlier;
    if (inlier) {
      ++kept;
      squared += residual * residual;
    }
  }
  if (kept < min_inliers) {
    return py::none();
  }
  const double spread = std::sqrt(squared / static_cast<double>(kept));

  py::array_t<double> translation(2);
  auto translation_view = translation.mutable_unchecked<1>();
  translation_view(0) = centre.x();
  translation_view(1) = centre.y();
  return py::make_tuple(translation, mask, spread, yaw, applied);
}

// Accumulate the volume along one line of pixels, in place into `total`.
//
// The recurrence is the usual semi-global one: a plane keeps its own cost plus
// the cheapest way of arriving at it -- staying put, stepping one plane for
// `small_step`, or jumping anywhere for `big_step` -- with the line's running
// minimum subtracted so the numbers cannot grow without bound.
void walk_line(
  const std::vector<float> & cost, std::vector<float> & total, int planes,
  size_t first, size_t stride, int steps, float small_step, float big_step,
  size_t plane_stride, std::vector<float> & previous, std::vector<float> & current)
{
  for (int d = 0; d < planes; ++d) {
    previous[d] = cost[first + static_cast<size_t>(d) * plane_stride];
    total[first + static_cast<size_t>(d) * plane_stride] += previous[d];
  }
  for (int step = 1; step < steps; ++step) {
    float floor = previous[0];
    for (int d = 1; d < planes; ++d) {
      floor = std::min(floor, previous[d]);
    }
    const size_t here = first + static_cast<size_t>(step) * stride;
    for (int d = 0; d < planes; ++d) {
      float best = std::min(previous[d], floor + big_step);
      if (d > 0) {
        best = std::min(best, previous[d - 1] + small_step);
      }
      if (d + 1 < planes) {
        best = std::min(best, previous[d + 1] + small_step);
      }
      const size_t at = here + static_cast<size_t>(d) * plane_stride;
      current[d] = cost[at] + best - floor;
      total[at] += current[d];
    }
    previous.swap(current);
  }
}

// Four paths, along and against each axis. Diagonals cost as much again and
// buy little on a grid this coarse.
void aggregate_volume(
  std::vector<float> & volume, int planes, int rows, int cols,
  float small_step, float big_step)
{
  const size_t plane_stride = static_cast<size_t>(rows) * cols;
  const float infinity = std::numeric_limits<float>::infinity();
  std::vector<float> filled(volume.size());
  for (size_t i = 0; i < volume.size(); ++i) {
    filled[i] = std::isfinite(volume[i]) ? volume[i] : 1.0e6f;
  }
  std::vector<float> total(volume.size(), 0.0f);

  // Rows do not share memory with other rows, nor columns with other columns,
  // so each sweep direction parallelises outright. The two orientations do
  // share `total` and stay in sequence.
  cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range & range) {
    std::vector<float> previous(planes), current(planes);
    for (int r = range.start; r < range.end; ++r) {
      const size_t row_start = static_cast<size_t>(r) * cols;
      walk_line(filled, total, planes, row_start, 1, cols,
                small_step, big_step, plane_stride, previous, current);
      walk_line(filled, total, planes, row_start + cols - 1,
                static_cast<size_t>(-1), cols,
                small_step, big_step, plane_stride, previous, current);
    }
  });
  cv::parallel_for_(cv::Range(0, cols), [&](const cv::Range & range) {
    std::vector<float> previous(planes), current(planes);
    for (int c = range.start; c < range.end; ++c) {
      walk_line(filled, total, planes, static_cast<size_t>(c),
                static_cast<size_t>(cols), rows,
                small_step, big_step, plane_stride, previous, current);
      walk_line(filled, total, planes,
                static_cast<size_t>(rows - 1) * cols + c,
                static_cast<size_t>(-static_cast<ptrdiff_t>(cols)), rows,
                small_step, big_step, plane_stride, previous, current);
    }
  });

  for (size_t i = 0; i < volume.size(); ++i) {
    volume[i] = std::isfinite(volume[i]) ? total[i] : infinity;
  }
}

// A plane sweep over candidate heights, and the semi-global accumulation that
// makes its answers worth anything.
//
// The Python this replaces spends 199 ms of its 274 on the matching alone,
// not because OpenCV is slow -- warpPerspective and boxFilter are the same
// calls -- but because there are 312 of them per keyframe, each dispatched
// through the interpreter with its own temporaries. Here the buffers are
// allocated once and reused, and the volume never leaves C++.
//
// The arithmetic follows the Python step for step, including the order of the
// divide and the threshold that marks a plane unseen, so that a score taken
// with one is comparable with a score taken with the other.
py::tuple plane_sweep(
  py::array_t<float, py::array::c_style | py::array::forcecast> reference,
  const std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> & sources,
  py::array_t<double, py::array::c_style | py::array::forcecast> homographies,
  int window_x, int window_y, int cost_kind, int road_plane,
  double small_step, double big_step, bool want_volume)
{
  const auto shape = reference.request();
  if (shape.ndim != 2) {
    throw std::runtime_error("reference must be two dimensional");
  }
  const int rows = static_cast<int>(shape.shape[0]);
  const int cols = static_cast<int>(shape.shape[1]);
  const auto matrices = homographies.request();
  if (matrices.ndim != 4 || matrices.shape[2] != 3 || matrices.shape[3] != 3) {
    throw std::runtime_error("homographies must be (sources, planes, 3, 3)");
  }
  const int source_count = static_cast<int>(matrices.shape[0]);
  const int planes = static_cast<int>(matrices.shape[1]);
  if (source_count != static_cast<int>(sources.size())) {
    throw std::runtime_error("one homography set per source");
  }

  const cv::Mat reference_mat(
    rows, cols, CV_32F, const_cast<float *>(reference.data()));
  const double * homography_data = static_cast<const double *>(matrices.ptr);

  const size_t plane_size = static_cast<size_t>(rows) * cols;
  std::vector<float> volume(plane_size * planes, 0.0f);
  std::vector<float> seen(plane_size * planes, 0.0f);
  for (int s = 0; s < source_count; ++s) {
    const auto source_shape = sources[s].request();
    if (source_shape.shape[0] != rows || source_shape.shape[1] != cols) {
      throw std::runtime_error("sources must match the reference in size");
    }
  }

  // One pass over the pixels builds both the masked difference and the mask,
  // interleaved so a single box filter covers the pair; the Python spent four
  // full-image passes and two filters per hypothesis on the same result.
  //
  // The sources are summed before the filter rather than after. A box filter
  // is linear, so the two orders give the same answer, but one runs the filter
  // once per plane instead of once per plane per source -- twenty-six times
  // here rather than a hundred and fifty-six.
  //
  // The planes are independent, so they are swept in parallel, which is where
  // the port actually pays: the matching turned out to be OpenCV's work rather
  // than the interpreter's, and a straight transcription only bought a
  // quarter.
  const cv::Size kernel(window_x, window_y);
  const cv::Size size(cols, rows);
  cv::parallel_for_(cv::Range(0, planes), [&](const cv::Range & range) {
    cv::Mat warped, pair(rows, cols, CV_32FC2), blurred;
    // Correlation needs six window sums instead of two, and unlike the
    // difference it cannot be shared between sources: it divides by a spread
    // that belongs to one pair of images. So the filter runs per source here
    // and the costs are added afterwards, which is what the numpy did and
    // what makes the two comparable.
    cv::Mat first(rows, cols, CV_32FC3), second(rows, cols, CV_32FC3);
    cv::Mat first_blur, second_blur;
    for (int d = range.start; d < range.end; ++d) {
      pair.setTo(cv::Scalar(0.0, 0.0));
      float * pair_row = pair.ptr<float>();
      for (int s = 0; s < source_count; ++s) {
        const cv::Mat source(
          rows, cols, CV_32F, const_cast<float *>(sources[s].data()));
        const cv::Mat matrix(
          3, 3, CV_64F,
          const_cast<double *>(homography_data + (static_cast<size_t>(s) * planes + d) * 9));
        cv::warpPerspective(
          source, warped, matrix, size,
          cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
          cv::Scalar(-1.0));
        const float * warped_row = warped.ptr<float>();
        const float * reference_row = reference_mat.ptr<float>();
        if (cost_kind == 0) {
          for (size_t i = 0; i < plane_size; ++i) {
            const float value = warped_row[i];
            if (value >= 0.0f) {
              pair_row[2 * i] += std::abs(value - reference_row[i]);
              pair_row[2 * i + 1] += 1.0f;
            }
          }
          continue;
        }
        float * first_row = first.ptr<float>();
        float * second_row = second.ptr<float>();
        for (size_t i = 0; i < plane_size; ++i) {
          const float value = warped_row[i];
          const float held = reference_row[i];
          const bool inside = value >= 0.0f;
          first_row[3 * i] = inside ? 1.0f : 0.0f;
          first_row[3 * i + 1] = inside ? held : 0.0f;
          first_row[3 * i + 2] = inside ? held * held : 0.0f;
          second_row[3 * i] = inside ? value : 0.0f;
          second_row[3 * i + 1] = inside ? value * value : 0.0f;
          second_row[3 * i + 2] = inside ? held * value : 0.0f;
        }
        cv::boxFilter(first, first_blur, -1, kernel, cv::Point(-1, -1), false);
        cv::boxFilter(second, second_blur, -1, kernel, cv::Point(-1, -1), false);
        const float * fb = first_blur.ptr<float>();
        const float * sb = second_blur.ptr<float>();
        for (size_t i = 0; i < plane_size; ++i) {
          const float count = fb[3 * i];
          const float divisor = std::max(count, 1.0f);
          const float mean_a = fb[3 * i + 1] / divisor;
          const float mean_b = sb[3 * i] / divisor;
          const float var_a = fb[3 * i + 2] / divisor - mean_a * mean_a;
          const float var_b = sb[3 * i + 1] / divisor - mean_b * mean_b;
          const float covariance = sb[3 * i + 2] / divisor - mean_a * mean_b;
          const float correlation = covariance / std::sqrt(
            std::max(var_a, 0.0f) * std::max(var_b, 0.0f) + 1.0f);
          pair_row[2 * i] += (1.0f - correlation) * 100.0f * count;
          pair_row[2 * i + 1] += count;
        }
      }
      float * volume_plane = volume.data() + static_cast<size_t>(d) * plane_size;
      float * seen_plane = seen.data() + static_cast<size_t>(d) * plane_size;
      if (cost_kind == 0) {
        cv::boxFilter(pair, blurred, -1, kernel, cv::Point(-1, -1), false);
        const float * blurred_row = blurred.ptr<float>();
        for (size_t i = 0; i < plane_size; ++i) {
          volume_plane[i] = blurred_row[2 * i];
          seen_plane[i] = blurred_row[2 * i + 1];
        }
      } else {
        for (size_t i = 0; i < plane_size; ++i) {
          volume_plane[i] = pair_row[2 * i];
          seen_plane[i] = pair_row[2 * i + 1];
        }
      }
    }
  });

  const float unseen =
    static_cast<float>(window_x) * window_y * 0.5f * source_count;
  const float infinity = std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < volume.size(); ++i) {
    volume[i] /= std::max(seen[i], 1.0f);
    if (seen[i] < unseen) {
      volume[i] = infinity;
    }
  }

  if (small_step > 0.0) {
    aggregate_volume(volume, planes, rows, cols,
                     static_cast<float>(small_step), static_cast<float>(big_step));
  }

  py::array_t<int> best_index({rows, cols});
  py::array_t<float> best_cost({rows, cols});
  py::array_t<float> road_cost({rows, cols});
  py::array_t<float> second_cost({rows, cols});
  int * index_out = best_index.mutable_data();
  float * cost_out = best_cost.mutable_data();
  float * road_out = road_cost.mutable_data();
  float * second_out = second_cost.mutable_data();
  for (size_t i = 0; i < plane_size; ++i) {
    int argmin = 0;
    float lowest = volume[i];
    for (int d = 1; d < planes; ++d) {
      const float here = volume[static_cast<size_t>(d) * plane_size + i];
      if (here < lowest) {
        lowest = here;
        argmin = d;
      }
    }
    // The runner-up, taken from outside the winner's own shoulder. If it is
    // nearly as cheap the pixel has not chosen anything; it has two answers
    // and no way to tell them apart, which is the case winner-take-all
    // silently reports as a confident height.
    // The planes at and below the road all say "this is ground", so they are
    // not each other's rivals any more than a neighbouring plane is.
    const bool on_ground = argmin <= road_plane;
    float runner_up = std::numeric_limits<float>::infinity();
    for (int d = 0; d < planes; ++d) {
      if (d >= argmin - 1 && d <= argmin + 1) {
        continue;
      }
      if (on_ground && d <= road_plane) {
        continue;
      }
      runner_up = std::min(runner_up, volume[static_cast<size_t>(d) * plane_size + i]);
    }
    index_out[i] = argmin;
    cost_out[i] = lowest;
    second_out[i] = runner_up;
    // The best of the planes at or below the road, not the road plane alone.
    // Identical when the sweep starts at the road; with planes underneath, a
    // pixel matching one of them is low road rather than an obstacle.
    float ground = std::numeric_limits<float>::infinity();
    for (int d = 0; d <= road_plane && d < planes; ++d) {
      ground = std::min(ground, volume[static_cast<size_t>(d) * plane_size + i]);
    }
    road_out[i] = ground;
  }
  if (!want_volume) {
    return py::make_tuple(best_index, best_cost, road_cost, second_cost);
  }
  py::array_t<float> whole({planes, rows, cols});
  std::copy(volume.begin(), volume.end(), whole.mutable_data());
  return py::make_tuple(best_index, best_cost, road_cost, second_cost, whole);
}

// The same sweep on the GPU. Same arguments, same three images back; the
// homographies and the images go down as float, which is the one place the two
// paths are not doing identical arithmetic. OpenCV also interpolates its warp
// in fixed point where the kernel uses plain bilinear, so the two agree to
// about a grey level rather than to the bit.
py::tuple plane_sweep_cuda(
  py::array_t<float, py::array::c_style | py::array::forcecast> reference,
  const std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> & sources,
  py::array_t<double, py::array::c_style | py::array::forcecast> homographies,
  int window_x, int window_y, int cost_kind, int road_plane,
  double small_step, double big_step, bool want_volume,
  // Fisheye. `geometry` is (source_count, 30) and `offsets` is (planes,);
  // both empty means the pinhole path, where the homography stack carries the
  // whole correspondence. See zncc_accumulate_fisheye for the layout.
  py::array_t<double, py::array::c_style | py::array::forcecast> geometry =
    py::array_t<double, py::array::c_style | py::array::forcecast>(),
  py::array_t<double, py::array::c_style | py::array::forcecast> offsets =
    py::array_t<double, py::array::c_style | py::array::forcecast>())
{
  const auto shape = reference.request();
  if (shape.ndim != 2) {
    throw std::runtime_error("reference must be two dimensional");
  }
  const int rows = static_cast<int>(shape.shape[0]);
  const int cols = static_cast<int>(shape.shape[1]);
  const auto matrices = homographies.request();
  const auto lens = geometry.request();
  const auto heights = offsets.request();
  const bool fisheye = lens.size > 0 && heights.size > 0;
  const int source_count = fisheye
    ? static_cast<int>(lens.shape[0]) : static_cast<int>(matrices.shape[0]);
  const int planes = fisheye
    ? static_cast<int>(heights.shape[0]) : static_cast<int>(matrices.shape[1]);
  if (fisheye && lens.shape[1] != 30) {
    throw std::runtime_error("fisheye geometry must be (sources, 30)");
  }
  if (source_count != static_cast<int>(sources.size())) {
    throw std::runtime_error("one homography set per source");
  }

  const size_t pixels = static_cast<size_t>(rows) * cols;
  std::vector<float> stack(pixels * source_count);
  for (int s = 0; s < source_count; ++s) {
    std::copy(sources[s].data(), sources[s].data() + pixels,
              stack.begin() + static_cast<size_t>(s) * pixels);
  }
  std::vector<float> flat;
  std::vector<float> lens_flat;
  std::vector<float> height_flat;
  if (fisheye) {
    const double * g = static_cast<const double *>(lens.ptr);
    lens_flat.resize(static_cast<size_t>(source_count) * 30);
    for (size_t i = 0; i < lens_flat.size(); ++i) {
      lens_flat[i] = static_cast<float>(g[i]);
    }
    const double * o = static_cast<const double *>(heights.ptr);
    height_flat.resize(static_cast<size_t>(planes));
    for (size_t i = 0; i < height_flat.size(); ++i) {
      height_flat[i] = static_cast<float>(o[i]);
    }
  } else {
    const double * source_matrices = static_cast<const double *>(matrices.ptr);
    flat.resize(static_cast<size_t>(source_count) * planes * 9);
    for (size_t i = 0; i < flat.size(); ++i) {
      flat[i] = static_cast<float>(source_matrices[i]);
    }
  }

  py::array_t<int> best_index({rows, cols});
  py::array_t<float> best_cost({rows, cols});
  py::array_t<float> road_cost({rows, cols});
  py::array_t<float> second_cost({rows, cols});
  plane_sweep_cuda_impl(
    reference.data(), stack.data(), source_count,
    fisheye ? nullptr : flat.data(), planes, rows,
    cols, window_x, window_y, cost_kind, road_plane,
    static_cast<float>(small_step),
    static_cast<float>(big_step), best_index.mutable_data(),
    best_cost.mutable_data(), road_cost.mutable_data(),
    second_cost.mutable_data(), nullptr,
    fisheye ? lens_flat.data() : nullptr,
    fisheye ? height_flat.data() : nullptr);
  if (!want_volume) {
    return py::make_tuple(best_index, best_cost, road_cost, second_cost);
  }
  py::array_t<float> whole({planes, rows, cols});
  plane_sweep_cuda_volume(whole.mutable_data(),
                          static_cast<size_t>(planes) * rows * cols);
  return py::make_tuple(best_index, best_cost, road_cost, second_cost, whole);
}

// Every cell each road-bearing ray passed through, counted once per ray.
//
// Carving the whole ray rather than its far end is the single largest lever on
// the map's reliability: switching it off leaves 413 cells on the driven path
// where carving every ray leaves none. But the numpy that did it builds a
// rays-by-steps matrix and six more the same size -- ten thousand rays at four
// hundred samples is four million elements per temporary -- and once the sweep
// itself went to the GPU this became the run's bottleneck, two and a half times
// everything else put together.
//
// The samples along one ray are ordered, and a straight line enters and leaves
// a cell exactly once, so "different from the previous sample's cell" is the
// same de-duplication that np.unique was doing over the whole array.
py::array_t<float> carve_rays(
  double camera_x, double camera_y,
  py::array_t<double, py::array::c_style | py::array::forcecast> world_x,
  py::array_t<double, py::array::c_style | py::array::forcecast> world_y,
  double origin_x, double origin_y, double resolution,
  int width, int height, double spacing,
  py::object blocked_in)
{
  // A ray that has already met something cannot go on to report the ground
  // behind it as seen. Without this the sweep carved straight through the
  // parked cars: a smooth door panel matches the road plane about as badly as
  // it matches anything, the pixel is called road, its ground intersection
  // lands metres past the car, and the ray home rubs out every cell of the car
  // on the way. All four cars were marked free from two metres away.
  const unsigned char * blocked = nullptr;
  py::buffer_info blocked_info;
  if (!blocked_in.is_none()) {
    auto array = blocked_in.cast<
      py::array_t<unsigned char, py::array::c_style | py::array::forcecast>>();
    blocked_info = array.request();
    if (blocked_info.size != static_cast<py::ssize_t>(height) * width) {
      throw std::runtime_error("blocked must match the grid");
    }
    blocked = static_cast<const unsigned char *>(blocked_info.ptr);
  }
  const auto xs = world_x.request();
  const auto ys = world_y.request();
  if (xs.size != ys.size) {
    throw std::runtime_error("world_x and world_y must be the same length");
  }
  const double * ex = static_cast<const double *>(xs.ptr);
  const double * ey = static_cast<const double *>(ys.ptr);

  py::array_t<float> hits({height, width});
  float * counts = hits.mutable_data();
  std::fill(counts, counts + static_cast<size_t>(height) * width, 0.0f);

  for (py::ssize_t r = 0; r < xs.size; ++r) {
    const double dx = ex[r] - camera_x;
    const double dy = ey[r] - camera_y;
    const double reach = std::hypot(dx, dy);
    if (!(reach > 0.0) || !std::isfinite(reach)) {
      continue;
    }
    const int steps = static_cast<int>(std::ceil(reach / spacing)) + 1;
    long long previous = -1;
    for (int s = 0; s < steps; ++s) {
      const double along = s * spacing;
      if (along > reach) {
        break;
      }
      const double share = along / reach;
      const int column = static_cast<int>(std::floor(
        (camera_x + share * dx - origin_x) / resolution));
      const int row = static_cast<int>(std::floor(
        (camera_y + share * dy - origin_y) / resolution));
      if (column < 0 || column >= width || row < 0 || row >= height) {
        previous = -1;
        continue;
      }
      const long long flat = static_cast<long long>(row) * width + column;
      if (flat == previous) {
        continue;
      }
      if (blocked != nullptr && blocked[flat] != 0) {
        break;
      }
      previous = flat;
      counts[flat] += 1.0f;
    }
  }
  return hits;
}

PYBIND11_MODULE(monoscale_fast, m)
{
  m.doc() = "Ground projection and anchor alignment, ported from numpy.";
  // No gil_scoped_release call guard on either of these, though the two
  // cameras are worked by two threads and would overlap if there were. Both
  // build their numpy results as they go, and allocating a Python object
  // without the interpreter lock crashes the process rather than racing:
  // every drive scored "no result" the moment it was added. Overlapping them
  // means returning plain buffers and converting once at the end, which is a
  // change to the shape of the port rather than a flag on it.
  m.def(
    "pixels_to_ground", &pixels_to_ground,
    py::arg("pixels"), py::arg("k_inverse"), py::arg("rotation_base_from_camera"),
    py::arg("origin"), py::arg("max_distance"), py::arg("min_distance"),
    py::arg("tilt") = py::none());
  m.def(
    "plane_sweep", &plane_sweep,
    py::arg("reference"), py::arg("sources"), py::arg("homographies"),
    py::arg("window_x"), py::arg("window_y"), py::arg("cost_kind"),
    py::arg("road_plane"), py::arg("small_step"), py::arg("big_step"),
    py::arg("want_volume") = false);
  m.def(
    "plane_sweep_cuda", &plane_sweep_cuda,
    py::arg("reference"), py::arg("sources"), py::arg("homographies"),
    py::arg("window_x"), py::arg("window_y"), py::arg("cost_kind"),
    py::arg("road_plane"), py::arg("small_step"), py::arg("big_step"),
    py::arg("want_volume") = false,
    py::arg("geometry") = py::array_t<double>(),
    py::arg("offsets") = py::array_t<double>());
  m.def("plane_sweep_cuda_available", &plane_sweep_cuda_available);
  m.def(
    "carve_rays", &carve_rays,
    py::arg("camera_x"), py::arg("camera_y"), py::arg("world_x"),
    py::arg("world_y"), py::arg("origin_x"), py::arg("origin_y"),
    py::arg("resolution"), py::arg("width"), py::arg("height"),
    py::arg("spacing") = 0.05, py::arg("blocked") = py::none());
  m.def(
    "align_to_anchors", &align_to_anchors,
    py::arg("body_points"), py::arg("world_points"), py::arg("weights"),
    py::arg("yaw"), py::arg("threshold"), py::arg("min_inliers"),
    py::arg("refine_yaw") = false);
}
