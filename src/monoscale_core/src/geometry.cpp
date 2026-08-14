#include "monoscale_core/geometry.hpp"

#include <algorithm>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace monoscale
{

namespace
{

// A view of a row-major Nx2 block as OpenCV sees point lists: N rows of one
// two-channel element. No copy; the memory layout is already what OpenCV wants.
cv::Mat as_point_mat(const Points2 & points)
{
  return cv::Mat(
    static_cast<int>(points.rows()), 1, CV_64FC2,
    const_cast<double *>(points.data()));
}

Points2 from_point_mat(const cv::Mat & mat)
{
  Points2 out(mat.rows, 2);
  const cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();
  std::copy(
    continuous.ptr<double>(), continuous.ptr<double>() + 2 * mat.rows, out.data());
  return out;
}

}  // namespace

Lens lens_from_name(const std::string & distortion_model)
{
  return (distortion_model == "equidistant" || distortion_model == "fisheye")
         ? Lens::Equidistant : Lens::Pinhole;
}

CameraModel make_camera_model(
  const Eigen::Matrix3d & k, const Eigen::Matrix3d & rotation_base_from_camera,
  const Eigen::Vector3d & translation_base_from_camera,
  const Eigen::VectorXd & distortion, Lens lens)
{
  CameraModel model;
  model.k = k;
  model.rotation_base_from_camera = rotation_base_from_camera;
  model.translation_base_from_camera = translation_base_from_camera;
  model.distortion = distortion;
  model.lens = lens;
  model.refresh();
  return model;
}

Pose2 Pose2::compose(const PlanarMotion & motion) const
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return Pose2{
    x + c * motion.x - s * motion.y,
    y + s * motion.x + c * motion.y,
    wrap_pi(yaw + motion.yaw)};
}

PlanarMotion relative_motion(const Pose2 & origin, const Pose2 & pose)
{
  const double c = std::cos(origin.yaw);
  const double s = std::sin(origin.yaw);
  const double dx = pose.x - origin.x;
  const double dy = pose.y - origin.y;
  return PlanarMotion{
    c * dx + s * dy, -s * dx + c * dy, wrap_pi(pose.yaw - origin.yaw), 0, 1.0};
}

Eigen::Matrix3d intrinsic_from_fov(int width, int height, double horizontal_fov_deg)
{
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  const double fov = horizontal_fov_deg * M_PI / 180.0;
  if (!(fov > 0.0) || !(fov < M_PI)) {
    throw std::invalid_argument("horizontal FOV must be between 0 and 180 degrees");
  }
  const double focal = width / (2.0 * std::tan(0.5 * fov));
  Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
  k(0, 0) = focal;
  k(1, 1) = focal;
  k(0, 2) = 0.5 * (width - 1);
  k(1, 2) = 0.5 * (height - 1);
  return k;
}

Points2 undistort_pixels(const Points2 & pixels, const CameraModel & model)
{
  if (pixels.rows() == 0) {
    return pixels;
  }
  const bool has_coefficients =
    model.distortion.size() > 0 && model.distortion.cwiseAbs().maxCoeff() > 0.0;

  if (model.lens == Lens::Pinhole && !has_coefficients) {
    // CARLA renders an ideal pinhole and reports plumb_bob with zero
    // coefficients, so undistortion is the identity on every pixel.
    return pixels;
  }

  if (model.lens == Lens::Equidistant && !has_coefficients) {
    // r = f*theta, so the normalised radius already is the angle, and the
    // pinhole radius a downstream inv(K) expects is its tangent. Everything
    // else here is the same K applied forwards and backwards.
    Points2 out(pixels.rows(), 2);
    const double fx = model.k(0, 0);
    const double fy = model.k(1, 1);
    const double cx = model.k(0, 2);
    const double cy = model.k(1, 2);
    // tan() runs away at a right angle from the axis, which no real lens
    // reaches; OpenCV clamps at the same place.
    const double limit = 0.5 * M_PI - 1e-6;
    for (Eigen::Index i = 0; i < pixels.rows(); ++i) {
      const double nx = (pixels(i, 0) - cx) / fx;
      const double ny = (pixels(i, 1) - cy) / fy;
      const double theta = std::hypot(nx, ny);
      double scale = 1.0;
      if (theta > 1e-8) {
        scale = std::tan(std::min(theta, limit)) / theta;
      }
      out(i, 0) = nx * scale * fx + cx;
      out(i, 1) = ny * scale * fy + cy;
    }
    return out;
  }

  cv::Mat k(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      k.at<double>(r, c) = model.k(r, c);
    }
  }
  cv::Mat d(static_cast<int>(model.distortion.size()), 1, CV_64F);
  for (Eigen::Index i = 0; i < model.distortion.size(); ++i) {
    d.at<double>(static_cast<int>(i), 0) = model.distortion(i);
  }

  const cv::Mat points = as_point_mat(pixels);
  cv::Mat corrected;
  if (model.lens == Lens::Equidistant) {
    cv::Mat four = d.rowRange(0, std::min(4, d.rows)).clone();
    if (four.rows < 4) {
      four = cv::Mat::zeros(4, 1, CV_64F);
    }
    cv::fisheye::undistortPoints(points, corrected, k, four, cv::noArray(), k);
  } else {
    cv::undistortPoints(points, corrected, k, d, cv::noArray(), k);
  }
  return from_point_mat(corrected);
}

void pixels_to_ground(
  const Points2 & pixels, const CameraModel & model, double max_distance,
  double min_distance, const Eigen::Matrix3d * tilt, double range_scale,
  Points2 & ground_out, Mask & valid_out)
{
  const Eigen::Index count = pixels.rows();
  ground_out.resize(count, 2);
  valid_out.resize(count);
  if (count == 0) {
    return;
  }

  const Points2 corrected = undistort_pixels(pixels, model);

  Eigen::Vector3d origin = model.translation_base_from_camera;
  if (range_scale != 1.0) {
    // Every range out of here is proportional to the camera's height over the
    // plane, so the whole measurement scales with this one number.
    origin.z() /= range_scale;
  }

  Eigen::Vector3d normal(0.0, 0.0, 1.0);
  if (tilt != nullptr) {
    // The ground stays level in the world while the body tilts, so in body
    // coordinates the plane's normal is the world vertical seen from here.
    normal = tilt->transpose() * Eigen::Vector3d(0.0, 0.0, 1.0);
  }

  // One matrix instead of two: the pixel goes straight to a ray in base_link.
  const Eigen::Matrix3d ray_from_pixel = model.rotation_base_from_camera * model.k_inverse;
  const double height = origin.dot(normal);

  for (Eigen::Index i = 0; i < count; ++i) {
    const Eigen::Vector3d homogeneous(corrected(i, 0), corrected(i, 1), 1.0);
    const Eigen::Vector3d ray = ray_from_pixel * homogeneous;
    const double denominator = ray.dot(normal);
    const bool nonparallel = std::abs(denominator) > 1e-8;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    double lambda = std::numeric_limits<double>::quiet_NaN();
    if (nonparallel) {
      lambda = -height / denominator;
      point = origin + lambda * ray;
      if (tilt != nullptr) {
        // Into the level frame, so the two coordinates that come out are the
        // ones the planar solve is entitled to treat as a plane.
        point = (*tilt) * point;
      }
    }
    const double dx = point.x() - origin.x();
    const double dy = point.y() - origin.y();
    const double distance = std::hypot(dx, dy);
    ground_out(i, 0) = point.x();
    ground_out(i, 1) = point.y();
    valid_out(i) = nonparallel && std::isfinite(point.x()) && std::isfinite(point.y()) &&
      std::isfinite(point.z()) && lambda > 0.0 && distance <= max_distance &&
      distance >= min_distance;
  }
}

std::pair<Points2, Mask> pixels_to_ground(
  const Points2 & pixels, const CameraModel & model, double max_distance,
  double min_distance, const Eigen::Matrix3d * tilt, double range_scale)
{
  Points2 ground;
  Mask valid;
  pixels_to_ground(
    pixels, model, max_distance, min_distance, tilt, range_scale, ground, valid);
  return {std::move(ground), std::move(valid)};
}

std::optional<MotionEstimate> estimate_planar_motion(
  const Points2 & previous_points, const Points2 & current_points,
  double ransac_threshold, int min_inliers, double max_scale_error)
{
  const Eigen::Index count = previous_points.rows();
  if (count < std::max<Eigen::Index>(3, min_inliers) || current_points.rows() != count) {
    return std::nullopt;
  }
  cv::Mat inlier_mat;
  const cv::Mat affine = cv::estimateAffinePartial2D(
    as_point_mat(current_points), as_point_mat(previous_points), inlier_mat,
    cv::RANSAC, ransac_threshold, 2000, 0.995, 10);
  if (affine.empty() || inlier_mat.empty()) {
    return std::nullopt;
  }

  MotionEstimate result;
  result.inliers.resize(count);
  int inliers = 0;
  for (Eigen::Index i = 0; i < count; ++i) {
    const bool kept = inlier_mat.at<uchar>(static_cast<int>(i)) != 0;
    result.inliers(i) = kept;
    inliers += kept ? 1 : 0;
  }
  if (inliers < min_inliers) {
    return std::nullopt;
  }

  const double a00 = affine.at<double>(0, 0);
  const double a01 = affine.at<double>(0, 1);
  const double a10 = affine.at<double>(1, 0);
  const double a11 = affine.at<double>(1, 1);
  const double determinant = a00 * a11 - a01 * a10;
  const double scale = std::sqrt(std::max(determinant, 0.0));
  if (!std::isfinite(scale) || std::abs(scale - 1.0) > max_scale_error) {
    return std::nullopt;
  }
  const double denominator = std::max(scale, 1e-12);
  result.motion = PlanarMotion{
    affine.at<double>(0, 2), affine.at<double>(1, 2),
    std::atan2(a10 / denominator, a00 / denominator), inliers, scale};
  return result;
}

std::optional<MotionEstimate> estimate_planar_motion_with_yaw(
  const Points2 & previous_points, const Points2 & current_points, double yaw,
  double ransac_threshold, int min_inliers)
{
  const Eigen::Index count = previous_points.rows();
  if (count < std::max<Eigen::Index>(2, min_inliers) || current_points.rows() != count) {
    return std::nullopt;
  }
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);

  // With the heading known each correspondence votes for one translation, so
  // the fit is a robust average rather than a search.
  Points2 offsets(count, 2);
  for (Eigen::Index i = 0; i < count; ++i) {
    offsets(i, 0) = previous_points(i, 0) - (c * current_points(i, 0) - s * current_points(i, 1));
    offsets(i, 1) = previous_points(i, 1) - (s * current_points(i, 0) + c * current_points(i, 1));
  }

  // Up to a hundred votes get a turn as the consensus centre, spread evenly
  // over the set rather than sampled, so the result does not depend on a seed.
  const Eigen::Index candidates = std::min<Eigen::Index>(count, 100);
  const double threshold_squared = ransac_threshold * ransac_threshold;
  Eigen::Index best_index = -1;
  int best_count = 0;
  for (Eigen::Index c_index = 0; c_index < candidates; ++c_index) {
    // Evenly spread over the set and truncated, not rounded: this mirrors
    // numpy's linspace(...).astype(int), and the fallback it feeds runs during
    // the warm-up, where a different consensus centre becomes a constant
    // offset the rest of the drive carries.
    const Eigen::Index index = candidates == 1
      ? 0
      : static_cast<Eigen::Index>(
      std::floor(static_cast<double>(c_index) * (count - 1) / (candidates - 1)));
    int agreeing = 0;
    for (Eigen::Index i = 0; i < count; ++i) {
      const double dx = offsets(i, 0) - offsets(index, 0);
      const double dy = offsets(i, 1) - offsets(index, 1);
      if (dx * dx + dy * dy <= threshold_squared) {
        ++agreeing;
      }
    }
    if (agreeing > best_count) {
      best_count = agreeing;
      best_index = index;
    }
  }
  if (best_index < 0 || best_count < min_inliers) {
    return std::nullopt;
  }

  // Median of the winning set, then a mean over whoever agrees with that.
  std::vector<double> xs;
  std::vector<double> ys;
  xs.reserve(static_cast<size_t>(best_count));
  ys.reserve(static_cast<size_t>(best_count));
  for (Eigen::Index i = 0; i < count; ++i) {
    const double dx = offsets(i, 0) - offsets(best_index, 0);
    const double dy = offsets(i, 1) - offsets(best_index, 1);
    if (dx * dx + dy * dy <= threshold_squared) {
      xs.push_back(offsets(i, 0));
      ys.push_back(offsets(i, 1));
    }
  }
  const auto median = [](std::vector<double> & values) {
      const size_t middle = values.size() / 2;
      std::nth_element(values.begin(), values.begin() + middle, values.end());
      const double upper = values[middle];
      if (values.size() % 2 == 1) {
        return upper;
      }
      const double lower = *std::max_element(values.begin(), values.begin() + middle);
      return 0.5 * (lower + upper);
    };
  Eigen::Vector2d translation(median(xs), median(ys));

  MotionEstimate result;
  result.inliers.resize(count);
  int inliers = 0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  for (Eigen::Index i = 0; i < count; ++i) {
    const double dx = offsets(i, 0) - translation.x();
    const double dy = offsets(i, 1) - translation.y();
    const bool kept = dx * dx + dy * dy <= threshold_squared;
    result.inliers(i) = kept;
    if (kept) {
      ++inliers;
      sum_x += offsets(i, 0);
      sum_y += offsets(i, 1);
    }
  }
  if (inliers < min_inliers) {
    return std::nullopt;
  }
  result.motion = PlanarMotion{
    sum_x / inliers, sum_y / inliers, yaw, inliers, 1.0};
  return result;
}

std::optional<PlanarMotion> fuse_planar_motions(const std::vector<PlanarMotion> & motions)
{
  if (motions.empty()) {
    return std::nullopt;
  }
  double total = 0.0;
  for (const auto & motion : motions) {
    total += std::max(motion.inliers, 1);
  }
  double x = 0.0;
  double y = 0.0;
  double sine = 0.0;
  double cosine = 0.0;
  double scale = 0.0;
  int inliers = 0;
  for (const auto & motion : motions) {
    const double weight = std::max(motion.inliers, 1) / total;
    x += weight * motion.x;
    y += weight * motion.y;
    sine += weight * std::sin(motion.yaw);
    cosine += weight * std::cos(motion.yaw);
    scale += weight * motion.scale;
    inliers += motion.inliers;
  }
  return PlanarMotion{x, y, std::atan2(sine, cosine), inliers, scale};
}

void triangulate_temporal_points(
  const Points2 & previous_pixels, const Points2 & current_pixels,
  const CameraModel & model, const PlanarMotion & motion_current_to_previous,
  double min_parallax_deg, double max_reprojection_error, double min_distance,
  double max_distance, Points3 & points_out, Mask & valid_out)
{
  const Eigen::Index count = previous_pixels.rows();
  points_out.resize(count, 3);
  valid_out.resize(count);
  if (count == 0) {
    return;
  }

  const Points2 previous = undistort_pixels(previous_pixels, model);
  const Points2 current = undistort_pixels(current_pixels, model);

  const double c = std::cos(motion_current_to_previous.yaw);
  const double s = std::sin(motion_current_to_previous.yaw);
  Eigen::Matrix3d r_prev_from_current_base = Eigen::Matrix3d::Identity();
  r_prev_from_current_base(0, 0) = c;
  r_prev_from_current_base(0, 1) = -s;
  r_prev_from_current_base(1, 0) = s;
  r_prev_from_current_base(1, 1) = c;
  const Eigen::Vector3d t_prev_from_current_base(
    motion_current_to_previous.x, motion_current_to_previous.y, 0.0);

  const Eigen::Matrix3d & r_bc = model.rotation_base_from_camera;
  const Eigen::Vector3d & t_bc = model.translation_base_from_camera;
  const Eigen::Matrix3d r_c1_from_c2 = r_bc.transpose() * r_prev_from_current_base * r_bc;
  const Eigen::Vector3d t_c1_from_c2 =
    r_bc.transpose() * (r_prev_from_current_base * t_bc + t_prev_from_current_base - t_bc);
  const Eigen::Matrix3d r_c2_from_c1 = r_c1_from_c2.transpose();
  const Eigen::Vector3d t_c2_from_c1 = -r_c2_from_c1 * t_c1_from_c2;

  cv::Mat p1(3, 4, CV_64F, cv::Scalar(0.0));
  cv::Mat p2(3, 4, CV_64F, cv::Scalar(0.0));
  Eigen::Matrix<double, 3, 4> projection1 = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Matrix<double, 3, 4> projection2 = Eigen::Matrix<double, 3, 4>::Zero();
  projection1.leftCols<3>() = model.k;
  projection2.leftCols<3>() = model.k * r_c2_from_c1;
  projection2.rightCols<1>() = model.k * t_c2_from_c1;
  for (int r = 0; r < 3; ++r) {
    for (int col = 0; col < 4; ++col) {
      p1.at<double>(r, col) = projection1(r, col);
      p2.at<double>(r, col) = projection2(r, col);
    }
  }

  cv::Mat first(2, static_cast<int>(count), CV_64F);
  cv::Mat second(2, static_cast<int>(count), CV_64F);
  for (Eigen::Index i = 0; i < count; ++i) {
    first.at<double>(0, static_cast<int>(i)) = previous(i, 0);
    first.at<double>(1, static_cast<int>(i)) = previous(i, 1);
    second.at<double>(0, static_cast<int>(i)) = current(i, 0);
    second.at<double>(1, static_cast<int>(i)) = current(i, 1);
  }
  cv::Mat homogeneous;
  cv::triangulatePoints(p1, p2, first, second, homogeneous);

  const double min_parallax = min_parallax_deg * M_PI / 180.0;
  for (Eigen::Index i = 0; i < count; ++i) {
    const int column = static_cast<int>(i);
    const double w = homogeneous.at<double>(3, column);
    const Eigen::Vector3d point_c1(
      homogeneous.at<double>(0, column) / w,
      homogeneous.at<double>(1, column) / w,
      homogeneous.at<double>(2, column) / w);
    const Eigen::Vector3d point_c2 = r_c2_from_c1 * point_c1 + t_c2_from_c1;

    const Eigen::Vector3d projected1 = model.k * point_c1;
    const Eigen::Vector3d projected2 = model.k * point_c2;
    const double error = std::max(
      std::hypot(
        projected1.x() / projected1.z() - previous(i, 0),
        projected1.y() / projected1.z() - previous(i, 1)),
      std::hypot(
        projected2.x() / projected2.z() - current(i, 0),
        projected2.y() / projected2.z() - current(i, 1)));

    const Eigen::Vector3d ray1 = point_c1.normalized();
    const Eigen::Vector3d ray2 = (point_c1 - t_c1_from_c2).normalized();
    const double cosine = std::clamp(ray1.dot(ray2), -1.0, 1.0);
    const double parallax = std::acos(cosine);
    const double distance = point_c1.norm();

    const Eigen::Vector3d in_base = r_bc * point_c1 + t_bc;
    points_out(i, 0) = in_base.x();
    points_out(i, 1) = in_base.y();
    points_out(i, 2) = in_base.z();
    valid_out(i) = std::isfinite(point_c1.x()) && std::isfinite(point_c1.y()) &&
      std::isfinite(point_c1.z()) && point_c1.z() > 0.0 && point_c2.z() > 0.0 &&
      error <= max_reprojection_error && parallax >= min_parallax &&
      distance >= min_distance && distance <= max_distance;
  }
}

Points2 project_ground_to_pixels(const Points2 & points_xy, const CameraModel & model)
{
  const Eigen::Index count = points_xy.rows();
  Points2 pixels(count, 2);
  const Eigen::Matrix3d rotation = model.rotation_base_from_camera.transpose();
  for (Eigen::Index i = 0; i < count; ++i) {
    const Eigen::Vector3d in_base(points_xy(i, 0), points_xy(i, 1), 0.0);
    Eigen::Vector3d camera = rotation * (in_base - model.translation_base_from_camera);
    const bool behind = camera.z() <= 1e-6;
    if (behind) {
      pixels(i, 0) = std::numeric_limits<double>::quiet_NaN();
      pixels(i, 1) = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    const Eigen::Vector3d projected = model.k * camera;
    pixels(i, 0) = projected.x() / projected.z();
    pixels(i, 1) = projected.y() / projected.z();
  }
  return pixels;
}

std::optional<Points2> predict_ground_pixels(
  const Points2 & pixels, const CameraModel & model, const PlanarMotion & motion)
{
  Points2 ground;
  Mask valid;
  pixels_to_ground(
    pixels, model, std::numeric_limits<double>::infinity(), 0.0, nullptr, 1.0,
    ground, valid);
  if (!valid.any()) {
    return std::nullopt;
  }
  const double c = std::cos(motion.yaw);
  const double s = std::sin(motion.yaw);
  Points2 moved(ground.rows(), 2);
  for (Eigen::Index i = 0; i < ground.rows(); ++i) {
    const double dx = ground(i, 0) - motion.x;
    const double dy = ground(i, 1) - motion.y;
    moved(i, 0) = c * dx + s * dy;
    moved(i, 1) = -s * dx + c * dy;
  }
  Points2 predicted = project_ground_to_pixels(moved, model);
  for (Eigen::Index i = 0; i < predicted.rows(); ++i) {
    if (!valid(i) || !std::isfinite(predicted(i, 0)) || !std::isfinite(predicted(i, 1))) {
      predicted(i, 0) = pixels(i, 0);
      predicted(i, 1) = pixels(i, 1);
    }
  }
  return predicted;
}

Points3 transform_points_to_world(const Points3 & points, const Pose2 & pose)
{
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);
  Points3 out = points;
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    out(i, 0) = pose.x + c * points(i, 0) - s * points(i, 1);
    out(i, 1) = pose.y + s * points(i, 0) + c * points(i, 1);
  }
  return out;
}

void twist_from_motion(const PlanarMotion & motion, double & vx, double & vy, double & omega)
{
  omega = motion.yaw;
  if (std::abs(omega) < 1e-9) {
    vx = motion.x;
    vy = motion.y;
    return;
  }
  const double a = std::sin(omega) / omega;
  const double b = (1.0 - std::cos(omega)) / omega;
  const double denominator = a * a + b * b;
  vx = (a * motion.x + b * motion.y) / denominator;
  vy = (-b * motion.x + a * motion.y) / denominator;
}

PlanarMotion motion_from_twist(double vx, double vy, double omega, double duration)
{
  vx *= duration;
  vy *= duration;
  omega *= duration;
  if (std::abs(omega) < 1e-9) {
    return PlanarMotion{vx, vy, omega, 0, 1.0};
  }
  const double a = std::sin(omega) / omega;
  const double b = (1.0 - std::cos(omega)) / omega;
  return PlanarMotion{a * vx - b * vy, b * vx + a * vy, omega, 0, 1.0};
}

PlanarMotion rescale_motion(const PlanarMotion & motion, double ratio)
{
  if (ratio == 1.0) {
    return motion;
  }
  double vx = 0.0;
  double vy = 0.0;
  double omega = 0.0;
  twist_from_motion(motion, vx, vy, omega);
  PlanarMotion carried = motion_from_twist(vx, vy, omega, ratio);
  carried.inliers = motion.inliers;
  carried.scale = motion.scale;
  return carried;
}

}  // namespace monoscale
