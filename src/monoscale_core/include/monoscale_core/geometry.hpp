// Ground geometry: what a pixel means in metres, and how two views of the same
// ground relate.
//
// The metric scale of this whole stack lives here. A pixel becomes a ray, the
// ray meets a plane the camera's own mounting height defines, and the distance
// that falls out is in metres because the height is. Nothing downstream
// invents scale; it only carries this one forward.

#ifndef MONOSCALE_CORE__GEOMETRY_HPP_
#define MONOSCALE_CORE__GEOMETRY_HPP_

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace monoscale
{

// Row major so a block of them can be handed to OpenCV, or to a kernel, without
// a transpose. Two columns because the ground solve is planar; the third
// coordinate is only ever zero or a height being measured.
using Points2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Points3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Mask = Eigen::Array<bool, Eigen::Dynamic, 1>;
using Weights = Eigen::VectorXd;

// Which projection the lens actually is. Kept as an enum rather than the
// CameraInfo string because it is read once per point and compared once per
// call; and because there is exactly one distinction that matters here.
enum class Lens
{
  // r = f*tan(theta). CARLA's cameras, and every rectified image.
  Pinhole,
  // r = f*theta. The assembled fisheye pair. Zero coefficients is not the
  // absence of distortion here -- it is the definition of the projection.
  Equidistant,
};

Lens lens_from_name(const std::string & distortion_model);

struct CameraModel
{
  Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d rotation_base_from_camera = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_base_from_camera = Eigen::Vector3d::Zero();
  Eigen::VectorXd distortion;
  Lens lens = Lens::Pinhole;

  // Carry the camera's own position into the level frame before measuring how
  // far a ground point is from it. Comparing a level-frame point against a
  // body-frame origin is only harmless while the body is level, and 0.89 m of
  // camera height at the 5.4 degrees curve_s10 rolls is 78 mm against a 114 mm
  // hop. It lives on the model, not in a global: two estimators in one process
  // -- an A/B harness, say -- would otherwise share whichever was built last.
  bool level_frame_origin = true;

  // inv(K), kept beside K because every ray in the stack needs it and it does
  // not change while the frame size holds.
  Eigen::Matrix3d k_inverse = Eigen::Matrix3d::Identity();

  void refresh() { k_inverse = k.inverse(); }
};

// Builds a model with its cached inverse already in place. Constructing the
// struct by hand and forgetting `refresh()` leaves every ray multiplied by the
// identity, which does not fail -- it quietly reads the wrong ground.
CameraModel make_camera_model(
  const Eigen::Matrix3d & k, const Eigen::Matrix3d & rotation_base_from_camera,
  const Eigen::Vector3d & translation_base_from_camera,
  const Eigen::VectorXd & distortion = Eigen::VectorXd(), Lens lens = Lens::Pinhole);

struct PlanarMotion
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  int inliers = 0;
  double scale = 1.0;
};

struct Pose2
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;

  Pose2 compose(const PlanarMotion & motion) const;
};

inline double wrap_pi(double angle)
{
  return std::remainder(angle, 2.0 * M_PI);
}

// Motion from `origin` to `pose`, expressed in the frame of `origin`. Same
// convention as the frame to frame estimate, so it can be handed to
// triangulation as the transform between two views.
PlanarMotion relative_motion(const Pose2 & origin, const Pose2 & pose);

Eigen::Matrix3d intrinsic_from_fov(int width, int height, double horizontal_fov_deg);

// Pixels as this camera sees them, rewritten as pixels a pinhole would see.
//
// Everything downstream turns a pixel into a ray with inv(K), which is the
// pinhole inverse. A fisheye pixel put through it is simply read wrong.
//
// For the equidistant lens with no coefficients this is closed form: the
// normalised radius IS the angle, so the pinhole radius is its tangent. OpenCV
// reaches the same answer by running ten Newton steps that all find the
// residual already zero. The result is identical and the iteration is not.
Points2 undistort_pixels(const Points2 & pixels, const CameraModel & model);

// Ground intersections of the rays through `pixels`, and which are usable.
//
// `min_distance` exists because pixel motion goes as the inverse of range: the
// closest ground sweeps the image fastest and is the first thing optical flow
// loses.
//
// `tilt` is the vehicle's roll and pitch as a rotation from base into a level
// frame, and it is not a refinement. A camera pitched 30 degrees down sees the
// ground at a range that goes as 1/tan(30 deg + d), so one degree of body tilt
// moves every range in the frame by two and a half to five per cent. Passing
// nullptr keeps the level assumption.
//
// `range_scale` divides every range out by a factor measured for this camera.
// It is not a claim about where the camera is -- the extrinsics have to keep
// matching the sensor kit -- but about how far the ground is measured to have
// moved once the flow and this projection are both done with.
void pixels_to_ground(
  const Points2 & pixels, const CameraModel & model, double max_distance,
  double min_distance, const Eigen::Matrix3d * tilt, double range_scale,
  Points2 & ground_out, Mask & valid_out);

// Same, allocating its own results. The caller that runs this every solve
// should prefer the form above and keep its buffers.
std::pair<Points2, Mask> pixels_to_ground(
  const Points2 & pixels, const CameraModel & model, double max_distance,
  double min_distance = 0.0, const Eigen::Matrix3d * tilt = nullptr,
  double range_scale = 1.0);

struct MotionEstimate
{
  PlanarMotion motion;
  Mask inliers;
};

// Similarity fit through RANSAC, for the frames before the anchor map can
// answer. Returns nothing when too few points agree or the recovered scale
// strays further than `max_scale_error` from unity.
std::optional<MotionEstimate> estimate_planar_motion(
  const Points2 & previous_points, const Points2 & current_points,
  double ransac_threshold, int min_inliers, double max_scale_error);

// The same fit with the heading taken as known, which turns the search into a
// robust average: each correspondence votes for one translation.
std::optional<MotionEstimate> estimate_planar_motion_with_yaw(
  const Points2 & previous_points, const Points2 & current_points, double yaw,
  double ransac_threshold, int min_inliers, double softness = 0.0,
  const Weights & weights = Weights(), int passes = 1);

// Combine what several cameras made of the same hop.
//
// `weights` empty means weigh by inlier count, which is what this always did.
// Supplying them matters because the cameras are not independent: a body pitch
// error scales the forward camera's ground ranges down and the rearward one's
// up, by (R^2 + h^2) / (h R) each. The average cancels that error only when
// the weights sit in the inverse ratio of those sensitivities, and inlier
// counts have no reason to.
std::optional<PlanarMotion> fuse_planar_motions(
  const std::vector<PlanarMotion> & motions, const std::vector<double> & weights = {});

// Triangulate the same features seen from two poses, in the base frame of the
// earlier one. Used for obstacles, never for the pose.
void triangulate_temporal_points(
  const Points2 & previous_pixels, const Points2 & current_pixels,
  const CameraModel & model, const PlanarMotion & motion_current_to_previous,
  double min_parallax_deg, double max_reprojection_error, double min_distance,
  double max_distance, Points3 & points_out, Mask & valid_out);

// Project ground points, given in base_link, back into the image.
Points2 project_ground_to_pixels(const Points2 & points_xy, const CameraModel & model);

// Where ground features should land after the vehicle moves by `motion`.
//
// Optical flow searches around wherever the feature was last seen. Without a
// prediction the search settles short of the true displacement, and the
// consensus forms around features that appear barely to have moved. Measured
// on a straight 46 m run at 2 m/s, dropping this took drift from 3.3% to 21.7%.
std::optional<Points2> predict_ground_pixels(
  const Points2 & pixels, const CameraModel & model, const PlanarMotion & motion);

// Which way the camera moved, from features that are not on the ground.
//
// The ground solve reads scale off the plane, so its answer is only ever as
// good as the plane it assumed. Features off the plane -- buildings, poles,
// other vehicles -- carry no height and cannot give scale, but they do say
// which *direction* the camera went, and they say it without reference to any
// plane at all. That makes it an independent check on the half of the hop the
// ground is worst at.
//
// The epipolar constraint normally leaves five unknowns. Here the rotation is
// already known from the gyro and the motion is planar, so the translation has
// one degree of freedom -- its bearing -- and x2'[t]xR x1 = 0 collapses to a
// 2x2 homogeneous system whose smallest eigenvector is the answer. No RANSAC,
// no five point solver, no triangulation.
//
// Returns the unit bearing of the CAMERA's displacement in the earlier body
// frame, up to sign: the epipolar constraint cannot tell forward from reverse
// and the caller resolves it. `softness` reweights by angular residual, the
// same Gaussian the ground alignment uses.
std::optional<Eigen::Vector2d> epipolar_bearing(
  const Points2 & previous_pixels, const Points2 & current_pixels, const Mask & use,
  const CameraModel & model, double yaw_delta, double softness, int min_points);

Points3 transform_points_to_world(const Points3 & points, const Pose2 & pose);

// The constant body twist that would produce this motion in unit time.
//
// A vehicle turning at a steady rate travels an arc, not a chord, and its
// heading turns while it does. This is the SE(2) logarithm; straight motion
// falls out of it unchanged, so there is no separate case to get wrong.
void twist_from_motion(const PlanarMotion & motion, double & vx, double & vy, double & omega);

// Where a constant body twist carries the vehicle over `duration`.
PlanarMotion motion_from_twist(double vx, double vy, double omega, double duration = 1.0);

// The same twist, held for `ratio` times as long.
PlanarMotion rescale_motion(const PlanarMotion & motion, double ratio);

}  // namespace monoscale

#endif  // MONOSCALE_CORE__GEOMETRY_HPP_
