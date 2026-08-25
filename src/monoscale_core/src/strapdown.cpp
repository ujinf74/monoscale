#include "monoscale_core/strapdown.hpp"

#include <cmath>
#include <limits>

namespace monoscale
{

namespace
{
// Slices of the error state, named where they are used.
constexpr int kP = 0;
constexpr int kV = 3;
constexpr int kTh = 6;
constexpr int kBA = 9;
constexpr int kBG = 12;
constexpr int kPA = 15;
constexpr int kThA = 18;
constexpr int kScale = 21;

double wrap(double angle)
{
  return std::remainder(angle, 2.0 * M_PI);
}

Eigen::Matrix3d cross_matrix(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d out;
  out << 0.0, -v.z(), v.y(),
    v.z(), 0.0, -v.x(),
    -v.y(), v.x(), 0.0;
  return out;
}

// The rotation a body turning at `omega` for `dt` ends up having applied.
Eigen::Quaterniond turned(const Eigen::Vector3d & rotation_vector)
{
  const double angle = rotation_vector.norm();
  if (angle < 1e-12) {
    // Below this the axis is noise and the rotation is the vector itself.
    return Eigen::Quaterniond(
      1.0, 0.5 * rotation_vector.x(), 0.5 * rotation_vector.y(),
      0.5 * rotation_vector.z()).normalized();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle));
}

double yaw_of(const Eigen::Matrix3d & rotation)
{
  return std::atan2(rotation(1, 0), rotation(0, 0));
}

// How a body-frame attitude error shows up as a heading error. To first order
// the world-frame rotation vector is R times the body-frame one, and the
// heading is its component about the world's vertical -- so this is the
// rotation's bottom row, which is (0, 0, 1) for a vehicle that is level and
// leans away from it exactly as far as the vehicle does.
Eigen::Matrix<double, 1, 3> heading_jacobian(const Eigen::Matrix3d & rotation)
{
  return rotation.row(2);
}

// The rotation about the world vertical alone, which is the frame the ground
// solve measures its hop in.
Eigen::Matrix3d level_from_yaw(double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d out = Eigen::Matrix3d::Identity();
  out(0, 0) = c;
  out(0, 1) = -s;
  out(1, 0) = s;
  out(1, 1) = c;
  return out;
}
}  // namespace

SpatialMsckfFilter::SpatialMsckfFilter()
: SpatialMsckfFilter(Settings())
{
}

SpatialMsckfFilter::SpatialMsckfFilter(const Settings & settings)
: settings_(settings)
{
  covariance_.setZero();
  covariance_.block<3, 3>(kV, kV) =
    Eigen::Matrix3d::Identity() * settings.initial_velocity_variance;
  covariance_.block<3, 3>(kBA, kBA) =
    Eigen::Matrix3d::Identity() * settings.initial_bias_variance;
  covariance_.block<3, 3>(kBG, kBG) =
    Eigen::Matrix3d::Identity() * settings.initial_gyro_bias_variance;
  covariance_(kTh + 0, kTh + 0) = settings.initial_tilt_variance;
  covariance_(kTh + 1, kTh + 1) = settings.initial_tilt_variance;
  covariance_(kTh + 2, kTh + 2) = settings.initial_heading_variance;
  covariance_(kScale, kScale) = settings.initial_scale_variance;
}

double SpatialMsckfFilter::yaw() const
{
  return yaw_of(attitude_.toRotationMatrix());
}

double SpatialMsckfFilter::roll() const
{
  const Eigen::Matrix3d r = attitude_.toRotationMatrix();
  return std::atan2(r(2, 1), r(2, 2));
}

double SpatialMsckfFilter::pitch() const
{
  const Eigen::Matrix3d r = attitude_.toRotationMatrix();
  return std::asin(std::max(-1.0, std::min(1.0, -r(2, 0))));
}

Eigen::Matrix3d SpatialMsckfFilter::body_tilt() const
{
  // Everything the attitude holds except the heading, which the ground
  // registration owns: putting yaw here as well would rotate the ground twice.
  return level_from_yaw(yaw()).transpose() * attitude_.toRotationMatrix();
}

void SpatialMsckfFilter::set_pose(const Eigen::Vector2d & position, double yaw)
{
  position_.head<2>() = position;
  const Eigen::Matrix3d tilt = body_tilt();
  attitude_ = Eigen::Quaterniond(level_from_yaw(wrap(yaw)) * tilt);
  attitude_.normalize();
}

void SpatialMsckfFilter::set_position(const Eigen::Vector2d & position)
{
  position_.head<2>() = position;
}

void SpatialMsckfFilter::open_hop()
{
  anchor_position_ = position_;
  anchor_attitude_ = attitude_;
  covariance_.block<3, kSize>(kPA, 0) = covariance_.block<3, kSize>(kP, 0);
  covariance_.block<3, kSize>(kThA, 0) = covariance_.block<3, kSize>(kTh, 0);
  covariance_.block<kSize, 3>(0, kPA) = covariance_.block<kSize, 3>(0, kP);
  covariance_.block<kSize, 3>(0, kThA) = covariance_.block<kSize, 3>(0, kTh);
  // The clones' own cross term has to come from the originals too, or the
  // anchor's pose and attitude are correlated with everything except each
  // other.
  covariance_.block<3, 3>(kPA, kThA) = covariance_.block<3, 3>(kP, kTh);
  covariance_.block<3, 3>(kThA, kPA) = covariance_.block<3, 3>(kTh, kP);
  covariance_.block<3, 3>(kPA, kPA) = covariance_.block<3, 3>(kP, kP);
  covariance_.block<3, 3>(kThA, kThA) = covariance_.block<3, 3>(kTh, kTh);
}

void SpatialMsckfFilter::predict(
  const Eigen::Vector3d & acceleration_body, const Eigen::Vector3d & rate_body, double dt)
{
  if (dt <= 0.0) {
    return;
  }
  const Eigen::Vector3d specific = acceleration_body - accel_bias_;
  const Eigen::Vector3d omega = rate_body - gyro_bias_;
  const Eigen::Matrix3d rotation = attitude_.toRotationMatrix();
  const Eigen::Vector3d gravity(0.0, 0.0, -settings_.gravity);
  // The part reports specific force, so what the vehicle is actually doing is
  // that with gravity put back.
  const Eigen::Vector3d world = rotation * specific + gravity;

  position_ += velocity_ * dt + 0.5 * world * dt * dt;
  velocity_ += world * dt;
  attitude_ = (attitude_ * turned(omega * dt)).normalized();

  const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d lever = rotation * cross_matrix(specific);
  Covariance transition = Covariance::Identity();
  transition.block<3, 3>(kP, kV) = eye * dt;
  transition.block<3, 3>(kP, kTh) = lever * (-0.5 * dt * dt);
  transition.block<3, 3>(kP, kBA) = rotation * (-0.5 * dt * dt);
  transition.block<3, 3>(kV, kTh) = lever * (-dt);
  transition.block<3, 3>(kV, kBA) = rotation * (-dt);
  transition.block<3, 3>(kTh, kTh) = eye - cross_matrix(omega) * dt;
  transition.block<3, 3>(kTh, kBG) = eye * (-dt);
  covariance_ = transition * covariance_ * transition.transpose();

  const double spectral = settings_.acceleration_noise * settings_.acceleration_noise;
  covariance_.block<3, 3>(kP, kP) += eye * (spectral * dt * dt * dt / 3.0);
  covariance_.block<3, 3>(kP, kV) += eye * (spectral * dt * dt / 2.0);
  covariance_.block<3, 3>(kV, kP) += eye * (spectral * dt * dt / 2.0);
  covariance_.block<3, 3>(kV, kV) += eye * (spectral * dt);
  // Roll and pitch on their own budget, and the heading on its own: they are
  // measured by different things and used for different things.
  Eigen::Vector3d attitude_walk(
    settings_.tilt_gyro_noise * settings_.tilt_gyro_noise,
    settings_.tilt_gyro_noise * settings_.tilt_gyro_noise,
    settings_.gyro_noise * settings_.gyro_noise);
  // Those are body axes and the split is about the world's vertical, so the
  // budget is carried into the body the vehicle is actually in.
  const Eigen::Matrix3d body_walk =
    rotation.transpose() * attitude_walk.asDiagonal() * rotation;
  covariance_.block<3, 3>(kTh, kTh) += body_walk * dt;
  covariance_.block<3, 3>(kBA, kBA) += eye * (settings_.bias_walk * settings_.bias_walk * dt);
  covariance_.block<3, 3>(kBG, kBG) +=
    eye * (settings_.gyro_bias_walk * settings_.gyro_bias_walk * dt);
  // Only if the scale is a state at all: a walk added to a variance of zero is
  // a state that was meant to be held at one and is not.
  if (settings_.initial_scale_variance > 0.0) {
    covariance_(kScale, kScale) += settings_.scale_walk * settings_.scale_walk * dt;
  }
}

Eigen::Matrix3d SpatialMsckfFilter::measurement_covariance(
  int inliers, double spread, double extra_variance, double yaw_sigma) const
{
  const double count = std::max(static_cast<double>(inliers), 1.0);
  const double floor = settings_.vision_noise_m * settings_.vision_noise_m *
    (settings_.vision_reference_inliers / count);
  double position = spread <= 0.0 ? floor : std::max(floor, spread * spread / count);
  position += std::max(extra_variance, 0.0);
  // A floor, not a fallback. The fit reports how tightly its own residuals pin
  // a rotation down -- fit over lever over the root of the inlier count -- and
  // on a good frame that is 2e-4 rad, a hundredth of what this is configured
  // at. The shrink by the root of the count is what makes it so small, and it
  // assumes the residuals are independent. A mounting or tilt error is not: it
  // leans the whole frame the same way, so more points do not average it out.
  // That is the same systematic error the heading measurement exists to catch,
  // and letting the fit claim 2e-4 hands it to the filter as certainty.
  const double configured = settings_.vision_yaw_noise * settings_.vision_yaw_noise;
  const double heading = (!std::isfinite(yaw_sigma) || yaw_sigma <= 0.0)
    ? configured : std::max(yaw_sigma * yaw_sigma, configured);
  Eigen::Matrix3d noise = Eigen::Matrix3d::Zero();
  noise(0, 0) = position;
  noise(1, 1) = position;
  noise(2, 2) = heading;
  return noise;
}

void SpatialMsckfFilter::observation(
  Eigen::Matrix<double, 3, kSize> & model, Eigen::Vector3d & predicted) const
{
  const Eigen::Matrix3d anchor_rotation = anchor_attitude_.toRotationMatrix();
  const double anchor_yaw = yaw_of(anchor_rotation);
  const Eigen::Matrix3d inverse = level_from_yaw(anchor_yaw).transpose();
  const Eigen::Vector3d offset = position_ - anchor_position_;
  // What vision will report, not what the vehicle did: the two differ by
  // whatever the ground projection's scale is wrong by, and that difference is
  // the only thing that makes the scale observable at all.
  predicted = range_scale_ * (inverse * offset);

  model.setZero();
  model.block<2, 3>(0, kP) = range_scale_ * inverse.topRows<2>();
  model.block<2, 3>(0, kPA) = -range_scale_ * inverse.topRows<2>();
  // The scale multiplies the whole hop, so its column is the hop itself.
  model.block<2, 1>(0, kScale) =
    range_scale_ > 1e-9 ? (predicted.head<2>() / range_scale_).eval() : Eigen::Vector2d::Zero();
  // Turning the anchor's frame turns what the hop looks like in it. Only the
  // horizontal part moves: the frame is a rotation about the vertical.
  Eigen::Matrix3d perpendicular = Eigen::Matrix3d::Zero();
  perpendicular(0, 1) = 1.0;
  perpendicular(1, 0) = -1.0;
  const Eigen::Matrix<double, 1, 3> anchor_heading = heading_jacobian(anchor_rotation);
  model.block<2, 3>(0, kThA) = (perpendicular * predicted).head<2>() * anchor_heading;

  model.block<1, 3>(2, kTh) = heading_jacobian(attitude_.toRotationMatrix());
  model.block<1, 3>(2, kThA) = -anchor_heading;
}

void SpatialMsckfFilter::correct(const Eigen::Matrix<double, kSize, 1> & error)
{
  position_ += error.segment<3>(kP);
  velocity_ += error.segment<3>(kV);
  attitude_ = (attitude_ * turned(error.segment<3>(kTh))).normalized();
  accel_bias_ += error.segment<3>(kBA);
  range_scale_ += error(kScale);
  gyro_bias_ += error.segment<3>(kBG);
  anchor_position_ += error.segment<3>(kPA);
  anchor_attitude_ = (anchor_attitude_ * turned(error.segment<3>(kThA))).normalized();
}

void SpatialMsckfFilter::apply(
  const Eigen::Matrix<double, 3, kSize> & model, const Eigen::Vector3d & innovation,
  const Eigen::Matrix3d & noise, const Eigen::Matrix3d & inverse)
{
  const Eigen::Matrix<double, kSize, 3> gain = covariance_ * model.transpose() * inverse;
  correct(gain * innovation);
  const Covariance spread = Covariance::Identity() - gain * model;
  covariance_ = spread * covariance_ * spread.transpose() + gain * noise * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

bool SpatialMsckfFilter::update(
  const Eigen::Vector2d & displacement_ground, double yaw_delta, int inliers,
  double extra_variance, double spread, double yaw_sigma)
{
  Eigen::Matrix<double, 3, kSize> model;
  Eigen::Vector3d predicted;
  observation(model, predicted);

  const double turn = wrap(yaw() - yaw_of(anchor_attitude_.toRotationMatrix()));
  Eigen::Vector3d innovation;
  innovation.head<2>() = displacement_ground - predicted.head<2>();
  innovation(2) = wrap(yaw_delta - turn);

  UpdateRecord record;
  record.innovation = innovation;

  if (settled() && innovation.head<2>().norm() > settings_.reject_beyond_m) {
    ++dropped_;
    record.dropped = true;
    record.nis = std::numeric_limits<double>::infinity();
    last_update_ = record;
    return false;
  }

  Eigen::Matrix3d noise = measurement_covariance(inliers, spread, extra_variance, yaw_sigma);
  Eigen::Matrix3d block = model * covariance_ * model.transpose() + noise;
  Eigen::Matrix3d inverse;
  bool invertible = false;
  double determinant = 0.0;
  block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible) {
    return false;
  }

  bool accepted = true;
  const double distance = innovation.dot(inverse * innovation);
  if (settled() && distance > settings_.innovation_gate) {
    ++rejected_;
    accepted = false;
    if (settings_.outlier_inflation <= 0.0) {
      ++dropped_;
      record.dropped = true;
      record.nis = distance;
      last_update_ = record;
      return false;
    }
    noise *= settings_.outlier_inflation;
    block = model * covariance_ * model.transpose() + noise;
    block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
    if (!invertible) {
      return false;
    }
  }

  record.nis = distance;
  record.accepted = accepted;
  last_update_ = record;
  apply(model, innovation, noise, inverse);
  ++updates_;
  return accepted;
}

bool SpatialMsckfFilter::update_gravity(const Eigen::Vector3d & acceleration_body)
{
  const Eigen::Vector3d measured = acceleration_body - accel_bias_;

  const Eigen::Matrix3d rotation = attitude_.toRotationMatrix();
  // Whether the vehicle is doing anything, asked of all three axes rather than
  // of the magnitude. The magnitude cannot see horizontal acceleration: a
  // vehicle pulling 0.3 m/s2 along its own x reads 9.8112 against gravity's
  // 9.80665, four thousandths out, so a magnitude gate set at 0.3 lets nearly
  // every accelerating sample through and levels the attitude against the
  // vehicle instead of against the planet. Measured, that left the attitude
  // pitched by 0.64 degrees on a recording of flat ground; rotated into the
  // world and with gravity taken off, the same 0.3 is 0.3, and it left 0.017.
  const Eigen::Vector3d world =
    rotation * measured - Eigen::Vector3d(0.0, 0.0, settings_.gravity);
  // Widened by what this filter's own attitude could be wrong by. Without that
  // the gate refuses to fix the error it exists to notice: an attitude out by
  // two degrees produces the same 0.34 m/s2 a vehicle accelerating at 0.34
  // does, so a fixed threshold locks in whatever the filter started with. It
  // tightens on its own as the tilt is pinned down, which is the point.
  const double tilt_sigma = std::sqrt(
    std::max(covariance_(kTh, kTh), covariance_(kTh + 1, kTh + 1)));
  if (world.norm() > settings_.gravity_tolerance + settings_.gravity * tilt_sigma) {
    return false;
  }
  // At rest the part reads gravity pushing up through it, which in the body
  // frame is the world's up turned into the body.
  const Eigen::Vector3d expected =
    rotation.transpose() * Eigen::Vector3d(0.0, 0.0, settings_.gravity);
  const Eigen::Vector3d innovation = measured - expected;

  // What the part would read, differentiated: the reading is the world's up
  // turned into the body plus the bias, so both enter with a plus. Turning the
  // body by a body-frame error moves that vector by [expected]x -- the sign
  // matters here in the way only a sign can, and getting it backwards tips the
  // attitude over instead of levelling it.
  Eigen::Matrix<double, 3, kSize> model = Eigen::Matrix<double, 3, kSize>::Zero();
  model.block<3, 3>(0, kTh) = cross_matrix(expected);
  model.block<3, 3>(0, kBA) = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d noise =
    Eigen::Matrix3d::Identity() * (settings_.gravity_noise * settings_.gravity_noise);
  Eigen::Matrix3d block = model * covariance_ * model.transpose() + noise;
  Eigen::Matrix3d inverse;
  bool invertible = false;
  double determinant = 0.0;
  block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible) {
    return false;
  }
  Eigen::Matrix<double, kSize, 3> gain = covariance_ * model.transpose() * inverse;
  // Gravity cannot see the heading. Turning about it leaves every accelerometer
  // reading exactly where it was, so that direction is not in the measurement
  // at all, and whatever the gain puts there came from a cross term rather than
  // from the instrument. Left in, it is a heading correction made out of
  // nothing, applied on every sample -- sixteen hundred times over a recording
  // -- and it showed up as hops scattering sideways at twice the rate the
  // planar filter's do.
  //
  // So the attitude's share of the correction is projected off that direction,
  // which in the body frame is where the reading itself points. Everything else
  // the levelling implies -- the velocity a tilt error was integrating into,
  // the bias it was hiding in -- is real and is left alone.
  const Eigen::Vector3d unobservable = expected.normalized();
  gain.block<3, 3>(kTh, 0) -=
    unobservable * (unobservable.transpose() * gain.block<3, 3>(kTh, 0));
  correct(gain * innovation);
  const Covariance spread = Covariance::Identity() - gain * model;
  covariance_ = spread * covariance_ * spread.transpose() + gain * noise * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
  ++levelled_;
  return true;
}

void SpatialMsckfFilter::update_zero_velocity(double sigma, double yaw_sigma)
{
  if (!(sigma > 0.0)) {
    return;
  }
  Eigen::Matrix<double, 4, kSize> model = Eigen::Matrix<double, 4, kSize>::Zero();
  model.block<3, 3>(0, kV) = Eigen::Matrix3d::Identity();
  model.block<1, 3>(3, kTh) = heading_jacobian(attitude_.toRotationMatrix());
  model.block<1, 3>(3, kThA) = -heading_jacobian(anchor_attitude_.toRotationMatrix());

  Eigen::Vector4d innovation;
  innovation.head<3>() = -velocity_;
  innovation(3) = -wrap(yaw() - yaw_of(anchor_attitude_.toRotationMatrix()));

  Eigen::Matrix4d noise = Eigen::Matrix4d::Identity() * (sigma * sigma);
  noise(3, 3) = std::max(yaw_sigma, 1e-6) * std::max(yaw_sigma, 1e-6);
  Eigen::Matrix4d block = model * covariance_ * model.transpose() + noise;
  Eigen::Matrix4d inverse;
  bool invertible = false;
  double determinant = 0.0;
  block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible) {
    return;
  }
  const Eigen::Matrix<double, kSize, 4> gain = covariance_ * model.transpose() * inverse;
  correct(gain * innovation);
  const Covariance spread = Covariance::Identity() - gain * model;
  covariance_ = spread * covariance_ * spread.transpose() + gain * noise * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

void SpatialMsckfFilter::update_height()
{
  if (!(settings_.height_noise_m > 0.0)) {
    return;
  }
  const Eigen::Matrix3d inverse =
    level_from_yaw(yaw_of(anchor_attitude_.toRotationMatrix())).transpose();
  const double climbed = (inverse * (position_ - anchor_position_)).z();

  Eigen::Matrix<double, 1, kSize> model = Eigen::Matrix<double, 1, kSize>::Zero();
  model.block<1, 3>(0, kP) = inverse.row(2);
  model.block<1, 3>(0, kPA) = -inverse.row(2);

  const double noise = settings_.height_noise_m * settings_.height_noise_m;
  const double block = (model * covariance_ * model.transpose())(0, 0) + noise;
  if (!(block > 0.0)) {
    return;
  }
  const Eigen::Matrix<double, kSize, 1> gain = covariance_ * model.transpose() / block;
  correct(gain * -climbed);
  const Covariance spread = Covariance::Identity() - gain * model;
  covariance_ = spread * covariance_ * spread.transpose() + gain * noise * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

void SpatialMsckfFilter::update_heading(double measured_yaw, double sigma)
{
  if (!(sigma > 0.0) || !std::isfinite(sigma)) {
    return;
  }
  Eigen::Matrix<double, 1, kSize> model = Eigen::Matrix<double, 1, kSize>::Zero();
  model.block<1, 3>(0, kTh) = heading_jacobian(attitude_.toRotationMatrix());
  const double innovation = wrap(measured_yaw - yaw());
  const double block = (model * covariance_ * model.transpose())(0, 0) + sigma * sigma;
  if (!(block > 0.0)) {
    return;
  }
  const Eigen::Matrix<double, kSize, 1> gain = covariance_ * model.transpose() / block;
  correct(gain * innovation);
  const Covariance spread = Covariance::Identity() - gain * model;
  covariance_ = spread * covariance_ * spread.transpose() +
    gain * (sigma * sigma) * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

Eigen::Vector2d SpatialMsckfFilter::body_translation() const
{
  const Eigen::Matrix3d inverse =
    level_from_yaw(yaw_of(anchor_attitude_.toRotationMatrix())).transpose();
  return (inverse * (position_ - anchor_position_)).head<2>();
}

double SpatialMsckfFilter::hop_yaw() const
{
  return wrap(yaw() - yaw_of(anchor_attitude_.toRotationMatrix()));
}

}  // namespace monoscale
