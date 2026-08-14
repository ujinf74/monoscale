#include "monoscale_core/fusion.hpp"

#include <algorithm>
#include <cmath>

namespace monoscale
{

namespace
{
// Slices of the displacement filter's state, named where they are used.
constexpr int kPosition = 0;
constexpr int kAnchor = 2;
constexpr int kVelocity = 4;
constexpr int kBias = 6;
}  // namespace

PlanarVelocityFilter::PlanarVelocityFilter()
: PlanarVelocityFilter(Settings())
{
}

PlanarVelocityFilter::PlanarVelocityFilter(const Settings & settings)
: settings_(settings),
  covariance_(Eigen::Matrix2d::Identity() * settings.initial_variance)
{
}

void PlanarVelocityFilter::predict(const Eigen::Vector2d & acceleration_world, double dt)
{
  if (dt <= 0.0) {
    return;
  }
  velocity_ += acceleration_world * dt;
  // Variance grows as q^2*dt, the white-noise form, not (q*dt)^2. While
  // predict() is driven by a fixed-rate IMU the two differ only by a constant
  // and the tuning absorbs it -- which is why q=12 was a sharp optimum under
  // the old form. They stop agreeing the moment the rate changes, and the
  // vehicle's IMU runs at 100 Hz where the simulation's runs at 60.
  const double growth = settings_.acceleration_noise * settings_.acceleration_noise * dt;
  covariance_ += Eigen::Matrix2d::Identity() * growth;
}

double PlanarVelocityFilter::measurement_variance(int inliers) const
{
  const double count = std::max(static_cast<double>(inliers), 1.0);
  const double scale = settings_.vision_reference_inliers / count;
  return settings_.vision_noise * settings_.vision_noise * scale;
}

bool PlanarVelocityFilter::update(
  const Eigen::Vector2d & measured, int inliers, double extra_variance)
{
  const double variance = measurement_variance(inliers) + std::max(extra_variance, 0.0);
  const Eigen::Vector2d innovation = measured - velocity_;
  Eigen::Matrix2d innovation_covariance = covariance_ + Eigen::Matrix2d::Identity() * variance;

  Eigen::Matrix2d inverse;
  bool invertible = false;
  double determinant = 0.0;
  innovation_covariance.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible) {
    return false;
  }

  // Normalised innovation squared, the standard consistency check. A solve that
  // disagrees with both the model and its own uncertainty is more likely wrong
  // than the vehicle is surprising.
  bool accepted = true;
  if (settled() && innovation.dot(inverse * innovation) > settings_.innovation_gate) {
    ++rejected_;
    accepted = false;
    innovation_covariance =
      covariance_ + Eigen::Matrix2d::Identity() * (variance * settings_.outlier_inflation);
    innovation_covariance.computeInverseAndDetWithCheck(inverse, determinant, invertible);
    if (!invertible) {
      return false;
    }
  }

  const Eigen::Matrix2d gain = covariance_ * inverse;
  velocity_ += gain * innovation;
  covariance_ = (Eigen::Matrix2d::Identity() - gain) * covariance_;
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
  ++updates_;
  return accepted;
}

Eigen::Vector2d PlanarVelocityFilter::body_translation(double dt, double yaw) const
{
  const Eigen::Vector2d delta = velocity_ * dt;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return Eigen::Vector2d(c * delta.x() + s * delta.y(), -s * delta.x() + c * delta.y());
}

std::optional<Eigen::Vector2d> world_velocity_from_motion(
  double motion_x, double motion_y, double dt, double yaw)
{
  if (dt <= 0.0) {
    return std::nullopt;
  }
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return Eigen::Vector2d(
    (c * motion_x - s * motion_y) / dt, (s * motion_x + c * motion_y) / dt);
}

PlanarDisplacementFilter::PlanarDisplacementFilter()
: PlanarDisplacementFilter(Settings())
{
}

PlanarDisplacementFilter::PlanarDisplacementFilter(const Settings & settings)
: settings_(settings)
{
  covariance_.setZero();
  covariance_(kVelocity, kVelocity) = settings.initial_velocity_variance;
  covariance_(kVelocity + 1, kVelocity + 1) = settings.initial_velocity_variance;
  covariance_(kBias, kBias) = settings.initial_bias_variance;
  covariance_(kBias + 1, kBias + 1) = settings.initial_bias_variance;
}

void PlanarDisplacementFilter::open_hop()
{
  state_.segment<2>(kAnchor) = state_.segment<2>(kPosition);
  covariance_.block<2, 8>(kAnchor, 0) = covariance_.block<2, 8>(kPosition, 0);
  covariance_.block<8, 2>(0, kAnchor) = covariance_.block<8, 2>(0, kPosition);
  covariance_.block<2, 2>(kAnchor, kAnchor) = covariance_.block<2, 2>(kPosition, kPosition);
}

void PlanarDisplacementFilter::predict(
  const Eigen::Vector2d & acceleration_world, double dt, double yaw)
{
  if (dt <= 0.0) {
    return;
  }
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix2d rotation;
  rotation << c, -s, s, c;
  const Eigen::Vector2d corrected = acceleration_world - rotation * state_.segment<2>(kBias);

  state_.segment<2>(kPosition) +=
    state_.segment<2>(kVelocity) * dt + 0.5 * corrected * dt * dt;
  state_.segment<2>(kVelocity) += corrected * dt;

  Covariance transition = Covariance::Identity();
  transition.block<2, 2>(kPosition, kVelocity) = Eigen::Matrix2d::Identity() * dt;
  transition.block<2, 2>(kPosition, kBias) = rotation * (-0.5 * dt * dt);
  transition.block<2, 2>(kVelocity, kBias) = rotation * (-dt);
  covariance_ = transition * covariance_ * transition.transpose();

  const double spectral = settings_.acceleration_noise * settings_.acceleration_noise;
  const Eigen::Matrix2d eye = Eigen::Matrix2d::Identity();
  covariance_.block<2, 2>(kPosition, kPosition) += eye * (spectral * dt * dt * dt / 3.0);
  covariance_.block<2, 2>(kPosition, kVelocity) += eye * (spectral * dt * dt / 2.0);
  covariance_.block<2, 2>(kVelocity, kPosition) += eye * (spectral * dt * dt / 2.0);
  covariance_.block<2, 2>(kVelocity, kVelocity) += eye * (spectral * dt);
  covariance_.block<2, 2>(kBias, kBias) += eye * (settings_.bias_walk * settings_.bias_walk * dt);
}

double PlanarDisplacementFilter::measurement_variance(int inliers, double spread) const
{
  const double count = std::max(static_cast<double>(inliers), 1.0);
  const double floor =
    settings_.vision_noise_m * settings_.vision_noise_m *
    (settings_.vision_reference_inliers / count);
  if (spread > 0.0) {
    return std::max(floor, spread * spread / count);
  }
  return floor;
}

bool PlanarDisplacementFilter::update(
  const Eigen::Vector2d & displacement_world, int inliers, double extra_variance,
  double spread)
{
  const double variance =
    measurement_variance(inliers, spread) + std::max(extra_variance, 0.0);

  // H picks out p - a.
  Eigen::Matrix<double, 2, 8> model = Eigen::Matrix<double, 2, 8>::Zero();
  model.block<2, 2>(0, kPosition) = Eigen::Matrix2d::Identity();
  model.block<2, 2>(0, kAnchor) = -Eigen::Matrix2d::Identity();

  const Eigen::Vector2d predicted =
    state_.segment<2>(kPosition) - state_.segment<2>(kAnchor);
  const Eigen::Vector2d innovation = displacement_world - predicted;
  const Eigen::Matrix2d prior = model * covariance_ * model.transpose();
  Eigen::Matrix2d block = prior + Eigen::Matrix2d::Identity() * variance;

  Eigen::Matrix2d inverse;
  bool invertible = false;
  double determinant = 0.0;
  block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible) {
    return false;
  }

  bool accepted = true;
  const double nis = innovation.dot(inverse * innovation);
  if (settled() && nis > settings_.innovation_gate) {
    ++rejected_;
    accepted = false;
    block = prior + Eigen::Matrix2d::Identity() * (variance * settings_.outlier_inflation);
    block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
    if (!invertible) {
      return false;
    }
  }

  const Eigen::Matrix<double, 8, 2> gain = covariance_ * model.transpose() * inverse;

  UpdateRecord record;
  record.measured = displacement_world;
  record.predicted = predicted;
  record.innovation = innovation;
  record.variance = variance;
  record.prior_trace = prior.trace();
  record.nis = nis;
  record.accepted = accepted;

  state_ += gain * innovation;
  // Joseph form. `P - K H P` is algebraically equal to it and numerically is
  // not: it subtracts two nearly equal matrices, and a filter that runs a few
  // hundred updates on eight states drifts out of positive definiteness that
  // way. Joseph stays symmetric and positive by construction at the cost of one
  // more product.
  const Covariance spread_matrix = Covariance::Identity() - gain * model;
  covariance_ = spread_matrix * covariance_ * spread_matrix.transpose() +
    gain * (Eigen::Matrix2d::Identity() * variance) * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();

  record.posterior = state_.segment<2>(kPosition) - state_.segment<2>(kAnchor);
  last_update_ = record;
  ++updates_;
  return accepted;
}

void PlanarDisplacementFilter::update_zero_velocity(double sigma)
{
  Eigen::Matrix<double, 2, 8> model = Eigen::Matrix<double, 2, 8>::Zero();
  model.block<2, 2>(0, kVelocity) = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d innovation = -state_.segment<2>(kVelocity);
  Eigen::Matrix2d block =
    model * covariance_ * model.transpose() + Eigen::Matrix2d::Identity() * (sigma * sigma);

  Eigen::Matrix2d inverse;
  bool invertible = false;
  double determinant = 0.0;
  block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible) {
    return;
  }
  const Eigen::Matrix<double, 8, 2> gain = covariance_ * model.transpose() * inverse;
  state_ += gain * innovation;
  const Covariance spread_matrix = Covariance::Identity() - gain * model;
  covariance_ = spread_matrix * covariance_ * spread_matrix.transpose() +
    gain * (Eigen::Matrix2d::Identity() * sigma * sigma) * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

Eigen::Vector2d PlanarDisplacementFilter::body_translation(double yaw) const
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const Eigen::Vector2d d = state_.segment<2>(kPosition) - state_.segment<2>(kAnchor);
  return Eigen::Vector2d(c * d.x() + s * d.y(), -s * d.x() + c * d.y());
}

namespace
{
// Slices of the MSCKF state, named where they are used.
constexpr int kP = 0;
constexpr int kV = 2;
constexpr int kT = 4;
constexpr int kBA = 5;
constexpr int kBG = 7;
constexpr int kPA = 8;
constexpr int kTA = 10;

double wrap(double angle)
{
  return std::remainder(angle, 2.0 * M_PI);
}
}  // namespace

PlanarMsckfFilter::PlanarMsckfFilter()
: PlanarMsckfFilter(Settings())
{
}

PlanarMsckfFilter::PlanarMsckfFilter(const Settings & settings)
: settings_(settings)
{
  covariance_.setZero();
  covariance_(kV, kV) = settings.initial_velocity_variance;
  covariance_(kV + 1, kV + 1) = settings.initial_velocity_variance;
  covariance_(kBA, kBA) = settings.initial_bias_variance;
  covariance_(kBA + 1, kBA + 1) = settings.initial_bias_variance;
  covariance_(kBG, kBG) = settings.initial_gyro_bias_variance;
}

void PlanarMsckfFilter::set_pose(const Eigen::Vector2d & position, double yaw)
{
  state_.segment<2>(kP) = position;
  state_(kT) = wrap(yaw);
}

void PlanarMsckfFilter::open_hop()
{
  state_.segment<2>(kPA) = state_.segment<2>(kP);
  state_(kTA) = state_(kT);
  covariance_.block<2, kSize>(kPA, 0) = covariance_.block<2, kSize>(kP, 0);
  covariance_.block<1, kSize>(kTA, 0) = covariance_.block<1, kSize>(kT, 0);
  covariance_.block<kSize, 2>(0, kPA) = covariance_.block<kSize, 2>(0, kP);
  covariance_.block<kSize, 1>(0, kTA) = covariance_.block<kSize, 1>(0, kT);
  // The two clones' cross term has to come from the originals as well, or the
  // anchor's pose and heading are correlated with everything except each other.
  covariance_.block<2, 1>(kPA, kTA) = covariance_.block<2, 1>(kP, kT);
  covariance_.block<1, 2>(kTA, kPA) = covariance_.block<1, 2>(kT, kP);
  covariance_.block<2, 2>(kPA, kPA) = covariance_.block<2, 2>(kP, kP);
  covariance_(kTA, kTA) = covariance_(kT, kT);
}

void PlanarMsckfFilter::predict(
  const Eigen::Vector2d & acceleration_body, double rate, double dt)
{
  if (dt <= 0.0) {
    return;
  }
  const Eigen::Vector2d corrected = acceleration_body - state_.segment<2>(kBA);
  const double turn = rate - state_(kBG);
  const double theta = state_(kT);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  Eigen::Matrix2d rotation;
  rotation << c, -s, s, c;
  // d/dtheta of the rotation, which is how a heading error becomes a position
  // one -- the term the older filter had no state to carry.
  Eigen::Matrix2d perpendicular;
  perpendicular << 0.0, -1.0, 1.0, 0.0;
  const Eigen::Matrix2d turned = rotation * perpendicular;
  const Eigen::Vector2d world = rotation * corrected;

  state_.segment<2>(kP) += state_.segment<2>(kV) * dt + 0.5 * world * dt * dt;
  state_.segment<2>(kV) += world * dt;
  state_(kT) = wrap(theta + turn * dt);

  Covariance transition = Covariance::Identity();
  transition.block<2, 2>(kP, kV) = Eigen::Matrix2d::Identity() * dt;
  transition.block<2, 1>(kP, kT) = (turned * corrected) * (0.5 * dt * dt);
  transition.block<2, 2>(kP, kBA) = rotation * (-0.5 * dt * dt);
  transition.block<2, 1>(kV, kT) = (turned * corrected) * dt;
  transition.block<2, 2>(kV, kBA) = rotation * (-dt);
  transition(kT, kBG) = -dt;
  covariance_ = transition * covariance_ * transition.transpose();

  const double spectral = settings_.acceleration_noise * settings_.acceleration_noise;
  const Eigen::Matrix2d eye = Eigen::Matrix2d::Identity();
  covariance_.block<2, 2>(kP, kP) += eye * (spectral * dt * dt * dt / 3.0);
  covariance_.block<2, 2>(kP, kV) += eye * (spectral * dt * dt / 2.0);
  covariance_.block<2, 2>(kV, kP) += eye * (spectral * dt * dt / 2.0);
  covariance_.block<2, 2>(kV, kV) += eye * (spectral * dt);
  covariance_(kT, kT) += settings_.gyro_noise * settings_.gyro_noise * dt;
  covariance_.block<2, 2>(kBA, kBA) += eye * (settings_.bias_walk * settings_.bias_walk * dt);
  covariance_(kBG, kBG) += settings_.gyro_bias_walk * settings_.gyro_bias_walk * dt;
}

Eigen::Matrix3d PlanarMsckfFilter::measurement_covariance(
  int inliers, double spread, double extra_variance, double yaw_sigma) const
{
  const double count = std::max(static_cast<double>(inliers), 1.0);
  const double floor = settings_.vision_noise_m * settings_.vision_noise_m *
    (settings_.vision_reference_inliers / count);
  double position = spread <= 0.0 ? floor : std::max(floor, spread * spread / count);
  position += std::max(extra_variance, 0.0);
  const double yaw = (!std::isfinite(yaw_sigma) || yaw_sigma <= 0.0)
    ? settings_.vision_yaw_noise * settings_.vision_yaw_noise
    : yaw_sigma * yaw_sigma;
  Eigen::Matrix3d noise = Eigen::Matrix3d::Zero();
  noise(0, 0) = position;
  noise(1, 1) = position;
  noise(2, 2) = yaw;
  return noise;
}

void PlanarMsckfFilter::observation(
  Eigen::Matrix<double, 3, kSize> & model, Eigen::Vector2d & predicted) const
{
  const double theta_a = state_(kTA);
  const double c = std::cos(theta_a);
  const double s = std::sin(theta_a);
  Eigen::Matrix2d inverse;
  inverse << c, s, -s, c;
  const Eigen::Vector2d offset = state_.segment<2>(kP) - state_.segment<2>(kPA);

  model.setZero();
  model.block<2, 2>(0, kP) = inverse;
  model.block<2, 2>(0, kPA) = -inverse;
  Eigen::Matrix2d perpendicular;
  perpendicular << 0.0, 1.0, -1.0, 0.0;
  model.block<2, 1>(0, kTA) = perpendicular * inverse * offset;
  model(2, kT) = 1.0;
  model(2, kTA) = -1.0;
  predicted = inverse * offset;
}

void PlanarMsckfFilter::apply(
  const Eigen::Matrix<double, 3, kSize> & model, const Eigen::Vector3d & innovation,
  const Eigen::Matrix3d & noise, const Eigen::Matrix3d & inverse)
{
  const Eigen::Matrix<double, kSize, 3> gain =
    covariance_ * model.transpose() * inverse;
  state_ += gain * innovation;
  state_(kT) = wrap(state_(kT));
  state_(kTA) = wrap(state_(kTA));
  const Covariance spread = Covariance::Identity() - gain * model;
  covariance_ = spread * covariance_ * spread.transpose() +
    gain * noise * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

bool PlanarMsckfFilter::update(
  const Eigen::Vector2d & displacement_body, double yaw_delta, int inliers,
  double extra_variance, double spread, double yaw_sigma)
{
  Eigen::Matrix<double, 3, kSize> model;
  Eigen::Vector2d predicted_position;
  observation(model, predicted_position);

  Eigen::Vector3d innovation;
  innovation.head<2>() = displacement_body - predicted_position;
  innovation(2) = wrap(yaw_delta - wrap(state_(kT) - state_(kTA)));

  UpdateRecord record;
  record.innovation = innovation;

  // A solve that is wrong by metres is not a measurement to down-weight. The
  // inflation exists so a merely surprising hop still tethers the integral;
  // beyond this the two are not describing the same vehicle, and the estimator
  // has its own gate to hold the pose while the filter waits.
  if (settled() && innovation.head<2>().norm() > settings_.reject_beyond_m) {
    ++dropped_;
    record.dropped = true;
    record.accepted = false;
    record.nis = std::numeric_limits<double>::infinity();
    last_update_ = record;
    return false;
  }

  Eigen::Matrix3d noise =
    measurement_covariance(inliers, spread, extra_variance, yaw_sigma);
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
      record.accepted = false;
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

void PlanarMsckfFilter::update_zero_velocity(double sigma, double yaw_sigma)
{
  Eigen::Matrix<double, 3, kSize> model = Eigen::Matrix<double, 3, kSize>::Zero();
  model.block<2, 2>(0, kV) = Eigen::Matrix2d::Identity();
  model(2, kT) = 1.0;
  model(2, kTA) = -1.0;

  Eigen::Vector3d innovation;
  innovation.head<2>() = -state_.segment<2>(kV);
  innovation(2) = -wrap(state_(kT) - state_(kTA));

  Eigen::Matrix3d noise = Eigen::Matrix3d::Zero();
  noise(0, 0) = sigma * sigma;
  noise(1, 1) = sigma * sigma;
  noise(2, 2) = yaw_sigma * yaw_sigma;

  Eigen::Matrix3d block = model * covariance_ * model.transpose() + noise;
  Eigen::Matrix3d inverse;
  bool invertible = false;
  double determinant = 0.0;
  block.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (!invertible) {
    return;
  }
  apply(model, innovation, noise, inverse);
}

void PlanarMsckfFilter::update_heading(double measured_yaw, double sigma)
{
  if (!(sigma > 0.0) || !std::isfinite(sigma)) {
    return;
  }
  Eigen::Matrix<double, 1, kSize> model = Eigen::Matrix<double, 1, kSize>::Zero();
  model(0, kT) = 1.0;
  const double innovation = wrap(measured_yaw - state_(kT));
  const double block = (model * covariance_ * model.transpose())(0, 0) + sigma * sigma;
  if (!(block > 0.0)) {
    return;
  }
  const Eigen::Matrix<double, kSize, 1> gain = covariance_ * model.transpose() / block;
  state_ += gain * innovation;
  state_(kT) = wrap(state_(kT));
  state_(kTA) = wrap(state_(kTA));
  const Covariance spread = Covariance::Identity() - gain * model;
  covariance_ = spread * covariance_ * spread.transpose() +
    gain * (sigma * sigma) * gain.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

Eigen::Vector2d PlanarMsckfFilter::body_translation() const
{
  Eigen::Matrix<double, 3, kSize> model;
  Eigen::Vector2d predicted;
  observation(model, predicted);
  return predicted;
}

double PlanarMsckfFilter::hop_yaw() const
{
  return wrap(state_(kT) - state_(kTA));
}

}  // namespace monoscale
