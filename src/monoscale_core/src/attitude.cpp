#include "monoscale_core/attitude.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "monoscale_core/geometry.hpp"

namespace monoscale
{

AttitudeFilter::AttitudeFilter(
  double tau, double tolerance, double gravity, double bias_tau, double bias_limit,
  double bias_gate)
: tau_(tau), tolerance_(tolerance), gravity_(gravity),
  bias_tau_(bias_tau), bias_limit_(bias_limit), bias_gate_(bias_gate)
{
}

void AttitudeFilter::update(
  const Eigen::Vector3d & gyro, const Eigen::Vector3d & accel, double dt)
{
  const double ax = accel.x();
  const double ay = accel.y();
  const double az = accel.z();
  const double level_roll = std::atan2(ay, az);
  const double level_pitch = std::atan2(-ax, std::hypot(ay, az));

  if (!started_) {
    // Level, not whatever the first sample happens to say. A vehicle on a road
    // is within a degree or two of level and that is a far better prior than
    // one accelerometer reading: seeding from the first sample of a drive that
    // opens under throttle starts the pitch 15 degrees out, and a sixty second
    // trim never recovers inside a thirty second drive. It scored 18.9 m.
    roll_ = 0.0;
    pitch_ = 0.0;
    started_ = true;
    return;
  }
  if (dt <= 0.0) {
    return;
  }

  // Body rates resolved through the current attitude. At the angles a road
  // vehicle reaches these are the exact Euler rates and the small angle form
  // would do just as well.
  const double tan_pitch = std::tan(pitch_);
  roll_ += dt * ((gyro.x() - roll_bias_) + gyro.y() * std::sin(roll_) * tan_pitch +
    gyro.z() * std::cos(roll_) * tan_pitch);
  pitch_ += dt * ((gyro.y() - pitch_bias_) * std::cos(roll_) -
    gyro.z() * std::sin(roll_));

  // What corrupts the apparent vertical is the vehicle's own acceleration, and
  // the magnitude gate is the wrong instrument for it: a horizontal `a` moves
  // |accel| by only a^2/2g while tilting the apparent vertical by atan(a/g).
  // The deployed 0.10 m/s2 therefore still admits a = 1.40 and 8.1 degrees of
  // false tilt. Take the estimated gravity out of the specific force and gate
  // on what is left, which is the acceleration itself.
  if (horizontal_tolerance_ > 0.0) {
    const double cr = std::cos(roll_);
    const double sr = std::sin(roll_);
    const double cp = std::cos(pitch_);
    const double sp = std::sin(pitch_);
    const Eigen::Vector3d down(-sp, sr * cp, cr * cp);
    const Eigen::Vector3d own = Eigen::Vector3d(ax, ay, az) - gravity_ * down;
    if (own.head<2>().norm() > horizontal_tolerance_) {
      return;
    }
  } else {
    const double magnitude = std::sqrt(ax * ax + ay * ay + az * az);
    if (std::abs(magnitude - gravity_) > tolerance_) {
      return;
    }
  }
  ++corrections_;
  const double gain = std::min(dt / tau_, 1.0);
  const double roll_error = level_roll - roll_;
  const double pitch_error = level_pitch - pitch_;
  // Where the trim keeps having to pull the same way, the rate is biased.
  const bool steady = bias_gate_ <= 0.0 ||
    (std::abs(roll_error) < bias_gate_ && std::abs(pitch_error) < bias_gate_);
  if (bias_tau_ > 0.0 && tau_ > 0.0 && steady) {
    const double rate = dt / (tau_ * bias_tau_);
    const double limit = bias_limit_ > 0.0 ? bias_limit_ :
      std::numeric_limits<double>::infinity();
    roll_bias_ = std::clamp(roll_bias_ - rate * roll_error, -limit, limit);
    pitch_bias_ = std::clamp(pitch_bias_ - rate * pitch_error, -limit, limit);
  }
  roll_ += gain * roll_error;
  pitch_ += gain * pitch_error;

  if (slope_tau_ > 0.0) {
    if (!slope_started_) {
      slope_roll_ = roll_;
      slope_pitch_ = pitch_;
      slope_started_ = true;
    } else {
      const double slow = std::min(dt / slope_tau_, 1.0);
      slope_roll_ += slow * (roll_ - slope_roll_);
      slope_pitch_ += slow * (pitch_ - slope_pitch_);
    }
  }
}

Eigen::Matrix3d AttitudeFilter::body_tilt() const
{
  // What the projection wants is the body against the ROAD. The filter tracks
  // it against gravity, and the difference is the road's slope, which the slow
  // running average below holds.
  const double roll = slope_tau_ > 0.0 ? roll_ - slope_roll_ : roll_;
  const double pitch = slope_tau_ > 0.0 ? pitch_ - slope_pitch_ : pitch_;
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  Eigen::Matrix3d tilt;
  tilt <<
    cp, sp * sr, sp * cr,
    0.0, cr, -sr,
    -sp, cp * sr, cp * cr;
  return tilt;
}

HeadingBiasFilter::HeadingBiasFilter(double bias_sigma, double walk_sigma, double noise_sigma)
: walk_(walk_sigma * walk_sigma),
  noise_(noise_sigma * noise_sigma),
  enabled_(bias_sigma > 0.0)
{
  // No idea which way the bias points, every idea of how big it is.
  covariance_(1, 1) = bias_sigma * bias_sigma;
}

void HeadingBiasFilter::predict(double dt)
{
  if (!enabled_ || dt <= 0.0) {
    return;
  }
  error_ += (1.0 - corrected_) * rate_ * dt;
  Eigen::Matrix2d transition;
  transition << 1.0, dt, 0.0, 1.0;
  covariance_ = transition * covariance_ * transition.transpose();
  covariance_(0, 0) += noise_ * dt;
  covariance_(1, 1) += walk_ * dt;
}

double HeadingBiasFilter::update(double residual, double sigma)
{
  if (!enabled_ || !std::isfinite(sigma) || sigma <= 0.0) {
    return 0.0;
  }
  const double innovation = residual - error_;
  const double variance = covariance_(0, 0) + sigma * sigma;
  if (variance <= 0.0) {
    return 0.0;
  }
  const Eigen::Vector2d gain = covariance_.col(0) / variance;
  error_ += gain(0) * innovation;
  rate_ += gain(1) * innovation;
  Eigen::Matrix2d reduction = Eigen::Matrix2d::Identity();
  reduction -= gain * Eigen::RowVector2d(1.0, 0.0);
  covariance_ = reduction * covariance_;
  const double applied = error_;
  // Injected into the pose, so the error state starts again from zero while
  // what it taught us about the rate stays.
  error_ = 0.0;
  return applied;
}

}  // namespace monoscale
