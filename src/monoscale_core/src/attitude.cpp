#include "monoscale_core/attitude.hpp"

#include <algorithm>
#include <cmath>

#include "monoscale_core/geometry.hpp"

namespace monoscale
{

AttitudeFilter::AttitudeFilter(double tau, double tolerance, double gravity)
: tau_(tau), tolerance_(tolerance), gravity_(gravity)
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
  roll_ += dt * (gyro.x() + gyro.y() * std::sin(roll_) * tan_pitch +
    gyro.z() * std::cos(roll_) * tan_pitch);
  pitch_ += dt * (gyro.y() * std::cos(roll_) - gyro.z() * std::sin(roll_));

  const double magnitude = std::sqrt(ax * ax + ay * ay + az * az);
  if (std::abs(magnitude - gravity_) > tolerance_) {
    return;
  }
  ++corrections_;
  const double gain = std::min(dt / tau_, 1.0);
  roll_ += gain * (level_roll - roll_);
  pitch_ += gain * (level_pitch - pitch_);
}

Eigen::Matrix3d AttitudeFilter::body_tilt() const
{
  const double cr = std::cos(roll_);
  const double sr = std::sin(roll_);
  const double cp = std::cos(pitch_);
  const double sp = std::sin(pitch_);
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
  error_ += rate_ * dt;
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
