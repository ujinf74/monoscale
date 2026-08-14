#include "monoscale_core/inertial.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace monoscale
{

namespace
{

Eigen::Matrix3d rotation_from_quaternion(const Eigen::Vector4d & q)
{
  const double x = q(0);
  const double y = q(1);
  const double z = q(2);
  const double w = q(3);
  const double norm = x * x + y * y + z * z + w * w;
  if (norm < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  const double s = 2.0 / norm;
  Eigen::Matrix3d rotation;
  rotation <<
    1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
    s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w),
    s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y);
  return rotation;
}

double median_of(std::vector<double> & values)
{
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2 == 1) {
    return upper;
  }
  const double lower = *std::max_element(values.begin(), values.begin() + middle);
  return 0.5 * (lower + upper);
}

}  // namespace

Integration integration_from_name(const std::string & name)
{
  if (name == "zoh") {
    return Integration::Zoh;
  }
  if (name == "rk4") {
    return Integration::Rk4;
  }
  throw std::invalid_argument("integration_method must be 'zoh' or 'rk4'");
}

PlanarInertialPropagator::PlanarInertialPropagator()
: PlanarInertialPropagator(Settings())
{
}

PlanarInertialPropagator::PlanarInertialPropagator(const Settings & settings)
: settings_(settings)
{
  settings_.max_horizontal_acceleration = std::max(settings_.max_horizontal_acceleration, 0.0);
  settings_.median_window = std::max(settings_.median_window, 1);
}

PlanarInertialPropagator::Step PlanarInertialPropagator::add_sample(
  double stamp, const Eigen::Vector4d & orientation, const Eigen::Vector3d & acceleration)
{
  Eigen::Vector3d world = rotation_from_quaternion(orientation) * acceleration;
  // The accelerometer measures proper acceleration, so at rest it reads gravity
  // pointing up. Remove it before integrating.
  world.z() -= settings_.gravity;
  const Eigen::Vector2d horizontal = world.head<2>();

  // CARLA occasionally reports physics impulses of hundreds or thousands of
  // m/s2 when the suspension contacts or the actor is corrected. Such a sample
  // is not useful vehicle motion and corrupts velocity for many frames after a
  // single 10 ms integration step. Reject it before the short causal median
  // used for the remaining contact noise.
  if (settings_.max_horizontal_acceleration > 0.0 &&
    horizontal.norm() > settings_.max_horizontal_acceleration)
  {
    ++rejected_samples_;
  } else {
    window_.push_back(horizontal);
    while (static_cast<int>(window_.size()) > settings_.median_window) {
      window_.pop_front();
    }
  }

  Eigen::Vector2d filtered = Eigen::Vector2d::Zero();
  if (!window_.empty()) {
    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(window_.size());
    ys.reserve(window_.size());
    for (const auto & sample : window_) {
      xs.push_back(sample.x());
      ys.push_back(sample.y());
    }
    filtered = Eigen::Vector2d(median_of(xs), median_of(ys));
  }

  // Do not integrate the spawn/drop transient before vision supplies the
  // unknown constant velocity. This also keeps the separate velocity filter
  // from receiving unanchored acceleration at startup.
  const Eigen::Vector2d usable = corrected_ ? filtered : Eigen::Vector2d::Zero();

  Step step;
  step.acceleration = usable;
  if (last_stamp_.has_value()) {
    const double dt = stamp - *last_stamp_;
    if (dt > 0.0 && dt <= settings_.max_gap_sec) {
      if (settings_.integration == Integration::Rk4 && previous_acceleration_.has_value()) {
        const Eigen::Vector2d previous = *previous_acceleration_;
        step.acceleration = 0.5 * (previous + usable);
        position_ += velocity_ * dt + dt * dt * (previous / 3.0 + usable / 6.0);
        velocity_ += 0.5 * (previous + usable) * dt;
      } else {
        position_ += velocity_ * dt + 0.5 * usable * dt * dt;
        velocity_ += usable * dt;
      }
    } else if (dt > settings_.max_gap_sec) {
      // A gap this long makes the integral meaningless; start clean.
      velocity_.setZero();
      previous_acceleration_.reset();
    }
  }

  const double raw_step = last_stamp_.has_value() ? stamp - *last_stamp_ : 0.0;
  step.dt = (raw_step > 0.0 && raw_step <= settings_.max_gap_sec) ? raw_step : 0.0;
  last_stamp_ = stamp;
  previous_acceleration_ = usable;

  history_.push_back(Sample{stamp, position_, velocity_});
  while (!history_.empty() && stamp - history_.front().stamp > settings_.history_sec) {
    history_.pop_front();
  }
  return step;
}

std::optional<PlanarInertialPropagator::Sample> PlanarInertialPropagator::sample_at(
  double stamp) const
{
  if (history_.empty()) {
    return std::nullopt;
  }
  if (stamp <= history_.front().stamp) {
    return history_.front();
  }
  Sample previous = history_.front();
  for (const auto & entry : history_) {
    if (entry.stamp >= stamp) {
      const double span = entry.stamp - previous.stamp;
      if (span <= 0.0) {
        return entry;
      }
      const double ratio = (stamp - previous.stamp) / span;
      return Sample{
        stamp,
        previous.position + ratio * (entry.position - previous.position),
        previous.velocity + ratio * (entry.velocity - previous.velocity)};
    }
    previous = entry;
  }
  return history_.back();
}

std::optional<Eigen::Vector2d> PlanarInertialPropagator::predicted_translation(
  double start, double end, double yaw) const
{
  if (!corrected_ || end <= start) {
    return std::nullopt;
  }
  const auto first = sample_at(start);
  const auto second = sample_at(end);
  if (!first.has_value() || !second.has_value()) {
    return std::nullopt;
  }
  const Eigen::Vector2d delta = second->position - first->position;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return Eigen::Vector2d(c * delta.x() + s * delta.y(), -s * delta.x() + c * delta.y());
}

void PlanarInertialPropagator::correct_velocity(
  const Eigen::Vector2d & velocity_world, double gain)
{
  gain = std::clamp(gain, 0.0, 1.0);
  corrected_ = true;
  velocity_ += gain * (velocity_world - velocity_);
  if (!history_.empty()) {
    history_.back().velocity = velocity_;
  }
}

bool agrees_with_prediction(
  const Eigen::Vector2d & vision, const std::optional<Eigen::Vector2d> & predicted,
  double absolute_tolerance, double relative_tolerance, double minimum_prediction)
{
  if (!predicted.has_value()) {
    return true;
  }
  const double reach = predicted->norm();
  if (reach < minimum_prediction) {
    return true;
  }
  const double tolerance = absolute_tolerance + relative_tolerance * reach;
  return (vision - *predicted).norm() <= tolerance;
}

}  // namespace monoscale
