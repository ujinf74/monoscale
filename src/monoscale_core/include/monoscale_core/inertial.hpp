// Short-horizon translation prediction from the IMU.
//
// The estimator solved every frame from scratch, with nothing to check the
// answer against. When tracking degraded the solve could return "barely moved"
// and there was no way to know it was wrong; above about 2 m/s that is exactly
// what happened.
//
// Integrating the accelerometer between two camera frames gives an independent
// prediction of how far the vehicle travelled. Over the 50 ms between frames
// the integration is trustworthy; over a minute it is not, which is why vision
// corrects the velocity every time it produces an answer worth believing.
//
// This deliberately does not supply scale. Scale stays with the ground plane,
// which is observable at constant velocity where an accelerometer is not.

#ifndef MONOSCALE_CORE__INERTIAL_HPP_
#define MONOSCALE_CORE__INERTIAL_HPP_

#include <deque>
#include <optional>
#include <string>

#include <Eigen/Dense>

namespace monoscale
{

enum class Integration
{
  // Zero-order hold: the acceleration reported at the start of the step is
  // taken to hold across it.
  Zoh,
  // Linear interpolation between adjacent samples. For x'=v, v'=a that gives
  //   x1 = x0 + v0*dt + dt^2*(a0/3 + a1/6)
  //   v1 = v0 + dt*(a0 + a1)/2
  // which is exact for that interpolation model.
  Rk4,
};

Integration integration_from_name(const std::string & name);

// Dead reckons horizontal motion, and takes velocity corrections.
class PlanarInertialPropagator
{
public:
  struct Settings
  {
    double gravity = 9.81;
    double max_gap_sec = 0.25;
    double history_sec = 4.0;
    double max_horizontal_acceleration = 12.0;
    int median_window = 1;
    Integration integration = Integration::Rk4;
  };

  struct Step
  {
    // The interval-effective horizontal acceleration, so a filter downstream
    // can propagate with the same integration method.
    Eigen::Vector2d acceleration = Eigen::Vector2d::Zero();
    double dt = 0.0;
  };

  PlanarInertialPropagator();
  explicit PlanarInertialPropagator(const Settings & settings);

  // Integrate one IMU sample. Orientation is (x, y, z, w).
  Step add_sample(
    double stamp, const Eigen::Vector4d & orientation, const Eigen::Vector3d & acceleration);

  // Translation between two times, expressed in the body frame at `yaw`.
  std::optional<Eigen::Vector2d> predicted_translation(
    double start, double end, double yaw) const;

  // Fold a vision-derived velocity into the integrated one. Without this the
  // integral walks off within seconds even on a perfect accelerometer, because
  // nothing observes the constant of integration.
  void correct_velocity(const Eigen::Vector2d & velocity_world, double gain = 1.0);

  bool corrected() const {return corrected_;}
  const Eigen::Vector2d & velocity() const {return velocity_;}
  const Eigen::Vector2d & position() const {return position_;}
  int rejected_samples() const {return rejected_samples_;}

private:
  struct Sample
  {
    double stamp;
    Eigen::Vector2d position;
    Eigen::Vector2d velocity;
  };

  std::optional<Sample> sample_at(double stamp) const;

  Settings settings_;
  std::deque<Eigen::Vector2d> window_;
  std::optional<Eigen::Vector2d> previous_acceleration_;
  int rejected_samples_ = 0;
  Eigen::Vector2d velocity_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d position_ = Eigen::Vector2d::Zero();
  std::optional<double> last_stamp_;
  // An integral nobody has anchored has an unknown constant. Spawning the
  // vehicle drops it onto the road, and that transient integrates into a
  // velocity that never existed, so predictions stay unavailable until vision
  // has pinned the velocity down at least once.
  bool corrected_ = false;
  std::deque<Sample> history_;
};

// Whether a vision translation is close enough to the inertial one.
//
// Only judges when the prediction carries information. A vehicle at constant
// velocity produces no horizontal acceleration, so a propagator that has not
// yet been told its velocity predicts nothing; gating on that would reject
// perfectly good vision, including the first solve after startup.
//
// Generous on purpose otherwise: this exists to catch the collapse to near zero
// motion, not to second-guess a working solve.
bool agrees_with_prediction(
  const Eigen::Vector2d & vision, const std::optional<Eigen::Vector2d> & predicted,
  double absolute_tolerance, double relative_tolerance, double minimum_prediction = 0.05);

}  // namespace monoscale

#endif  // MONOSCALE_CORE__INERTIAL_HPP_
