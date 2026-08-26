// Velocity and displacement fused from the accelerometer and from vision.
//
// The arrangement this replaces was a switch: believe vision, or throw it away
// and dead reckon. That discards the thing both sources actually have, which is
// a partial and quantifiable opinion about how fast the vehicle is going.
//
// Here the accelerometer propagates and grows its uncertainty, and each vision
// solve arrives as a measurement whose noise falls as its inlier count rises. A
// solve backed by five hundred agreeing ground points moves the estimate; one
// backed by forty barely does.
//
// Scale still comes from the ground plane. Neither filter invents scale of its
// own.

#ifndef MONOSCALE_CORE__FUSION_HPP_
#define MONOSCALE_CORE__FUSION_HPP_

#include <optional>

#include <Eigen/Dense>

namespace monoscale
{

// A two state Kalman filter over world-frame planar velocity. The older of the
// two models: it takes the vision translation divided by the interval as a
// velocity measurement, which carries the interval's own error into the
// measurement and treats an average over the interval as the value at its end.
// Kept because every measurement before the displacement model was taken with
// it, and because it is what runs when `fusion_model` says so.
class PlanarVelocityFilter
{
public:
  struct Settings
  {
    // Spectral density of the acceleration the model does not know about, in
    // m/s^2. Larger means the filter leans on vision sooner.
    double acceleration_noise = 1.55;
    // Velocity noise of a vision solve carrying the reference inlier count.
    double vision_noise = 0.25;
    double vision_reference_inliers = 300.0;
    double initial_variance = 25.0;
    double innovation_gate = 9.0;
    // A measurement that fails the consistency check is not discarded, its
    // variance is inflated. Discarding leaves the filter on dead reckoning
    // alone, and an integral with nothing observing it walks away: it read
    // 14 m/s while the vehicle did 6.
    double outlier_inflation = 25.0;
  };

  // Two declarations rather than a default argument: a default argument that
  // names the nested Settings is parsed before that class is complete.
  PlanarVelocityFilter();
  explicit PlanarVelocityFilter(const Settings & settings);

  // Whether vision has pinned the velocity down at least once. Before that the
  // covariance still carries the startup guess, and any prediction built on it
  // says more about the initial variance than about the vehicle.
  bool settled() const {return updates_ > 0;}
  const Eigen::Vector2d & velocity() const {return velocity_;}
  int updates() const {return updates_;}
  int rejected() const {return rejected_;}

  void predict(const Eigen::Vector2d & acceleration_world, double dt);

  // Vision noise falls as the supporting inlier count rises.
  double measurement_variance(int inliers) const;

  // Fold in a vision velocity. Returns false if it was gated out.
  //
  // `extra_variance` carries evidence the solve itself cannot see. The front
  // and rear cameras look at different ground and solve independently, so how
  // far apart their answers are is a measure of how much either can be trusted,
  // and no amount of internal agreement inside one of them reveals it.
  bool update(const Eigen::Vector2d & measured, int inliers, double extra_variance = 0.0);

  // Travel over dt at the fused velocity, in the body frame at `yaw`.
  Eigen::Vector2d body_translation(double dt, double yaw) const;

  double speed() const {return velocity_.norm();}
  double uncertainty() const {return std::sqrt(std::max(covariance_.trace(), 0.0));}

private:
  Settings settings_;
  Eigen::Vector2d velocity_ = Eigen::Vector2d::Zero();
  Eigen::Matrix2d covariance_;
  int updates_ = 0;
  int rejected_ = 0;
};

// Turn a body-frame frame-to-frame translation into world velocity.
std::optional<Eigen::Vector2d> world_velocity_from_motion(
  double motion_x, double motion_y, double dt, double yaw);

// Fuse what each sensor measures: acceleration, and displacement between two
// image times.
//
// The velocity filter divides the vision translation by the interval and calls
// the quotient a velocity measurement. That inflates its noise by 1/dt^2 and
// treats an average over the interval as the value at its end -- 15 cm of
// modelling error per update at 0.1 s hops and 3 m/s. It also has no bias
// state, so any accelerometer offset integrates into velocity and only vision
// removes it.
//
// The fix is not simply to measure displacement instead. Zeroing the
// displacement at every hop reproduces vision alone almost to the millimetre,
// because resetting the anchor throws away every correlation the previous hop
// earned. So the anchor is a state, cloned from the current position when the
// hop opens and carried with its cross-covariance -- stochastic cloning, which
// exists for exactly this. Vision measures p - a; the correlation between them
// is what lets one hop inform the next.
//
// State: [position, anchor, velocity, bias], all planar, the bias in the body
// frame because that is where an instrument's offset lives.
class PlanarDisplacementFilter
{
public:
  struct Settings
  {
    double acceleration_noise = 1.55;
    double bias_walk = 0.05;
    double vision_noise_m = 0.02;
    double vision_reference_inliers = 300.0;
    double initial_velocity_variance = 25.0;
    double initial_bias_variance = 1.0;
    double innovation_gate = 9.0;
    double outlier_inflation = 25.0;
  };

  // What the filter claimed on the last update, kept for the run that checks
  // whether it was true. A filter is only as good as its covariance: if the
  // innovation is routinely larger than the S it predicts, the noise model is
  // wrong and no amount of searching over q finds the right answer, it only
  // finds the value that hurts least.
  struct UpdateRecord
  {
    Eigen::Vector2d measured = Eigen::Vector2d::Zero();
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    Eigen::Vector2d innovation = Eigen::Vector2d::Zero();
    Eigen::Vector2d posterior = Eigen::Vector2d::Zero();
    double variance = 0.0;
    double prior_trace = 0.0;
    double nis = 0.0;
    bool accepted = false;
  };

  PlanarDisplacementFilter();
  explicit PlanarDisplacementFilter(const Settings & settings);

  bool settled() const {return updates_ > 0;}
  Eigen::Vector2d velocity() const {return state_.segment<2>(4);}
  Eigen::Vector2d bias() const {return state_.segment<2>(6);}
  int updates() const {return updates_;}
  int rejected() const {return rejected_;}
  const std::optional<UpdateRecord> & last_update() const {return last_update_;}

  // Clone the current position as the anchor the next hop measures from.
  void open_hop();

  // Propagate over dt. `yaw` orients the bias, which is body-fixed.
  void predict(const Eigen::Vector2d & acceleration_world, double dt, double yaw = 0.0);

  // How well this solve knows the displacement, in metres squared.
  //
  // The inlier count alone is a proxy: it says how many votes there were, not
  // how far apart they landed. The solve already measures the second thing --
  // `spread` is the RMS residual of the inlying votes -- and the standard error
  // of their mean is spread^2/N. Through a turn the votes can be many and still
  // disagree, which is precisely the case the count cannot see. The configured
  // noise stays as a floor: votes that happen to agree closely are not thereby
  // exact, they can still be wrong together.
  double measurement_variance(int inliers, double spread = 0.0) const;

  bool update(
    const Eigen::Vector2d & displacement_world, int inliers, double extra_variance = 0.0,
    double spread = 0.0);

  // The vehicle is standing still. Say so.
  //
  // This is where the accelerometer bias becomes observable: at rest the
  // instrument still reports something, and whatever velocity that something
  // integrates into is the bias. Without it the bias state has nothing to
  // separate it from real motion. A parking manoeuvre stops several times, so
  // the information is there for the taking.
  void update_zero_velocity(double sigma = 0.01);

  // The fused displacement of this hop, in the body frame at `yaw`.
  Eigen::Vector2d body_translation(double yaw) const;

  double speed() const {return state_.segment<2>(4).norm();}

private:
  using State = Eigen::Matrix<double, 8, 1>;
  using Covariance = Eigen::Matrix<double, 8, 8>;

  Settings settings_;
  State state_ = State::Zero();
  Covariance covariance_ = Covariance::Zero();
  int updates_ = 0;
  int rejected_ = 0;
  std::optional<UpdateRecord> last_update_;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__FUSION_HPP_
