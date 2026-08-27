// Attitude the planar solve cannot see, and the heading bias it earns.
//
// Roll and pitch are carried by the gyro and pulled back towards gravity; yaw
// is left alone because the ground registration owns it, and estimating it here
// as well would rotate the ground twice.

#ifndef MONOSCALE_CORE__ATTITUDE_HPP_
#define MONOSCALE_CORE__ATTITUDE_HPP_

#include <Eigen/Dense>

namespace monoscale
{

// Roll and pitch: carried by the gyro, pulled back towards gravity.
//
// The ground projection assumes the body is level, and one degree of tilt moves
// every range in the frame by two and a half to five per cent. Neither
// instrument can supply the attitude on its own. The gyro integrates cleanly
// but has nothing to hold it to, and the accelerometer measures gravity plus
// whatever the vehicle is doing and cannot tell a braking vehicle from a
// nose-down one.
//
// Measured against the simulator's own attitude over a drive with hard braking
// and hard steering, weighting these wrongly is far worse than not trying:
// correcting on every sample with a one second constant left pitch out by 4.3
// degrees against a signal of 0.04. The gyro has to carry it and the
// accelerometer has to be a slow trim on top -- a sixty second constant,
// applied only while the measured acceleration is within 0.3 m/s2 of gravity's
// own -- which lands at 0.046 degrees.
//
// The trim above is proportional only, and a proportional loop against a biased
// rate settles at `bias * tau` rather than at zero. Measured against the truth
// attitude that offset is real and it is the estimator's dominant error: the
// pitch it hands the projection is out by 0.17 to 0.37 degrees on the straight
// drives, it scales exactly with tau (0.067 / 0.127 / 0.234 / 0.387 at 2 / 4 /
// 8 / 16 s), and it differs between three recordings of one drive -- so it is
// not calibration, it is a per-run gyro bias of up to 0.05 deg/s.
//
// It matters because a pitch error is a common-mode scale error on every range
// in the frame, with opposite sign on a forward and a rearward camera. It
// explains 67% of the variance in the measured front/rear travel split at
// -23 % per degree, and that split is 4 to 11 per cent while the per-point
// scatter accounts for under 2 per cent of the solve's error.
//
// So the bias is a state. This is the argument the heading filter below already
// makes and wins -- a bias is not noise, and it has to be a state or it cannot
// be cancelled -- applied one axis up, where it had not been.
class AttitudeFilter
{
public:
  // `bias_tau` is the integral's own constant; zero leaves the loop
  // proportional. `bias_limit` bounds the estimate to what a MEMS gyro can
  // plausibly carry: without it the integral chases the vehicle's own
  // acceleration through a manoeuvre and winds up to rates no instrument has
  // (measured, -0.76 deg/s on a curve, against real biases under 0.05).
  //
  // `bias_gate` is what admits a sample to the integral, and it is a direction
  // and not a magnitude. The magnitude gate above is the wrong quantity for
  // this: a lateral acceleration `a` moves |f| by only a^2/2g while tilting the
  // apparent vertical by atan(a/g), so at 0.3 it admits 13.9 degrees of false
  // tilt. The proportional trim survives that because it is a weak pull. An
  // integral does not -- it accumulates the lie. Gating instead on how far the
  // apparent vertical is from what is already held separates a biased rate,
  // whose innovation is a fraction of a degree, from a manoeuvre, whose
  // innovation is degrees.
  AttitudeFilter(
    double tau, double tolerance, double gravity = 9.80665,
    double bias_tau = 0.0, double bias_limit = 0.0, double bias_gate = 0.0);

  void update(const Eigen::Vector3d & gyro, const Eigen::Vector3d & accel, double dt);

  bool started() const {return started_;}
  double roll() const {return roll_;}
  double pitch() const {return pitch_;}
  double roll_bias() const {return roll_bias_;}
  double pitch_bias() const {return pitch_bias_;}
  int corrections() const {return corrections_;}

  // Roll and pitch as a rotation from the body into a level frame. Yaw is left
  // out: the planar solve owns it, and putting it here as well would rotate the
  // ground twice.
  // How slowly the road's own slope is tracked and taken back out. The vehicle
  // follows the road, so the slow part of its attitude is the road's slope and
  // what is left is the body on its suspension. The projection wants the
  // second, and the lever arm multiplies the first, so the two only separate
  // together. Zero leaves the tilt measured against gravity.
  // Gate on the vehicle's own horizontal acceleration in m/s^2 rather than
  // on the specific force's magnitude. Zero keeps the magnitude gate.
  void set_horizontal_tolerance(double a) {horizontal_tolerance_ = a;}
  void set_slope_tau(double tau) {slope_tau_ = tau;}
  Eigen::Matrix3d body_tilt() const;

private:
  double horizontal_tolerance_ = 0.0;
  double slope_tau_ = 0.0;
  double slope_roll_ = 0.0;
  double slope_pitch_ = 0.0;
  bool slope_started_ = false;
  double roll_ = 0.0;
  double pitch_ = 0.0;
  bool started_ = false;
  double roll_bias_ = 0.0;
  double pitch_bias_ = 0.0;
  double tau_;
  double tolerance_;
  double gravity_;
  double bias_tau_ = 0.0;
  double bias_limit_ = 0.0;
  double bias_gate_ = 0.0;
  int corrections_ = 0;
};

// Two states: how far the heading is out, and how fast it is going out.
//
// The reason this exists rather than a gain. The heading the estimator is
// handed drifts because the instrument behind it has a bias, and a bias is not
// noise: inflate its variance and pay the variance off against a measurement,
// and the next interval simply earns it back. Measured, that is exactly what
// happens -- a precision weighted share of the ground solve's own heading left
// a drifting drive at 0.172 m where believing the gyro outright gave 0.169, and
// made the two undrifted drives worse. The bias has to be a state or it cannot
// be cancelled.
//
// So: the error and its rate, the error fed by the rate, both corrected by what
// the ground says the heading should have been. The error is injected into the
// pose and reset to zero after every update, which is what makes this an error
// state filter rather than a filter on the heading itself -- the heading lives
// in the pose, where the rest of the estimator can see it.
class HeadingBiasFilter
{
public:
  HeadingBiasFilter(double bias_sigma, double walk_sigma, double noise_sigma);

  bool enabled() const {return enabled_;}
  double rate() const {return rate_;}
  const Eigen::Matrix2d & covariance() const {return covariance_;}
  // Writable so a test can deny the filter its second state and show that the
  // rate is what does the work.
  Eigen::Matrix2d & covariance() {return covariance_;}

  void predict(double dt);

  // Fold in one ground solve's opinion and return the heading offset.
  //
  // `residual` is how far that solve wanted to move the heading from where it
  // was handed. Returns what to actually move it by, which is the error state
  // after the update and before it is reset.
  double update(double residual, double sigma);

private:
  double error_ = 0.0;
  double rate_ = 0.0;
  Eigen::Matrix2d covariance_ = Eigen::Matrix2d::Zero();
  double walk_;
  double noise_;
  bool enabled_;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__ATTITUDE_HPP_
