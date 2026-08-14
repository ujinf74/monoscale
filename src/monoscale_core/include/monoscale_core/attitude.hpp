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
class AttitudeFilter
{
public:
  AttitudeFilter(double tau, double tolerance, double gravity = 9.80665);

  void update(const Eigen::Vector3d & gyro, const Eigen::Vector3d & accel, double dt);

  bool started() const {return started_;}
  double roll() const {return roll_;}
  double pitch() const {return pitch_;}
  int corrections() const {return corrections_;}

  // Roll and pitch as a rotation from the body into a level frame. Yaw is left
  // out: the planar solve owns it, and putting it here as well would rotate the
  // ground twice.
  Eigen::Matrix3d body_tilt() const;

private:
  double roll_ = 0.0;
  double pitch_ = 0.0;
  bool started_ = false;
  double tau_;
  double tolerance_;
  double gravity_;
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
