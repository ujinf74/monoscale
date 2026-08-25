// The same filter with the road taken off the page.
//
// The planar filters carry x, y and a heading because that is what a vehicle on
// a flat road does. It buys a great deal: three states instead of six, no
// attitude to keep orthonormal, and a ground projection that only ever needs
// the roll and pitch a separate filter hands it.
//
// What it cannot buy is the camera-ground geometry. The height the ground
// projection scales with, and the pitch the camera is bolted at, are coupled to
// how the body is actually moving -- they change with load, with suspension
// travel, with the grade of the road. A planar filter has nowhere to put a
// state whose process model is the body's own attitude, so it can only carry
// such a state as a random walk driven by nothing. This is the container that
// makes them ordinary states rather than bolt-ons.
//
// Building the container does not make them observable. What the alignment can
// see is recorded on AnchorAlignment; the height in particular is not there,
// and has to be earned against the accelerometer. Nothing of that is here yet:
// this is the six degrees of freedom and the constraints that keep them
// honest, and the calibration states come next.

#ifndef MONOSCALE_CORE__STRAPDOWN_HPP_
#define MONOSCALE_CORE__STRAPDOWN_HPP_

#include <optional>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace monoscale
{

// Position, velocity, attitude and both biases, with the anchor cloned beside
// them the way the planar MSCKF clones its own.
//
// The error state is the one that is filtered: attitude lives in a quaternion
// and is corrected by a three-vector, so nothing has to keep four numbers on a
// sphere while a Kalman gain pulls at them.
class SpatialMsckfFilter
{
public:
  struct Settings
  {
    double acceleration_noise = 1.55;
    double gyro_noise = 0.01;
    // What the gyro is worth on roll and pitch, separately from the heading,
    // and it has to be far smaller. The heading wants to be responsive: it is
    // measured every hop by the ground and every solve by the instrument. Roll
    // and pitch are measured only by gravity, through a gate that rejects
    // whenever the vehicle is doing anything, and what they are used for is the
    // ground projection -- where one degree moves every range in the frame by
    // two and a half to five per cent.
    //
    // So an attitude that jitters is a set of ranges that jitter, and that goes
    // straight into the hop vision reports. Measured: at the heading's own
    // 0.01, the six degree of freedom filter's relative error ran 4.4, 3.3 and
    // 4.4 per cent against the planar filters' 2.0, 2.2 and 2.3, and handing
    // the projection the separate attitude filter's answer instead brought it
    // back to 2.1, 2.3 and 2.3. The separate filter is not better; it is
    // stiffer -- a sixty second trim against this one's effective two and a
    // half. This is that time constant, expressed where it belongs.
    double tilt_gyro_noise = 4.4e-4;
    double bias_walk = 0.01;
    double gyro_bias_walk = 0.0005;
    double vision_noise_m = 0.005;
    double vision_yaw_noise = 0.01;
    double vision_reference_inliers = 300.0;
    double initial_velocity_variance = 25.0;
    // Tighter than the planar filters carry, and it has to be. There, nothing
    // competed with the accelerometer bias; here gravity does. A body pitched
    // two degrees reads g*sin(2 deg) = 0.34 m/s2 along its own x, and a bias of
    // 0.34 m/s2 reads exactly the same -- at rest the two are one measurement,
    // and whichever the filter believes is less certain takes all of it. Left
    // at the planar 1.0 the bias took it every time and the attitude never
    // levelled at all.
    //
    // So this is what a part's bias is actually worth: 0.1 m/s2, a generous
    // reading of an automotive MEMS accelerometer and a third of what the
    // smallest tilt worth seeing produces. The bias is still learned, from
    // vision -- the hop against what the acceleration integrated to -- which is
    // a different measurement and not degenerate with the attitude at all.
    double initial_bias_variance = 0.01;
    double initial_gyro_bias_variance = 1.0e-4;
    // Roll and pitch start genuinely unknown -- five degrees, which is more
    // than a vehicle ever leans and is meant to be. The levelling gate widens
    // with this, so starting confident is starting unable to be corrected.
    double initial_tilt_variance = 7.6e-3;
    double initial_heading_variance = 1.0e-4;

    // Three degrees of freedom, as the planar filter has: 11.3 is the 99th
    // percentile for three. The vehicle staying on the road is not one of them
    // -- it is an assumption rather than something vision measured, and gating
    // the two together let a violated assumption throw away good vision. It
    // did, on 57% of the hops.
    double innovation_gate = 11.3;
    double outlier_inflation = 25.0;
    double reject_beyond_m = 1.5;

    double gravity = 9.80665;
    // How much acceleration the vehicle may be under and still have its
    // accelerometer believed about which way is down, in m/s2.
    //
    // Not the same number AttitudeFilter carries under a similar name. That one
    // gates the reading's magnitude, which cannot see horizontal acceleration
    // at all; this one gates the residual after the reading is rotated into the
    // world and gravity taken off, where 0.3 means 0.3 on every axis. Sharing a
    // parameter between the two would be sharing a name, not a meaning.
    double gravity_tolerance = 0.15;
    // What levelling against the accelerometer is worth, in m/s2. Loose,
    // because the vehicle is the noise here rather than the instrument: this is
    // the slow trim the attitude filter applied with a sixty second constant,
    // expressed as a variance instead of a gain.
    double gravity_noise = 2.0;
    // How far the vehicle may climb over one hop. It is a hop-length number,
    // not a drive-length one: over a few centimetres of travel a road vehicle
    // does not change height, and over a kilometre of grade it certainly does.
    // Constraining the hop leaves the grade alone.
    double height_noise_m = 0.01;

    // What the ground projection's scale might be wrong by, and how fast that
    // can change. This is Ground-VIO's camera height, in the only form this
    // stack can carry it: the projection divides every range by a measured
    // constant, so an error in the camera's height is an error in that
    // constant, and it multiplies every hop vision reports.
    //
    // Off by default, and the measurement is why.
    //
    // The state works when the error is large. Replayed with the ground scale
    // deliberately wrong by five and ten per cent, it recovers three fifths to
    // four fifths of it in either direction, and takes ATE from 1.125 m to
    // 0.896 and from 0.462 to 0.275. That is real and it is what a prior of
    // twenty per cent buys.
    //
    // It does not work on what a calibrated rig actually has left. Three drives
    // of the same rig on the same day, whose residual scale runs near one per
    // cent, taught it 0.992, 1.022 and 0.993 -- the same rig, in opposite
    // directions. It is fitting noise at that level, and it costs: ATE against
    // the state switched off, 0.345 against 0.355, 0.393 against 0.411, 0.197
    // against 0.197.
    //
    // So the default is zero, which is the state held at one, and the freedom
    // is there for a rig believed to be badly out rather than slightly. What
    // would change this is a drive long enough for the estimate to converge --
    // over 27 metres it is still moving when the recording ends.
    double initial_scale_variance = 0.0;
    double scale_walk = 1.0e-5;
  };

  struct UpdateRecord
  {
    Eigen::Vector3d innovation = Eigen::Vector3d::Zero();
    double nis = 0.0;
    bool accepted = false;
    bool dropped = false;
  };

  SpatialMsckfFilter();
  explicit SpatialMsckfFilter(const Settings & settings);

  bool settled() const {return updates_ > 0;}
  const Eigen::Vector3d & position() const {return position_;}
  const Eigen::Vector3d & velocity() const {return velocity_;}
  const Eigen::Quaterniond & attitude() const {return attitude_;}
  double yaw() const;
  double roll() const;
  double pitch() const;
  const Eigen::Vector3d & bias() const {return accel_bias_;}
  const Eigen::Vector3d & gyro_bias() const {return gyro_bias_;}
  // How much further than the truth vision is measuring the ground to have
  // moved. 1 is the calibration being right.
  double range_scale() const {return range_scale_;}
  int updates() const {return updates_;}
  int rejected() const {return rejected_;}
  int dropped() const {return dropped_;}
  int levelled() const {return levelled_;}
  const std::optional<UpdateRecord> & last_update() const {return last_update_;}

  // Roll and pitch as a rotation from the body into a level frame, yaw left
  // out. The same thing AttitudeFilter hands the ground projection, from a
  // state that shares its covariance with everything else instead of standing
  // beside the filter with none.
  Eigen::Matrix3d body_tilt() const;

  // Put the pose where the caller says it is, in the plane. Height and tilt are
  // the filter's own: the caller is placing a vehicle on a road, not in space.
  void set_pose(const Eigen::Vector2d & position, double yaw);
  // Position alone -- see PlanarMsckfFilter::set_position.
  void set_position(const Eigen::Vector2d & position);

  void open_hop();

  // Propagate on the instrument. Both vectors are in the body frame, and the
  // acceleration is specific force -- gravity included, the way the part
  // reports it.
  void predict(
    const Eigen::Vector3d & acceleration_body, const Eigen::Vector3d & rate_body, double dt);

  Eigen::Matrix3d measurement_covariance(
    int inliers, double spread, double extra_variance, double yaw_sigma) const;

  // The hop the ground solve measured, in the frame the anchor was cloned in:
  // how far, and how much it turned. Exactly what the planar filter is given.
  bool update(
    const Eigen::Vector2d & displacement_ground, double yaw_delta, int inliers,
    double extra_variance = 0.0, double spread = 0.0, double yaw_sigma = 0.0);

  // The vehicle did not climb over this hop, which is what "on the ground"
  // means and the only thing holding the height up at all.
  //
  // Its own update rather than a fourth component of the hop. It is an
  // assumption, not something vision measured, and the two do not belong behind
  // one gate: folded in together, a hop where the assumption was strained took
  // the vision measurement down with it, and 57% of them failed.
  void update_height();

  // Level the attitude against gravity, when what the accelerometer reports is
  // close enough to gravity's own magnitude to be gravity.
  //
  // Returns whether it was believed. The tolerance is the whole mechanism: an
  // accelerometer measures the vehicle as well as the planet, and the one
  // moment it can be trusted about which way is down is the moment it is not
  // measuring the vehicle. Applied on every sample with a one second constant
  // this left pitch out by 4.3 degrees against a signal of 0.04.
  bool update_gravity(const Eigen::Vector3d & acceleration_body);

  void update_zero_velocity(double sigma = 0.01, double yaw_sigma = 0.002);

  // What the instrument says the heading is, as a measurement rather than as
  // truth. PlanarMsckfFilter carries why this has to exist at all.
  void update_heading(double measured_yaw, double sigma);

  // The fused hop, in the frame the anchor was cloned in.
  Eigen::Vector2d body_translation() const;
  double hop_yaw() const;
  double speed() const {return velocity_.head<2>().norm();}

private:
  // Error state: position, velocity, attitude, both biases, the anchor's
  // position and attitude cloned beside them, and the ground projection's
  // scale. The scale is not cloned: it is a property of the rig, not of the
  // instant the anchor was taken.
  static constexpr int kSize = 22;
  using Covariance = Eigen::Matrix<double, kSize, kSize>;

  void observation(
    Eigen::Matrix<double, 3, kSize> & model, Eigen::Vector3d & predicted) const;
  void apply(
    const Eigen::Matrix<double, 3, kSize> & model, const Eigen::Vector3d & innovation,
    const Eigen::Matrix3d & noise, const Eigen::Matrix3d & inverse);
  // Fold a correction of the error state back into the nominal one.
  void correct(const Eigen::Matrix<double, kSize, 1> & error);

  Settings settings_;
  Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond attitude_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d accel_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
  double range_scale_ = 1.0;
  Eigen::Vector3d anchor_position_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond anchor_attitude_ = Eigen::Quaterniond::Identity();
  Covariance covariance_ = Covariance::Zero();

  int updates_ = 0;
  int rejected_ = 0;
  int dropped_ = 0;
  int levelled_ = 0;
  std::optional<UpdateRecord> last_update_;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__STRAPDOWN_HPP_
