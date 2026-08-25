// One estimator over everything the cameras see.
//
// The stack this replaces has two paths: a feature whose ray meets the ground
// plane inside the band gets a metric position from a single sighting, and
// everything else is discarded. That split is not in the geometry. Both are
// landmarks; the plane is a very strong prior on the depth of some of them,
// and where the code has a branch a covariance belongs.
//
// The branches that go with it are the ones this estimator has been paying
// for: the band gate that admits or discards, the age cutoff that keeps an
// anchor at full value until the instant it is thrown away, and the either/or
// between the map path and the two-frame fallback -- which was measured to
// fail as a pooled vote precisely because an anchor's error is correlated with
// the pose that wrote it, and there was nowhere to say so. The full joint
// covariance is where that is said.
//
// Accuracy first throughout. The state is dense and every correlation is kept;
// the update is batched over the frame and relinearised, because a sequential
// update makes the answer depend on the order the features arrived; and a
// landmark waits until its geometry can carry a position rather than being
// initialised badly and corrected later. Trimming this for cost comes after it
// is right, not before.
#ifndef MONOSCALE_CORE__LANDMARK_HPP_
#define MONOSCALE_CORE__LANDMARK_HPP_

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "monoscale_core/geometry.hpp"

namespace monoscale
{

struct LandmarkSettings
{
  // What one bearing is worth, in radians. Measured rather than assumed: the
  // induced-homography residual on features that actually moved runs 0.11 px
  // at 2 m/s and 1.26 at 8, and this rig carries 263 px/rad at the width the
  // tracker runs, so half a pixel is 1.9e-3.
  double bearing_noise_rad = 0.0019;
  // How far one step can differ from carrying the last hop forward.
  double hop_process_noise_m = 0.08;
  // The instrument's heading. On this rig the gyro yaw correlates 0.985-0.9997
  // with truth at scale 0.97-1.004, so this is tight on purpose.
  double yaw_noise_rad = 0.002;

  // The whole of the ground / not-ground distinction, expressed as a prior on
  // range rather than as a branch.
  //
  // A ground point's range comes from the plane and one bearing:
  // sigma_range = (R^2 + h^2)/h * sigma_bearing, which at 2 m and 0.89 m of
  // camera height is 5.38 * sigma. Computed per sighting rather than fixed.
  bool range_from_plane = true;
  // Across the ray both start at the bearing noise times the range, which is
  // the same for either kind. Nothing else differs.

  // A landmark with no plane under it waits here until its own bearings have
  // spread far enough to carry a range. Initialising it early and letting the
  // filter fix it is what makes an XYZ landmark inconsistent.
  double initialise_parallax_rad = 0.035;
  int initialise_min_views = 4;

  // Retirement is by what a landmark still contributes, not by a clock and not
  // by a capacity chosen for arithmetic.
  int retire_unseen_frames = 60;
  int max_landmarks = 4000;

  // A sighting further than this from where the landmark says it should be is
  // not that landmark. Chi-square on two degrees of freedom.
  double reject_chi_square = 9.0;
  // How many times the batch is relinearised. One is an EKF; more is an
  // iterated EKF, which costs arithmetic and buys consistency.
  int iterations = 3;
};

// A planar pose and the structure it is measured against, in one covariance.
class LandmarkFilter
{
public:
  explicit LandmarkFilter(const LandmarkSettings & settings);

  struct Sighting
  {
    int64_t identity = 0;
    // Unit ray in the body frame at the time of the sighting, and where the
    // camera sits in that frame.
    Eigen::Vector3d bearing = Eigen::Vector3d::UnitX();
    Eigen::Vector3d mount = Eigen::Vector3d::Zero();
    // Range along the ray to the ground plane, where the ray meets it inside
    // the band. Absent means no plane prior -- the landmark waits for parallax.
    std::optional<double> ground_range;
  };

  // Carry the pose forward and open the step's uncertainty. The turn is the
  // instrument's; the translation is whatever the last accepted step was.
  void predict(const Eigen::Vector2d & hop_body, double yaw_delta);

  // Fold in one frame's sightings, all of them at once, relinearising between
  // iterations. Returns how many were used.
  int observe(const std::vector<Sighting> & sightings, int64_t frame);

  void retire(int64_t frame);

  const Eigen::Vector3d & pose() const {return pose_;}
  Eigen::Matrix3d pose_covariance() const {return covariance_.topLeftCorner<3, 3>();}
  size_t landmarks() const {return order_.size();}
  size_t waiting() const {return pending_.size();}
  int64_t initialised() const {return initialised_;}
  int64_t rejected() const {return rejected_;}
  int64_t updated() const {return updated_;}

private:
  struct Entry
  {
    Eigen::Index at = 0;      // row of this landmark's x in the state
    int64_t seen = 0;
    int64_t sightings = 0;
  };
  // A landmark that cannot yet carry a position, holding its bearings until it
  // can. World frame, so the pose that took each one is already folded in.
  struct Pending
  {
    std::vector<Eigen::Vector3d> centres;
    std::vector<Eigen::Vector3d> bearings;
    int64_t seen = 0;
  };

  Eigen::Index size() const {return 3 + 3 * static_cast<Eigen::Index>(order_.size());}
  // World position and camera centre of a sighting under the current pose.
  void world_of(
    const Sighting & sighting, Eigen::Vector3d & centre, Eigen::Vector3d & ray) const;
  // `by_pose` is the derivative of the new position on the pose it was built
  // from, which is what ties it to everything already held.
  bool admit(int64_t identity, const Eigen::Vector3d & position,
    const Eigen::Matrix3d & covariance, const Eigen::Matrix3d & by_pose);
  void drop(int64_t identity);

  LandmarkSettings settings_;
  Eigen::Vector3d pose_ = Eigen::Vector3d::Zero();   // x, y, yaw
  Eigen::VectorXd landmarks_;                        // 3 per entry, world frame
  Eigen::MatrixXd covariance_;
  std::unordered_map<int64_t, Entry> index_;
  std::unordered_map<int64_t, Pending> pending_;
  std::vector<int64_t> order_;
  int64_t initialised_ = 0;
  int64_t rejected_ = 0;
  int64_t updated_ = 0;
  int64_t seen_frame_ = 0;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__LANDMARK_HPP_
