// Persistent ground features and the map they are matched against.
//
// The two-frame estimator asked one question per frame: where did these points
// go since last time? When tracking degraded the wrong answer could win, and it
// did, badly, above about 2 m/s.
//
// This keeps each feature's position in the world, averaged over every frame it
// has been seen in, and matches the current view against that instead. A
// feature that mistracks now disagrees with its own history rather than voting
// with it, which is the same reason scan-to-map beats scan-to-scan in LiDAR
// odometry.
//
// The two priors that make this stack work are untouched: the ground plane
// still supplies metric scale, and yaw still comes from the IMU.

#ifndef MONOSCALE_CORE__ANCHORS_HPP_
#define MONOSCALE_CORE__ANCHORS_HPP_

#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "monoscale_core/geometry.hpp"

namespace monoscale
{

using Identities = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;

struct AnchorSettings
{
  int max_anchors = 4000;
  int max_age_frames = 40;
  double update_gain = 0.35;
  int max_observations = 20;
  // How far each anchor's sightings scatter around it. A point on a parked car,
  // on a slope, or simply mistracked keeps landing somewhere else, and counting
  // sightings cannot tell that apart from a good one seen often.
  double initial_variance = 0.04;
  double max_variance = 0.09;
  int trial_observations = 4;
  // Off, an anchor is trusted for having been seen often, which was the
  // original rule. On, it also has to keep landing in the same place.
  bool select_by_consistency = true;
};

// World positions of ground features, refined over the frames they survive.
//
// Storage is dense arrays with a free list. The identity table is the only
// index: identities are handed out from zero upwards by the tracker, so they
// index an array directly and nothing has to be kept sorted. The Python this
// replaces also carried a dictionary of live slots, which existed to serve its
// properties rather than the solve.
class GroundAnchorMap
{
public:
  explicit GroundAnchorMap(const AnchorSettings & settings = AnchorSettings());

  // How many anchors currently hold a position.
  int size() const {return live_;}
  int64_t frame() const {return frame_;}
  int64_t discarded() const {return discarded_;}

  // Mask of which ids already have a world position.
  void anchored(const Identities & ids, Mask & out) const;

  // World positions and weights together, from one lookup. The solve wants
  // both for the same features, and asking twice means searching the index
  // twice for an answer that could not have changed in between.
  void anchor_view(const Identities & ids, Points2 & world_out, Weights & weights_out) const;

  // Fold in this frame's sightings.
  //
  // `allow_new` separates observing the map from defining it. Every frame
  // refines the anchors it can see, but a new anchor is born at whatever pose
  // the estimate held at that instant, so seeding from every frame fills the
  // map with points inheriting short-baseline error.
  //
  // `information` is how much each sighting is worth, and it is not optional in
  // spirit. A point on the ground eight metres out is pinned down far less
  // precisely than the same point at one metre -- the same angular error is
  // worth eight times as many centimetres -- and averaging the two as equals
  // leaves the anchor displaced along its own bearing. Over a frame full of
  // features that displacement does not cancel: it reads as a rotation, and the
  // ground solve asks for 2.7 mrad of heading that is not there. Weighted by
  // precision it asks for 0.24. Pass an empty vector to weight equally.
  void update(
    const Identities & ids, const Points2 & world_points, bool allow_new,
    const Weights & information);

  // Position of one anchor, for tests and diagnostics.
  std::optional<Eigen::Vector2d> position_of(int64_t identity) const;
  std::optional<int> observations_of(int64_t identity) const;
  std::optional<double> variance_of(int64_t identity) const;
  // Seen often and landing in the same place each time is what counts.
  std::optional<double> weight_of(int64_t identity) const;

private:
  int64_t slot_of(int64_t identity) const;
  void grow_table(int64_t highest);
  double weight_at(int64_t slot) const;
  void forget(int64_t slot);
  void prune();

  AnchorSettings settings_;
  Points2 position_;
  Eigen::Matrix<int64_t, Eigen::Dynamic, 1> observation_;
  Eigen::VectorXd variance_;
  Eigen::Matrix<int64_t, Eigen::Dynamic, 1> seen_;
  // How much this anchor's position is worth knowing, summed over every
  // sighting that went into it.
  Eigen::VectorXd information_;
  Eigen::Matrix<int64_t, Eigen::Dynamic, 1> identifier_;
  std::vector<int64_t> free_;
  // Slot of each identity, indexed by the identity, -1 where unknown.
  std::vector<int64_t> by_id_;
  int live_ = 0;
  int64_t frame_ = 0;
  int64_t discarded_ = 0;
};

struct AnchorAlignment
{
  Eigen::Vector2d translation;
  Mask inliers;
  // RMS residual of the inlying votes: how precise this camera's answer was.
  double spread = 0.0;
  double yaw = 0.0;
  // One standard deviation of that heading, or infinity when the fit was not
  // asked to solve for it.
  double yaw_sigma = std::numeric_limits<double>::infinity();
};

// Translation that places the current ground view onto its world anchors.
//
// With yaw taken as known each correspondence votes for one translation, so the
// solve is a robust average rather than a search.
//
// `refine_yaw` asks it to solve for heading as well. Off, the heading handed in
// stands. On, this becomes an iterated update in the sense FAST-LIO uses: the
// inliers chosen under one pose re-solve the pose, and the new pose reselects
// the inliers. Yaw and translation are not separable -- a heading error a
// hundredth of a radian wide drags the ground a centimetre sideways at every
// metre of range -- so the heading comes out of the same fit, as the 2D
// Procrustes rotation between the inlier sets.
//
// What comes back is the fit's own answer and how much that answer is worth,
// not a blend. Blending here belongs to the caller: a hard bound on the
// correction helped at 8 m/s and ruined 2.5, and weighting the share by
// precision was no better on any drive, because the gyro's error is a bias
// rather than noise. Cancelling a bias means estimating it, which is a state,
// which is the caller's business.
std::optional<AnchorAlignment> align_to_anchors(
  const Points2 & body_points, const Points2 & world_points, const Weights & weights,
  double yaw, double threshold, int min_inliers, bool refine_yaw);

struct CameraTranslation
{
  double x = 0.0;
  double y = 0.0;
  int count = 0;
  double spread = 0.0;
};

// Combine per-camera translations by how precise each one was.
//
// Weight is count over spread squared, the inverse variance of that camera's
// mean, so a camera that sees less ground but agrees with itself tightly is not
// drowned out by one that sees more and scatters. Weighting by count alone let
// the rear camera, which sits higher and sees more road, outvote the front
// twelve to one.
std::optional<CameraTranslation> fuse_by_precision(
  const std::vector<CameraTranslation> & estimates);

}  // namespace monoscale

#endif  // MONOSCALE_CORE__ANCHORS_HPP_
