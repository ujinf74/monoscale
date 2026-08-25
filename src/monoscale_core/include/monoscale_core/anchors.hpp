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
#include <unordered_map>
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
  // Floor under the precision-weighted gain, which is what bounds the anchor's
  // effective averaging window. Without it `information_` accumulates without
  // limit, the gain decays to zero, and an anchor stops following its own
  // sightings: whatever pose error it was born with is frozen in, and a later,
  // better estimate cannot wash it out. 0 keeps the unbounded mean.
  double min_update_gain = 0.0;
  // How close a sighting must land to an existing anchor for a camera that has
  // not seen it before to adopt it rather than found its own. This is what
  // makes the map shared in fact and not just in storage: the front camera
  // drives over ground the rear one sees 4.5 m later, and without this the
  // second sighting starts a second anchor carrying whatever pose error the
  // estimate had at that moment. 0 keeps each camera to its own anchors.
  double link_radius_m = 0.0;
  // Frames an anchor must go untouched by a source before that source may
  // rebind it to a different identity.
  int64_t link_rebind_grace_frames = 1;
  // When full, evict whatever was seen longest ago rather than whatever is
  // least well observed.
  bool evict_by_age = false;
  // Evict the longest-unseen anchors to seat features arriving now.
  bool evict_for_new = false;
  // Whether an adopting camera may move the anchor it adopted. Off, only the
  // camera that founded it writes to it, and to everyone else it is a fixed
  // landmark. That is the whole point of sharing: the gap between where the
  // second camera sees the ground and where the first one put it is a
  // measurement of how far the pose has drifted since, and the registration
  // spends it on the pose. Let the adopter write and the same gap is spent
  // moving the map instead, which is drift laundered into the landmark.
  bool link_adopter_writes = false;
  // Look for the crossing and record it, but do not bind it. Adoption itself
  // was measured and lost; the measurement it makes possible is separate.
  // Weigh an anchor in the registration by what is known about its position
  // rather than by how many times it was seen.
  //
  // `weight_at` has always returned the observation count. The map computes a
  // proper information for every sighting -- (h / (R^2 + h^2))^2, the inverse
  // variance the geometry implies -- averages the anchor's position with it,
  // and then throws it away at the one place it decides the pose. A near
  // anchor seen three times weighs less than a far one seen ten, which is
  // backwards: the registration residual grows sevenfold from the 1-2 m ring
  // to the 4-5 m one. `min_update_gain` already bounds how far the accumulated
  // information can run, so using it here is bounded too.
  bool weight_by_information = false;
  bool link_measure_only = false;
  // How fast an anchor's worth decays with the distance driven since it was
  // founded. Zero keeps the hard age cutoff on its own.
  double drift_variance_per_m = 0.0;
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
  // `sources` is how many cameras write into this map. One map shared by all
  // of them is the point: the front camera drives over ground the rear one
  // sees 4.5 m later, and with a map each that second sighting founded a
  // second anchor instead of refining the first.
  explicit GroundAnchorMap(
    const AnchorSettings & settings = AnchorSettings(), int sources = 1);

  // How many anchors currently hold a position.
  int size() const {return live_;}
  int64_t frame() const {return frame_;}
  int64_t discarded() const {return discarded_;}
  // How many times a camera adopted an anchor another camera had founded.
  int64_t adopted() const {return adopted_;}
  // Mean gap between where a camera saw the ground and where the anchor it
  // adopted already sat, and that same gap divided by the sighting's range.
  // A scale error grows with range so the second stays flat; a translation
  // error does not so the first stays flat. That is how they are told apart.
  double link_gap_m() const {return adopted_ > 0 ? link_gap_ / adopted_ : 0.0;}
  double link_gap_per_m() const {return adopted_ > 0 ? link_ratio_ / adopted_ : 0.0;}
  double link_range_m() const {return adopted_ > 0 ? link_range_ / adopted_ : 0.0;}
  // The crossing read as a length. The rear camera meets ground the front one
  // anchored some metres back, and the along-track part of their disagreement
  // is the pose error accumulated over exactly that stretch -- a scale
  // measurement over a four metre baseline instead of a 1.5 m sighting.
  int64_t crossings() const {return crossings_;}
  double crossing_along_m() const {return crossings_ > 0 ? along_ / crossings_ : 0.0;}
  double crossing_travel_m() const {return crossings_ > 0 ? gap_travel_ / crossings_ : 0.0;}
  // Least squares of the along-track gap on how far the vehicle came between
  // the two sightings. The **slope** is the scale error; the **intercept** is
  // the standing difference between the two cameras' projections, which does
  // not grow with distance and would otherwise be read as scale.
  double crossing_slope() const
  {
    const double n = static_cast<double>(crossings_);
    const double det = n * travel_squared_ - gap_travel_ * gap_travel_;
    return std::abs(det) > 1e-9 ? (n * cross_ - gap_travel_ * along_) / det : 0.0;
  }
  double crossing_intercept() const
  {
    const double n = static_cast<double>(crossings_);
    const double det = n * travel_squared_ - gap_travel_ * gap_travel_;
    return std::abs(det) > 1e-9
           ? (travel_squared_ * along_ - gap_travel_ * cross_) / det : 0.0;
  }
  // Where the vehicle is, so a crossing can be resolved along its heading and
  // dated by how far it has come.
  void set_frame_pose(double path, double yaw) {path_ = path; yaw_ = yaw;}

  // Mask of which ids already have a world position.
  void anchored(int source, const Identities & ids, Mask & out) const;
  // Single-source shorthand, which is what a one-camera map is.
  void anchored(const Identities & ids, Mask & out) const {anchored(0, ids, out);}

  // World positions and weights together, from one lookup. The solve wants
  // both for the same features, and asking twice means searching the index
  // twice for an answer that could not have changed in between.
  void anchor_view(
    int source, const Identities & ids, Points2 & world_out, Weights & weights_out) const;
  void anchor_view(const Identities & ids, Points2 & world_out, Weights & weights_out) const
  {
    anchor_view(0, ids, world_out, weights_out);
  }

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
  // Move the map on by one frame. Ageing is per frame, not per camera, so
  // this is called once even though every camera then writes into the map.
  void advance() {++frame_;}

  void update(
    int source, const Identities & ids, const Points2 & world_points, bool allow_new,
    const Weights & information);
  void update(
    const Identities & ids, const Points2 & world_points, bool allow_new,
    const Weights & information)
  {
    advance();
    update(0, ids, world_points, allow_new, information);
  }

  // Position of one anchor, for tests and diagnostics.
  std::optional<Eigen::Vector2d> position_of(int64_t identity) const;
  std::optional<int> observations_of(int64_t identity) const;
  std::optional<double> variance_of(int64_t identity) const;
  // Seen often and landing in the same place each time is what counts.
  std::optional<double> weight_of(int64_t identity) const;

private:
  int64_t slot_of(int source, int64_t identity) const;
  int64_t cell_of(double x, double y) const;
  void grid_insert(int64_t slot);
  void grid_erase(int64_t slot);
  // Nearest anchor within link_radius_m that this source has not bound yet.
  int64_t adoptable(int source, double x, double y) const;
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
  // Slot of each identity, per source. Track identities restart per camera, so
  // one table each.
  //
  // Sparse, not a vector indexed by the identity. Identities only ever go up --
  // 467,085 of them in 601 frames at road speed -- so a dense table grows with
  // every feature the drive has ever detected rather than with the ones it is
  // still holding, and an hour of driving would ask for hundreds of megabytes
  // of mostly -1. The road alignment's identities start at 1 << 40 and made
  // that fatal rather than merely wasteful.
  std::vector<std::unordered_map<int64_t, int64_t>> by_id_;
  // Which identity each source has bound to a slot, -1 where none. A slot may
  // carry one per source: that is what makes it a shared landmark rather than
  // two anchors in the same place.
  Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> owner_;
  // Which source founded each slot; only it may move the anchor.
  Eigen::Matrix<int64_t, Eigen::Dynamic, 1> founder_;
  int sources_ = 1;
  // Coarse spatial index over live anchors, one bucket per link-radius cell.
  std::unordered_map<int64_t, std::vector<int64_t>> grid_;
  int live_ = 0;
  int64_t frame_ = 0;
  int64_t discarded_ = 0;
  int64_t adopted_ = 0;
  double link_gap_ = 0.0;
  double link_ratio_ = 0.0;
  double link_range_ = 0.0;
  double along_ = 0.0;
  double gap_travel_ = 0.0;
  double travel_squared_ = 0.0;
  double cross_ = 0.0;
  int64_t crossings_ = 0;
  double path_ = 0.0;
  double yaw_ = 0.0;
  // Path length when each anchor was founded, so a crossing knows how far the
  // vehicle came between the two sightings.
  Eigen::VectorXd founded_path_;
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

  // How the inlying residuals lean along their own bearing from the camera,
  // regressed against range and normalised by `radial_reference`. `spread` is
  // these same residuals squared into one number, which is what the fit needs;
  // this is the part that squaring destroys.
  //
  // It measures the mounting pitch, and only that. Measured by replaying one
  // drive with the mounting deliberately turned, it runs -0.0035, -0.0014,
  // -0.0008, +0.0003, +0.0017, +0.0025, +0.0043 over pitch errors of -1 to +1
  // degree: monotone, and about 0.004 per degree.
  //
  // The camera's height does not appear here, and cannot. A height believed to
  // be h(1+d) puts every ground point at r(1+d), and the anchors are built from
  // that same projection, so the whole map is scaled with it and a rigid fit is
  // satisfied. The trajectory is wrong -- a 2% error took ATE from 0.111 m to
  // 0.336 -- and this residual never moves. Height is observable against the
  // accelerometer or not at all.
  //
  // What lets pitch through where height cannot is that its error depends on
  // range: dr = -e(h + r*r/h) against d*r. An anchor is averaged over the
  // frames its feature survived and the feature crosses the field, so each
  // sighting carried a different range error into that average and they do not
  // cancel. A quadratic term was fitted for as well, orthogonalised against
  // this one; over the range the ground band actually spans there is no support
  // left for it and it came back noise, so the linear term carries the whole
  // signal.
  double radial_linear = 0.0;
  double radial_reference = 0.0;
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
// `origin` is where the camera sits in the frame `body_points` are given in.
// The range that separates a height error from a tilt one is measured from the
// lens, not from the axle, and on this vehicle those are three metres apart.
// Leaving it at zero costs nothing but `radial_linear`.
//
// `radial_min_range` drops the nearest ground from that regression and from
// nothing else -- those points still vote for the pose. They have to go because
// a pixel there is worth five millimetres and optical flow loses them first, and
// weighting by precision hands them the fit. Measured on the front camera, whose
// ground band starts at 0.6 m: the lean over mounting errors of -0.5, 0 and +0.5
// degrees ran -0.0003, +0.0007, -0.0005 with them in, which says nothing, and
// -0.0040, -0.0008, +0.0010 with the first 1.5 m left out.
std::optional<AnchorAlignment> align_to_anchors(
  const Points2 & body_points, const Points2 & world_points, const Weights & weights,
  double yaw, double threshold, int min_inliers, bool refine_yaw,
  const Eigen::Vector2d & origin = Eigen::Vector2d::Zero(),
  double radial_min_range = 0.0,
  // Width of the Gaussian that replaces the hard inlier gate, in metres.
  // 0 keeps the gate. See the note in the implementation.
  double softness = 0.0,
  // Where to start the search. Without one the fit begins at the median vote,
  // which at low speed sits near zero simply because most of the hop is
  // smaller than the residuals around it -- and an iteration started there
  // settles into "did not move". Seeded with the previous hop, the same data
  // settles on the answer next to it instead.
  const Eigen::Vector2d * translation_prior = nullptr,
  // How many starting points the mode search tries. 1 keeps the median only.
  int restarts = 1,
  // Refuse the solve unless the winning mode beats the best rival mode by
  // this factor. 0 accepts whatever wins.
  double ambiguity = 0.0,
  // Reject modes further than this from the inertial expectation. 0 disables.
  const Eigen::Vector2d * inertial_hop = nullptr,
  double inertial_gate = 0.0,
  // Per point, how much wider than the configured gate and softness this
  // residual is allowed to be. Empty means one for everything. A bearing
  // measurement's metric error scales with its range, so a landmark twenty
  // metres out needs ten times the tolerance of one at two to be judged on the
  // same angle.
  const Weights & residual_scale = Weights());

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
