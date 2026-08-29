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
#include <array>
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
  // Rank the eviction by what the solve trusts, not by when it was last seen.
  //
  // Seniority and worth are different things. An anchor seen a moment ago on
  // one poor sighting is a worse thing to keep than one seen a second ago that
  // a dozen sightings agree on, and it is the second that a revisit will
  // actually register against. `weight_at` is the number the alignment already
  // weighs by, so evicting its smallest is evicting what the solve would have
  // ignored anyway.
  bool evict_by_weight = false;
  // Rank the eviction by what the anchor is worth to the solve, which is its
  // history times the axis it actually measures.
  //
  // A translation along the heading moves a point at bearing b and range R
  // radially by cos b and tangentially by sin b, and the two are not measured
  // alike: the radial direction carries the bearing error amplified by
  // (R^2+h^2)/h while the tangential carries only R. So the information a point
  // holds about *longitudinal* motion is
  //
  //   cos^2 b h^2/(R^2+h^2)^2 + sin^2 b / R^2
  //
  // -- seven times larger at 90 degrees than at 0 for a point at 2 m, and
  // thirty-three times at 5 m. Measured on the live map: the 92-98% of anchors
  // sitting within 15 degrees of straight behind hold 10-14% of the
  // longitudinal information, while twenty anchors off to the side hold as much
  // as all of them. The error this estimator carries is longitudinal.
  bool evict_by_information = false;
  // At most this many new anchors per update. Zero leaves it unbounded.
  //
  // Capacity is doing two jobs at once today. It holds the map's memory, and by
  // being full it rate-limits admission to whatever ageing frees -- about
  // thirty a solve. Freeing the wasted slots without replacing that limit lets
  // a solve's ~1600 fresh points in at once, the map fills with anchors carrying
  // one sighting each, and the weighting the alignment depends on collapses:
  // measured, walk 1.93 -> 8.4. So the two jobs have to be separated.
  int admit_per_update = 0;
  // Sightings an anchor needs before the solve will register against it.
  //
  // `anchored` asks only whether a slot exists, so a point seen once counts as
  // known and the map answers on it. Capacity saturation has been supplying
  // this condition by accident: a full map admits almost nothing, so the few it
  // knows are old and well settled, and it answers 42% of solves. Free the
  // wasted slots and it answers **every** solve on one-sighting anchors --
  // reachable anchors 171 -> 6506, known 11% -> 86%, map frames 377 -> 898 --
  // and ATE goes 0.0701 -> 0.1804. Rare good corrections help; frequent poor
  // ones hurt. Zero keeps the old behaviour.
  int anchored_min_observations = 0;
  // Forget an anchor once its bearing off the heading passes this, in degrees.
  //
  // Past about 165 degrees a ground point is astern and receding: it will never
  // be seen again, and its information about motion along the heading has gone
  // to `cos^2 b h^2/(R^2+h^2)^2`, which at that bearing and that range is
  // nothing. Measured, 92-98% of the map sits there holding 13.6% of the
  // longitudinal information.
  //
  // This is *not* the same policy as evicting the lowest-ranked whenever a new
  // anchor wants a slot. That runs every solve against 1600 candidates and
  // replaces the population wholesale; this kills only what has actually
  // crossed the line, so in the steady state the death rate equals the birth
  // rate and the map is never flooded. Zero leaves it off.
  double forget_beyond_bearing_deg = 0.0;
  // ...and only past this range as well. Both conditions, not either.
  //
  // Bearing alone is wrong and the mistake is expensive: the rear camera looks
  // *along* 105-180 degrees and covers 92-95% of the ground there, so anchors
  // one to three metres astern are its working set, not dead weight. Killing by
  // bearing alone took curve_s10 from 0.0821 to 0.5330 while helping only the
  // 8 m/s straight. What is genuinely unreachable is astern *and* beyond the
  // solve band.
  double forget_beyond_range_m = 0.0;
  // Ground cell size and how many anchors one cell may hold. Zero is off.
  //
  // Admission control by density, not eviction. The map piles 92-98% of its
  // slots into a thin trail directly astern, which holds 13.6% of the
  // longitudinal information -- but the answer is not to delete that trail
  // afterwards. Every eviction policy tried loses, and they lose for one
  // reason: replacing the population destroys the accumulated sightings the
  // alignment weighs by, and the map's drift binding with it (walk 1.93 -> 6.9,
  // which is what this estimator reads with no working map at all). Refusing
  // the redundant anchor at birth costs nothing, because nothing that has
  // accumulated anything is touched. And a cell in metres is speed-independent,
  // where a life in solves is not: 250 solves is 123 m at 8 m/s and 31 m at 2.
  double density_cell_m = 0.0;
  int density_quota = 0;
  // The same admission control, but on cells of bearing off the heading crossed
  // with range, rather than on ground squares.
  //
  // The pile-up is angular: 92-98% of the map sits within fifteen degrees of
  // straight astern. Bearing alone is not enough to act on, because the rear
  // camera *looks* along 105-180 degrees and the anchors a metre or two back
  // are its working set -- cutting by bearing alone took curve_s10 from 0.0821
  // to 0.5330. Crossing bearing with range separates the two: the far astern
  // ring saturates and stops taking births, while the near astern ring stays
  // open for the camera that is actually using it.
  //
  // A polar cell is in the vehicle's frame, so an anchor's membership changes
  // as the vehicle moves. The counts are therefore rebuilt once per solve
  // rather than carried, which is one pass over the live anchors.
  double polar_sector_deg = 0.0;
  double polar_ring_m = 0.0;
  int polar_rings = 0;
  int polar_quota = 0;
  // How many solves an anchor must go unseen before it may be evicted.
  //
  // One -- "not updated in this solve" -- is not a test for death. About 1600
  // of the 8000 slots are updated per solve, so it offers up 6400, and most of
  // those are live features that merely fell outside the band this time. The
  // front camera's ground *approaches*, so a point too far now comes into the
  // band later; evicting it throws that away. Measured with the one-solve test,
  // walk went 1.93 -> 8.4 and ATE more than doubled: the experiment destroyed
  // live anchors rather than recycling dead ones.
  int64_t evict_unseen_solves = 1;
  // Whether an adopting camera may move the anchor it adopted. Off, only the
  // camera that founded it writes to it, and to everyone else it is a fixed
  // landmark. That is the whole point of sharing: the gap between where the
  // second camera sees the ground and where the first one put it is a
  // measurement of how far the pose has drifted since, and the registration
  // spends it on the pose. Let the adopter write and the same gap is spent
  // moving the map instead, which is drift laundered into the landmark.
  bool link_adopter_writes = false;
  // Refuse to adopt an anchor this source founded itself. The point of the link
  // is that one camera meets ground another one mapped; binding a fresh
  // sighting to a position this same camera laid down earlier is a different
  // thing wearing the same mechanism.
  //
  // Named by the founder rather than by which camera is which, because which
  // camera leads is a property of the direction of travel and not of the rig.
  // Reverse and the rear camera is the one seeing new ground -- an index would
  // then have the rule exactly backwards, and nothing in the benchmark would
  // notice, because `segment_gears` is 'drive' on every recording in it.
  bool link_cross_source_only = false;
  // Solve the bearing residuals for a rotation about base_link with no sideways
  // slide, rather than for a free rotation and translation of the lens.
  //
  // base_link is the rear axle centre, which on a vehicle that does not slip is
  // the one point with no lateral velocity. Leaving that freedom in the model
  // costs the yaw everything: a lens rotation and a lens slide look alike
  // through a patch a metre wide, and the fit trades them -- measured at
  // r = -0.82 to -0.89 between the two. Taking the slide out leaves the yaw
  // with nothing to hide behind. Conditioned over the front camera's band the
  // yaw's sigma goes 0.1515 -> 0.0339 dropping the slide, and 0.0116 dropping
  // the roll as well; the rear, being 0.82 m from the axle rather than 3.694,
  // barely moves, which is the same asymmetry the physics has.
  bool bearing_nonholonomic = false;
  // What a full density cell does with a fresh sighting. Off, it refuses the
  // birth and the cell keeps whatever it has, however old and however far
  // astern -- so the map's capacity settles on the ground it has already
  // driven past, which is where its longitudinal information is *lowest*:
  // measured, 92% of anchors sit within fifteen degrees of straight astern
  // holding 9.7% of it, and 0.2% abreast hold 57%.
  //
  // On, the cell's least informative anchor is compared against the candidate
  // and gives way if the candidate beats it by this margin. This is not the
  // global eviction that lost every time it was tried -- age, weight,
  // information, unseen count -- because it never changes how many anchors a
  // cell holds, so the accumulated sightings the alignment weighs by are not
  // churned wholesale; one anchor is exchanged for a better-placed one.
  //
  // 1.0 replaces on any improvement. Higher is stickier.
  double density_replace_margin = 0.0;
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
  // Metres ahead of the vehicle at which an anchor's information is valued.
  // An anchor is not used where it is now; it is used where it will be when the
  // solve next reaches it. Valued at the current position a feature entering
  // the band from ahead scores nothing, which is exactly the anchor about to
  // become the most informative one there is. Negative values look astern,
  // which is the control this wants.
  //
  // Metres rather than seconds so the law does not change with speed: the band
  // is geometry and so is this.
  double lookahead_m = 0.0;
  // The same thing in seconds of travel, added to the metres above and resolved
  // against the speed each frame. Which of the two is right is an open
  // question: an anchor's usable life is a fixed stretch of band, which argues
  // metres, but how long the solve takes to reach it is a time.
  double lookahead_sec = 0.0;
  // Exponent on the anchor's geometric worth in the solve weight:
  //   I(b, R) = cos^2 b / radial(R)^2 + sin^2 b / R^2,  radial = (R^2+h^2)/h
  // evaluated `lookahead_m` ahead. Astern the first term dominates and falls as
  // R^4; abeam the second holds at R^2, so close side ground keeps its worth
  // where distant ground behind loses it -- continuously, with no cutoff.
  double geometry_power = 0.0;

  // Weight an anchor by the inverse variance of what it measures, instead of by
  // a count divided by a variance.
  //
  // A bearing error is homoscedastic -- the lens is equidistant, so half a
  // pixel is the same angle everywhere -- and reaches the ground as
  // `sigma_b * radial` radially and `sigma_b * R` tangentially. Projected on
  // the heading, the Fisher information about longitudinal displacement is
  // `I(b,R)/sigma_b^2` with I as above, so this frame's measurement of the
  // anchor has variance `sigma_b^2 / I`. The anchor's own stored position
  // carries a second, independent variance. They add:
  //
  //   w = 1 / ( sigma_b^2 / I(b_T,R_T)  +  variance_(slot) + drift * travelled )
  //
  // Everything the old weight tried to say is in there and nothing is said
  // twice. The observation count is gone because `variance_` already falls with
  // observations -- counting both was counting maturity twice, and the count
  // was capped at `max_observations` anyway, which left it a 1.25x range over
  // a population spanning 400:1. The drift is a variance growing linearly with
  // the path, which is what a random walk does, rather than a hyperbolic
  // discount that only reached 8x at 139 m. And the geometry is present at its
  // own strength rather than behind an exponent that existed to trade against
  // those two faults.
  bool weight_by_variance = false;
  // Variance of a tracked bearing, rad^2. Measured, not chosen: half a pixel at
  // the 640-wide processing width is 1.9e-3 rad against fx_eff 262.95 px/rad.
  double bearing_variance = 3.6e-6;

  // Weight an anchor by whether its re-observations agree with where it said it
  // would be, and by whether that agreement is getting better or worse.
  //
  // The bearing model above says where an anchor *ought* to be informative.
  // This says nothing about geometry and measures the same thing instead: an
  // anchor that is far, or on a bad patch, or not really a landmark, scatters
  // when it is seen again, and that scatter is already what `variance_` holds.
  // Modelling the noise is replaced by measuring it.
  //
  // The second EWMA is the part the level alone cannot give. Two anchors at the
  // same scatter are not worth the same if one is settling and the other coming
  // apart, and which one it is decides whether re-observing is buying anything.
  //
  //   sigma2_pred = fast * (fast / slow)^trend_power
  //
  // The ratio is below one while the residuals shrink, so the anchor is
  // credited with the improvement it is on course for rather than the one it
  // has banked; above one it is charged for the decline. `trend_power` 0 leaves
  // the level alone, 1 extrapolates one step of it.
  bool weight_by_trend = false;
  // Gain of the slow average. Smaller is a longer memory; it must be slower
  // than the fast one or the ratio carries no information.
  double trend_gain = 0.05;
  double trend_power = 0.0;
  // An anchor whose predicted scatter passes this is not a landmark and goes,
  // whatever its age and whatever room there is. In metres squared, so it is
  // the same quantity `max_variance` already gates on.
  double trend_evict_variance = 0.0;

  // Identities at or above this go to the front of the admission queue.
  //
  // A full map frees only the handful of slots ageing gives back, and the queue
  // that competes for them carries ~1500 corner candidates a frame. A road-grid
  // point is not refused, it is outvoted: measured, every source founds anchors
  // at the same ~2% rate, so a lattice offering 117 candidates over a drive
  // founds none while one offering 1774 founds 38. Feeding more of them fixes
  // admission and breaks the solve, because a lattice is one homography sampled
  // many times. This is the other way round -- offer few and let those few in.
  //
  // 0 leaves the queue in whatever order the frame built it.
  int64_t priority_identity_floor = 0;

  // Admit the most informative candidates rather than whichever the frame
  // happened to list first.
  //
  // A saturated map frees a handful of slots per update and ~1500 corner
  // candidates queue for them, so which 2% get in is decided by array order --
  // which is to say, not decided. Sorting by `longitudinal_information_at`
  // spends the scarce slots on the ground that will carry the solve, evaluated
  // at the lookahead point like every other use of that law.
  bool admit_by_information = false;
  // Admit the clearest candidates -- the ones the tracker's corner response
  // says are really landmarks -- rather than the first ones the frame listed.
  // Ranking by position was measured worse than not ranking at all; this ranks
  // by whether the patch is distinct, which the estimator has never had.
  bool admit_by_clarity = false;

  // How many consecutive sightings a candidate must survive before it may take
  // a slot. 1 founds on first sight, which is what this map has always done.
  //
  // Standard KLT practice does not do that: VINS-Mono only puts a feature into
  // the optimisation once it has been observed several times, and an MSCKF
  // triangulates a track only when it ends with enough observations and
  // baseline. Neither spends state on a feature that has proven nothing. Ours
  // does, and on the fast straight 46% of the anchors it founds are never seen
  // again -- so nearly half the map's capacity is committed to features that
  // did not survive one frame, and committed in the first ten metres, after
  // which nothing else can get in.
  int found_after_observations = 1;
  bool link_measure_only = false;
  // Compute the rebuild but do not apply it.
  bool rebuild_measure_only = false;
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
  // Where the live anchors are, relative to a pose. Returns how many sit within
  // `band` of it, and the spread of all of them along and across its heading.
  // The map holds thousands of slots; this says how many of them the next solve
  // could possibly use.
  // Every live anchor as (range, bearing off the heading, weight, sightings,
  // identity), for asking where the map's capacity actually sits and what kind
  // of observation put it there.
  void polar(
    double x, double y, double yaw,
    std::vector<std::array<double, 7>> & out) const;
  void extent(
    double x, double y, double yaw, double band, int & within, double & along,
    double & across) const;
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
  // `weight_yaw` is the heading an anchor's worth is judged from, and it should
  // be a filtered one. Every anchor's bearing turns with the vehicle at once,
  // so an unfiltered heading collapses the whole map's weight together on a
  // swerve the vehicle may be about to come back out of. The bearing is a claim
  // about where anchors will be useful; it should not follow a transient.
  void set_frame_pose(double path, double yaw, double weight_yaw)
  {
    path_ = path;
    yaw_ = yaw;
    weight_yaw_ = weight_yaw;
  }
  void set_frame_pose(double path, double yaw) {set_frame_pose(path, yaw, yaw);}
  // Where the vehicle is, so an anchor's worth can be asked of its geometry
  // rather than only of its history.
  // Where an anchor's worth is valued: `lookahead_m` plus `lookahead_sec` of
  // travel at the speed handed in. Resolved here so the map does not have to
  // know how fast the vehicle is going.
  // The vehicle does not travel along its current heading, it travels along an
  // arc, so projecting the evaluation point down the tangent puts it off the
  // road on anything that turns. `turn_rate` is the yaw rate in rad/s and must
  // be a *filtered* one: the raw per-hop rate carries the solve's own noise,
  // and squaring that into a position two seconds out is how a prediction
  // becomes worse than no prediction.
  void set_lookahead(double speed, double turn_rate)
  {
    const double distance =
      settings_.lookahead_m + settings_.lookahead_sec * std::max(speed, 0.0);
    if (!(distance > 0.0)) {
      look_f_ = 0.0;
      look_l_ = 0.0;
      return;
    }
    // How long the vehicle takes to cover it, and how far it turns in that time.
    const double seconds = speed > 1e-3 ? distance / speed : 0.0;
    const double swept = turn_rate * seconds;
    if (std::abs(swept) < 1e-6) {
      look_f_ = distance;
      look_l_ = 0.0;
      return;
    }
    const double radius = distance / swept;
    look_f_ = radius * std::sin(swept);
    look_l_ = radius * (1.0 - std::cos(swept));
  }
  void set_frame_position(double x, double y, double height)
  {
    at_x_ = x; at_y_ = y; lens_height_ = height;
    if (settings_.polar_quota > 0) {rebuild_polar_counts();}
  }

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
    const Weights & information, const Weights & clarity = Weights());

  // The same, remembering where each sighting was taken from. An anchor is a
  // running average, and folding a sighting in destroys which pose it came
  // from -- which is why a corrected trajectory cannot be pushed back into the
  // map. Keeping `(pose index, body point)` per sighting is the smallest thing
  // that makes the map re-derivable.
  // `clarity` is how distinct each candidate's patch is, from the tracker's
  // corner response. It orders admission when `admit_by_clarity` is set and is
  // ignored otherwise; empty means the tracker is not publishing it.
  void update(
    int source, const Identities & ids, const Points2 & world_points, bool allow_new,
    const Weights & information, const Points2 & body_points, int32_t pose_index,
    const Weights & clarity = Weights());

  // Recompute every anchor from the poses its sightings were taken at. `poses`
  // is indexed by the same `pose_index` handed to `update`.
  void rebuild(const std::vector<std::array<double, 3>> & poses);

  // How many sightings are being remembered, for pricing the storage.
  int64_t remembered() const {return remembered_;}
  // How far apart in pose index the remembered sightings of a live anchor
  // are, averaged. If this is a handful of frames the map holds no long
  // baseline and nothing in it can pin a correction.
  double sighting_span() const;

  // A constraint between two poses: both saw the same anchor, so their
  // separation is fixed by where each of them put it in its own body frame.
  // This is the only measurement on this rig that ties two parts of the
  // trajectory together -- the rear camera drives over ground the front camera
  // mapped 4.5 m earlier -- and it is what a pose graph needs and dead
  // reckoning cannot supply.
  struct Revisit
  {
    int32_t from;
    int32_t to;
    double bx_from;
    double by_from;
    double bx_to;
    double by_to;
    double weight;
  };
  std::vector<Revisit> revisits() const;
  // Mean distance a rebuild moves an anchor. With the poses unchanged this is
  // how far the remembered sightings fail to reproduce the running average --
  // the test of whether the storage is sufficient at all.
  double rebuild_shift() const
  {return rebuild_slots_ > 0 ? rebuild_shift_ / rebuild_slots_ : 0.0;}
  void clear_rebuild_shift() {rebuild_shift_ = 0.0; rebuild_slots_ = 0;}
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
  double longitudinal_information(int64_t slot) const;
  double predicted_variance(int64_t slot) const;
  // Position scatter of a live anchor, m^2. What `variance_` holds.
  double scatter_at(int64_t slot) const;
  // Slot is live, past trial, and still agreeing with itself: what the solve
  // will actually register against.
  bool usable_at(int64_t slot) const;
  // The same, for a position that has no slot yet -- what a candidate birth
  // would be worth if it were admitted.
  double longitudinal_information_at(double x, double y) const;
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
  // The slow companion to `variance_`, for the trend. Same units.
  Eigen::VectorXd variance_slow_;
  // Candidates that have not earned a slot yet: identity -> (run, last frame).
  // The run resets when a candidate misses a frame, so what it counts is a
  // track that survived, not a track that reappeared.
  std::vector<std::unordered_map<int64_t, std::pair<int32_t, int64_t>>> pending_;
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
  // Per slot, a ring of the most recent sightings: which pose they were taken
  // from and where the point sat in that pose's body frame.
  static constexpr int kRemember = 16;
  Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> sight_pose_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sight_body_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sight_weight_;
  // Which source took each sighting. A revisit is only a loop closure if the
  // two ends came from different cameras; one camera's own sightings restate
  // the odometry that already links those poses.
  Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> sight_source_;
  // Per slot and per source, the single strongest sighting that source ever
  // took of it. A loop closure needs one observation from each camera, not the
  // recent ones: the ring holds sixteen frames and one camera fills it before
  // the other arrives, so the ring alone can never hold both ends.
  Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> best_pose_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> best_body_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> best_weight_;
  Eigen::Matrix<int32_t, Eigen::Dynamic, 1> sight_count_;
  int64_t remembered_ = 0;
  double rebuild_shift_ = 0.0;
  int64_t rebuild_slots_ = 0;
  int sources_ = 1;
  // The evaluation point relative to the vehicle, forward and left.
  double look_f_ = 0.0;
  double look_l_ = 0.0;
  double weight_yaw_ = 0.0;
  double at_x_ = 0.0;
  double at_y_ = 0.0;
  double lens_height_ = 0.0;
  // Set once the map has reached capacity, and never cleared.
  bool saturated_ = false;
  // Coarse spatial index over live anchors, one bucket per link-radius cell.
  std::unordered_map<int64_t, std::vector<int64_t>> grid_;
  // Live anchors per density cell, kept alongside births and deaths.
  std::unordered_map<int64_t, int> density_;
  // Which slots each density cell holds, so a full cell can be asked what its
  // least valuable anchor is rather than only how many it has.
  std::unordered_map<int64_t, std::vector<int64_t>> density_slots_;
  int64_t density_cell_of(double x, double y) const;
  int polar_cell_of(double x, double y) const;
  void rebuild_polar_counts();
  std::vector<int> polar_;
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
  // The same radial residual split into the two things that can cause it.
  //
  // A camera believed to sit at the wrong height puts every ground point wrong
  // by `R dh/h` -- linear in range, and zero underneath the lens. A camera
  // believed to be at the wrong pitch puts it wrong by `(R^2+h^2)/h dtheta` --
  // quadratic, and `h dtheta` underneath the lens rather than zero. The single
  // linear fit above cannot tell them apart and reports both as height.
  //
  // Diagnostic only. The two bases are 98-99% collinear over any range this rig
  // sees (condition number 25-50), so this is worth something only because the
  // signal is large: over 0.5-5.8 m a milliradian of pitch moves the far
  // anchors 37.7 mm against the near ones, where over the photometric band's
  // 0.34-1.14 m it moves them 1.34 mm.
  double radial_height = 0.0;
  double radial_pitch = 0.0;

  // The same residuals read on the sphere instead of on the ground.
  //
  // Two reasons it is the better domain. A ground residual is the bearing
  // residual multiplied by (R^2 + h^2)/h, which over this band runs from 4 mm
  // at a metre to 73 mm at 5.8, so an unweighted least squares on the ground
  // is weighted by the wrong thing and the far anchors carry the fit. On the
  // sphere the noise is the sensor's own and does not depend on range at all.
  //
  // And the signatures separate: an attitude error turns every bearing by the
  // same `-w x b` whatever its range, while a translation error moves it by
  // `-(I - b b^T) t / rho`, which falls away with distance. No scale error can
  // imitate a rigid rotation of the field -- scaling the anchors about the lens
  // leaves every bearing exactly where it was -- so the angle comes out
  // independent of whatever the map's own scale is doing. The height does not,
  // which is why only the angles are reported here.
  double bearing_roll = 0.0;
  double bearing_pitch = 0.0;
  double bearing_yaw = 0.0;
  // How many bearings the fit stood on, 0 where it did not run.
  int bearing_terms = 0;
  // The two in-plane translations the same fit solved for. Carried out only to
  // ask whether they are taking the yaw with them: a yaw about the lens and a
  // sideways slide look alike through a patch held four metres off the rotation
  // centre, and if that degeneracy is what costs the yaw its precision, these
  // two move against it solve by solve.
  double bearing_tx = 0.0;
  double bearing_ty = 0.0;
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
  double lens_height = 0.0,
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
  const Weights & residual_scale = Weights(),
  // Solve the bearing residuals for a rotation about base_link with no sideways
  // slide. See AnchorSettings::bearing_nonholonomic for why.
  bool bearing_nonholonomic = false);

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
