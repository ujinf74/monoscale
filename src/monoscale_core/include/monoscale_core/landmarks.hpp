// Features that are not on the road, carried by an inverse depth.
//
// Everything else in this stack measures the hop from the ground plane, and the
// plane is what supplies metric scale. That is also its limit: a point only
// counts if it lands on the road between the band's two edges, which on a
// forward monocular rig is about a tenth of the frame, and on a smooth road at
// speed it is a tenth of the frame with nothing on it. Measured on KITTI,
// tracks living sixteen frames or more supply 334 sightings a frame above the
// horizon line against 101 on the road, and the longest-lived road track
// survives 81 frames against 1203 for one above it.
//
// The plane cannot place those. A bearing crossed with the road plane puts a
// feature standing z above it H/(H-z) times too far out, and the crossing walks
// back towards the camera by z/(H-z) of every metre driven -- a 0.17 m kerb
// under a 1.65 m camera drifts 0.09 m a hop. Above the camera's own height the
// ray never meets the plane at all. What places them is a depth, and the depth
// has to be triangulated against motion that is already metric.
//
// Placing them with the plane's own slip path was tried and cannot reach far
// enough: a feature at the camera's height has a ray that never meets the
// plane, so `obstacle_height_margin` caps that route at 0.7 H = 1.16 m, and
// everything worth having -- the building corners, the poles, the signs -- is
// above it.
//
// So this is deliberately not a second source of scale. It is a memory of the
// scale the plane supplied: while the road answers, each landmark's depth is
// refined against the metric hop; when the road stops answering, the landmarks
// whose depth has converged still know how far away they are, and the hop
// follows from their bearings.

#ifndef MONOSCALE_CORE__LANDMARKS_HPP_
#define MONOSCALE_CORE__LANDMARKS_HPP_

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

namespace monoscale
{

struct LandmarkSettings
{
  // Where a landmark's depth starts before any parallax has been seen, and how
  // wrong that is allowed to be. Inverse depth rather than depth because the
  // uncertainty of a far feature is symmetric in 1/d and wildly skewed in d --
  // a bearing that could be 40 m or 400 m away is one interval in inverse depth
  // and two orders of magnitude in depth.
  double initial_depth_m = 20.0;
  double initial_inverse_sigma = 0.045;
  // One pixel over the focal length, as an angle. The measurement noise on a
  // bearing.
  double bearing_sigma_rad = 0.002;
  // How well the depth has to be known before the landmark may vote on a hop,
  // as a fraction of the inverse depth itself.
  double converged_fraction = 0.25;
  // Depths outside this are not a landmark: the near end is the bonnet and the
  // far end is a bearing with no parallax in it.
  double min_depth_m = 3.0;
  double max_depth_m = 120.0;
  // A landmark that has gone unseen for this many frames is dropped.
  int patience = 5;
  int max_landmarks = 3000;
  // Fewest converged landmarks a hop may be solved from.
  int min_votes = 12;
  // How far the solved hop may sit from the one handed in before it is refused.
  double max_disagreement = 0.5;
  // Levenberg damping on the translation solve, and the furthest one step may
  // travel. Bearings from a narrow forward field of view leave the along-track
  // direction weakly determined, and an undamped step in it runs away.
  double damping = 1e-3;
  double max_step_m = 0.3;
};

// Inverse-depth landmarks in the body frame, propagated by the hop.
//
// The state is kept in the CURRENT body frame and moved forward with the
// vehicle rather than against a reference pose. That costs a little accuracy in
// the linearisation and buys the whole of the bookkeeping: no pose history, no
// reference frames, nothing to rebuild when the trajectory is corrected.
class LandmarkMap
{
public:
  explicit LandmarkMap(const LandmarkSettings & settings)
  : settings_(settings) {}

  // Fold one frame in. `bearings` are unit rays in the body frame at the
  // current frame, `ids` names them, and `rotation` and `translation` are the
  // motion from the previous body frame to this one -- the same hop the plane
  // solved, which is what makes the depths metric.
  //
  // Landmarks already held are predicted through the motion and corrected by
  // their new bearing; ones that are new are started at the prior depth.
  void observe(
    const Eigen::Matrix<double, Eigen::Dynamic, 3> & bearings,
    const std::vector<int64_t> & ids, const Eigen::Matrix3d & rotation,
    const Eigen::Vector3d & translation);

  // The hop those landmarks imply, given the same rotation and the bearings
  // they are seen at now. Solved before `observe` moves the state forward, so
  // the state is still expressed in the previous frame.
  //
  // Returns nothing where too few landmarks have converged. This is the whole
  // point of the class: it answers on frames the road cannot.
  std::optional<Eigen::Vector3d> solve_hop(
    const Eigen::Matrix<double, Eigen::Dynamic, 3> & bearings,
    const std::vector<int64_t> & ids, const Eigen::Matrix3d & rotation,
    const Eigen::Vector3d & guess, int & votes, double * residual_rms = nullptr) const;

  int size() const {return static_cast<int>(held_.size());}
  int converged() const;
  // Median depth of the landmarks that may vote, in metres. Zero where none can.
  double median_depth() const;

private:
  struct Landmark
  {
    // Unit bearing in the current body frame, and the inverse of the distance
    // along it.
    Eigen::Vector3d bearing = Eigen::Vector3d::UnitX();
    double inverse_depth = 0.05;
    double variance = 1.0;
    int observations = 0;
    int missed = 0;
  };

  bool usable(const Landmark & mark) const;

  LandmarkSettings settings_;
  std::unordered_map<int64_t, Landmark> held_;
  // Which ids arrived this frame, so the rest can still be predicted.
  mutable std::unordered_set<int64_t> seen_;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__LANDMARKS_HPP_
