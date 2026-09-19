#include "monoscale_core/landmarks.hpp"

#include <algorithm>
#include <vector>
#include <cmath>

namespace monoscale
{
namespace
{

// Two directions spanning the plane a unit bearing is normal to. A bearing
// error has two degrees of freedom and this is the basis they are read in;
// which two does not matter, only that they are orthogonal to the bearing and
// to each other.
void tangent_basis(const Eigen::Vector3d & bearing, Eigen::Vector3d & u, Eigen::Vector3d & v)
{
  const Eigen::Vector3d seed = std::abs(bearing.x()) < 0.9
    ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  u = bearing.cross(seed).normalized();
  v = bearing.cross(u);
}

}  // namespace

bool LandmarkMap::usable(const Landmark & mark) const
{
  if (mark.observations < 2 || !(mark.inverse_depth > 0.0)) {
    return false;
  }
  const double depth = 1.0 / mark.inverse_depth;
  if (depth < settings_.min_depth_m || depth > settings_.max_depth_m) {
    return false;
  }
  // Converged when the depth is known to a fraction of itself. In inverse depth
  // that is a fraction of the inverse depth, which is the same statement and
  // the reason the state is kept this way round.
  return std::sqrt(mark.variance) <
         settings_.converged_fraction * mark.inverse_depth;
}

int LandmarkMap::converged() const
{
  int count = 0;
  for (const auto & entry : held_) {
    count += usable(entry.second) ? 1 : 0;
  }
  return count;
}

double LandmarkMap::median_depth() const
{
  std::vector<double> depths;
  for (const auto & entry : held_) {
    if (usable(entry.second)) {
      depths.push_back(1.0 / entry.second.inverse_depth);
    }
  }
  if (depths.empty()) {
    return 0.0;
  }
  std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
  return depths[depths.size() / 2];
}

std::optional<Eigen::Vector3d> LandmarkMap::solve_hop(
  const Eigen::Matrix<double, Eigen::Dynamic, 3> & bearings,
  const std::vector<int64_t> & ids, const Eigen::Matrix3d & rotation,
  const Eigen::Vector3d & guess, int & votes, double * residual_rms) const
{
  votes = 0;
  double first_residual = 0.0;
  int64_t residual_count = 0;
  // Gauss-Newton on the translation alone. The rotation is handed in because
  // the gyro knows it far better than these bearings do -- 0.5 deg/100m over a
  // drive -- and solving for it here would spend the landmarks' information on
  // a quantity that is already settled.
  Eigen::Vector3d hop = guess;
  for (int iteration = 0; iteration < 6; ++iteration) {
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    int used = 0;
    for (Eigen::Index i = 0; i < bearings.rows(); ++i) {
      const auto found = held_.find(ids[static_cast<size_t>(i)]);
      if (found == held_.end() || !usable(found->second)) {
        continue;
      }
      const Landmark & mark = found->second;
      const Eigen::Vector3d point = mark.bearing / mark.inverse_depth;
      const Eigen::Vector3d moved = rotation * (point - hop);
      const double range = moved.norm();
      if (!(range > 1e-6)) {
        continue;
      }
      const Eigen::Vector3d predicted = moved / range;
      Eigen::Vector3d u;
      Eigen::Vector3d v;
      tangent_basis(predicted, u, v);
      const Eigen::Vector3d observed = bearings.row(i).transpose().normalized();
      // The residual is where the observed bearing lands in the predicted
      // one's tangent plane, which for small angles is the angle itself.
      const Eigen::Vector2d residual(u.dot(observed), v.dot(observed));
      if (iteration == 0) {
        first_residual += residual.squaredNorm();
        ++residual_count;
      }
      // d(predicted)/d(hop) = (I - pp^T)/range * (-R)
      const Eigen::Matrix3d projector =
        (Eigen::Matrix3d::Identity() - predicted * predicted.transpose()) / range;
      const Eigen::Matrix3d derivative = -projector * rotation;
      Eigen::Matrix<double, 2, 3> jacobian;
      jacobian.row(0) = u.transpose() * derivative;
      jacobian.row(1) = v.transpose() * derivative;
      // A landmark whose depth is poorly known predicts a bearing that is
      // poorly known, and along exactly one direction: the one the baseline
      // sweeps it through. Adding that to the bearing noise is what keeps a
      // half-converged landmark from voting as though it were sure.
      const Eigen::Vector3d along = projector * rotation * mark.bearing /
        (mark.inverse_depth * mark.inverse_depth);
      const Eigen::Vector2d sweep(u.dot(along), v.dot(along));
      const double noise = settings_.bearing_sigma_rad * settings_.bearing_sigma_rad;
      const Eigen::Matrix2d covariance =
        Eigen::Matrix2d::Identity() * noise + sweep * sweep.transpose() * mark.variance;
      const Eigen::Matrix2d information = covariance.inverse();
      normal += jacobian.transpose() * information * jacobian;
      gradient -= jacobian.transpose() * information * residual;
      ++used;
    }
    if (used < settings_.min_votes) {
      votes = used;
      return std::nullopt;
    }
    votes = used;
    // The vertical is not observable from one hop of a road vehicle and it is
    // not being asked for: pinning it leaves the two the hop actually has.
    normal(2, 2) += 1e6;
    for (int k = 0; k < 3; ++k) {
      normal(k, k) += settings_.damping * std::max(normal(k, k), 1e-9);
    }
    Eigen::Vector3d step = normal.ldlt().solve(gradient);
    if (!step.allFinite()) {
      return std::nullopt;
    }
    const double travel = step.head<2>().norm();
    if (travel > settings_.max_step_m) {
      step *= settings_.max_step_m / travel;
    }
    hop += step;
    if (step.head<2>().norm() < 1e-4) {
      break;
    }
  }
  if (residual_rms != nullptr && residual_count > 0) {
    *residual_rms = std::sqrt(first_residual / static_cast<double>(residual_count));
  }
  if (!hop.allFinite()) {
    return std::nullopt;
  }
  if (settings_.max_disagreement > 0.0 &&
    (hop - guess).head<2>().norm() > settings_.max_disagreement)
  {
    return std::nullopt;
  }
  return hop;
}

void LandmarkMap::observe(
  const Eigen::Matrix<double, Eigen::Dynamic, 3> & bearings,
  const std::vector<int64_t> & ids, const Eigen::Matrix3d & rotation,
  const Eigen::Vector3d & translation)
{
  for (auto & entry : held_) {
    ++entry.second.missed;
  }
  seen_.clear();
  for (Eigen::Index i = 0; i < bearings.rows(); ++i) {
    seen_.insert(ids[static_cast<size_t>(i)]);
  }
  for (Eigen::Index i = 0; i < bearings.rows(); ++i) {
    const int64_t id = ids[static_cast<size_t>(i)];
    const Eigen::Vector3d observed = bearings.row(i).transpose().normalized();
    if (!observed.allFinite()) {
      continue;
    }
    auto found = held_.find(id);
    if (found == held_.end()) {
      if (static_cast<int>(held_.size()) >= settings_.max_landmarks) {
        continue;
      }
      Landmark fresh;
      fresh.bearing = observed;
      fresh.inverse_depth = 1.0 / std::max(settings_.initial_depth_m, 1e-3);
      fresh.variance = settings_.initial_inverse_sigma * settings_.initial_inverse_sigma;
      fresh.observations = 1;
      fresh.missed = 0;
      held_.emplace(id, fresh);
      continue;
    }
    Landmark & mark = found->second;
    mark.missed = 0;
    if (mark.observations == 1) {
      // The second sighting triangulates rather than filters.
      //
      // An extended filter linearises about its prior, and the prior here is
      // one number for the whole scene. A landmark truly at 5 m, started at 20,
      // is corrected about a Jacobian taken four times too far away and settles
      // half as far out as it should -- measured, the depths that converged
      // read 9.8 m where the geometry says the near road, and the hop they
      // voted for came out 1.57 times the one the plane had. Two rays and a
      // baseline give the depth outright, and after that the filter has
      // something worth linearising about.
      const Eigen::Vector3d a = mark.bearing;
      const Eigen::Vector3d c = rotation.transpose() * observed;
      const double aa = a.dot(a);
      const double cc = c.dot(c);
      const double ac = a.dot(c);
      // Minimising |d1 a - d2 c - t|^2 over both depths gives
      //   [ aa  -ac ] [d1]   [ a.t]
      //   [-ac   cc ] [d2] = [-c.t]
      // and the first row of its inverse is the depth wanted here.
      const double det = aa * cc - ac * ac;
      if (std::abs(det) > 1e-9) {
        const double at = a.dot(translation);
        const double ct = c.dot(translation);
        const double depth = (at * cc - ac * ct) / det;
        // The parallax the pair actually carries. A ray that has barely moved
        // says nothing about depth however cleanly it is measured, and the
        // uncertainty has to say so.
        const double parallax = std::acos(std::clamp(ac / std::sqrt(aa * cc), -1.0, 1.0));
        if (std::isfinite(depth) && depth > settings_.min_depth_m &&
          depth < settings_.max_depth_m && parallax > 1e-4)
        {
          const Eigen::Vector3d placed = rotation * (a * depth - translation);
          const double range = placed.norm();
          if (range > 1e-6) {
            mark.bearing = observed;
            mark.inverse_depth = 1.0 / range;
            const double relative = settings_.bearing_sigma_rad / parallax;
            mark.variance = std::pow(mark.inverse_depth * relative, 2);
            ++mark.observations;
            continue;
          }
        }
      }
      // No parallax worth the name: leave it where it was and wait.
      mark.bearing = observed;
      continue;
    }
    // Predict: where the held point ends up after the hop, and what bearing it
    // would then be seen at.
    const Eigen::Vector3d point = mark.bearing / mark.inverse_depth;
    const Eigen::Vector3d moved = rotation * (point - translation);
    const double range = moved.norm();
    if (!(range > 1e-6)) {
      continue;
    }
    const Eigen::Vector3d predicted = moved / range;
    Eigen::Vector3d u;
    Eigen::Vector3d v;
    tangent_basis(predicted, u, v);
    // The one direction the inverse depth can move the prediction in. A hop
    // perpendicular to the bearing sweeps it; a hop along the bearing does
    // not, which is why a feature dead ahead never converges.
    const Eigen::Matrix3d projector =
      (Eigen::Matrix3d::Identity() - predicted * predicted.transpose()) / range;
    const Eigen::Vector3d along = -projector * rotation * mark.bearing /
      (mark.inverse_depth * mark.inverse_depth);
    const Eigen::Vector2d jacobian(u.dot(along), v.dot(along));
    const Eigen::Vector2d residual(u.dot(observed), v.dot(observed));
    const double noise = settings_.bearing_sigma_rad * settings_.bearing_sigma_rad;
    const double innovation = jacobian.squaredNorm() * mark.variance + noise;
    if (!(innovation > 0.0)) {
      continue;
    }
    const Eigen::Vector2d gain = mark.variance * jacobian / innovation;
    const double correction = gain.dot(residual);
    double updated = mark.inverse_depth + correction;
    // The depth is propagated to the new frame whether or not the correction
    // was sane: the state lives in the body frame and the body has moved.
    const double propagated_range = range;
    if (std::isfinite(updated) && updated > 1.0 / settings_.max_depth_m &&
      updated < 1.0 / settings_.min_depth_m)
    {
      // Re-express the corrected inverse depth in the new frame. The
      // correction was made in the old one, so the point has to be moved
      // through the same hop again with the new depth.
      const Eigen::Vector3d corrected = rotation * (mark.bearing / updated - translation);
      const double corrected_range = corrected.norm();
      if (corrected_range > 1e-6) {
        const double reduced = std::max((1.0 - gain.dot(jacobian)) * mark.variance, 1e-12);
        // The variance has to be carried through the change of frame with the
        // state, and leaving it behind is what turns this into a filter that
        // only ever grows more confident. Driving towards a landmark raises its
        // inverse depth, and the uncertainty in that inverse depth rises with
        // it: with d = 1/rho and y = b d - t,
        //
        //     d(rho_new)/d(rho_old) = d^2 (d - b.t) / |y|^3
        //
        // Without it every landmark reaches the convergence test eventually
        // whatever its parallax, and the hop they then vote on is 8.3 degrees
        // out at the pose the plane already knows is right.
        const double depth = 1.0 / updated;
        const double shift = depth - mark.bearing.dot(translation);
        const double jacobian_frame =
          depth * depth * shift / (corrected_range * corrected_range * corrected_range);
        // The bearing is taken from what was just seen, not from what was
        // predicted. Only the depth is hidden; the direction is measured every
        // frame to a pixel, and overwriting it with a prediction lets it drift
        // for as long as the landmark lives. Measured, that drift was the whole
        // of the error: the stored bearings sat 8.3 degrees off the observed
        // ones of the same features, and no convention, no staleness and no
        // amount of motion accounted for it.
        mark.bearing = observed;
        mark.inverse_depth = 1.0 / corrected_range;
        mark.variance = std::max(jacobian_frame * jacobian_frame * reduced, 1e-14);
        ++mark.observations;
        continue;
      }
    }
    mark.bearing = observed;
    mark.inverse_depth = 1.0 / propagated_range;
    ++mark.observations;
  }
  // Everything that was not seen this frame still moved.
  //
  // Leaving them frozen and then letting them vote when they reappear is what
  // made this read 8.6 degrees out at a pose the plane already had right: with
  // five frames of patience a re-seen landmark carried five hops of staleness,
  // and five hops at seventeen pixels a hop is eighty-five pixels. There is no
  // correction to make without an observation, but the prediction is owed
  // whether or not one arrives.
  for (auto & entry : held_) {
    Landmark & mark = entry.second;
    if (seen_.count(entry.first) != 0 || !(mark.inverse_depth > 0.0)) {
      continue;
    }
    const Eigen::Vector3d moved = rotation * (mark.bearing / mark.inverse_depth - translation);
    const double range = moved.norm();
    if (!(range > 1e-6)) {
      continue;
    }
    const double depth = 1.0 / mark.inverse_depth;
    const double shift = depth - mark.bearing.dot(translation);
    const double jacobian = depth * depth * shift / (range * range * range);
    mark.bearing = moved / range;
    mark.inverse_depth = 1.0 / range;
    mark.variance = std::max(jacobian * jacobian * mark.variance, 1e-14);
  }
  for (auto it = held_.begin(); it != held_.end(); ) {
    it = it->second.missed > settings_.patience ? held_.erase(it) : std::next(it);
  }
}

}  // namespace monoscale
