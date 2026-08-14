#include "monoscale_core/anchors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace monoscale
{

GroundAnchorMap::GroundAnchorMap(const AnchorSettings & settings)
: settings_(settings)
{
  const int capacity = std::max(settings_.max_anchors, 1) + 1;
  position_.setZero(capacity, 2);
  observation_.setZero(capacity);
  variance_.setZero(capacity);
  seen_.setZero(capacity);
  information_.setZero(capacity);
  identifier_.setConstant(capacity, -1);
  free_.reserve(static_cast<size_t>(capacity));
  for (int slot = capacity - 1; slot >= 0; --slot) {
    free_.push_back(slot);
  }
  by_id_.assign(4096, -1);
}

int64_t GroundAnchorMap::slot_of(int64_t identity) const
{
  if (identity < 0 || identity >= static_cast<int64_t>(by_id_.size())) {
    return -1;
  }
  return by_id_[static_cast<size_t>(identity)];
}

void GroundAnchorMap::grow_table(int64_t highest)
{
  if (highest < static_cast<int64_t>(by_id_.size())) {
    return;
  }
  // Geometric, and never below the identity that asked for it, so a whole
  // drive costs a handful of reallocations.
  const size_t size = std::max(2 * by_id_.size(), static_cast<size_t>(highest) + 1);
  by_id_.resize(size, -1);
}

double GroundAnchorMap::weight_at(int64_t slot) const
{
  const double count = static_cast<double>(
    std::min<int64_t>(observation_(slot), settings_.max_observations));
  if (!settings_.select_by_consistency) {
    return count;
  }
  return count / std::max(variance_(slot), 1e-4);
}

void GroundAnchorMap::anchored(const Identities & ids, Mask & out) const
{
  out.resize(ids.size());
  for (Eigen::Index i = 0; i < ids.size(); ++i) {
    out(i) = slot_of(ids(i)) >= 0;
  }
}

void GroundAnchorMap::anchor_view(
  const Identities & ids, Points2 & world_out, Weights & weights_out) const
{
  world_out.resize(ids.size(), 2);
  weights_out.resize(ids.size());
  for (Eigen::Index i = 0; i < ids.size(); ++i) {
    const int64_t slot = slot_of(ids(i));
    if (slot < 0) {
      world_out(i, 0) = 0.0;
      world_out(i, 1) = 0.0;
      weights_out(i) = 0.0;
      continue;
    }
    world_out(i, 0) = position_(slot, 0);
    world_out(i, 1) = position_(slot, 1);
    weights_out(i) = weight_at(slot);
  }
}

void GroundAnchorMap::update(
  const Identities & ids, const Points2 & world_points, bool allow_new,
  const Weights & information)
{
  ++frame_;
  const Eigen::Index count = ids.size();
  const bool weighted = information.size() == count;

  // Sightings first: every frame refines what it can see.
  std::vector<Eigen::Index> fresh;
  for (Eigen::Index i = 0; i < count; ++i) {
    const double x = world_points(i, 0);
    const double y = world_points(i, 1);
    if (!std::isfinite(x) || !std::isfinite(y)) {
      continue;
    }
    const int64_t slot = slot_of(ids(i));
    if (slot < 0) {
      if (allow_new) {
        fresh.push_back(i);
      }
      continue;
    }
    const int64_t counted = observation_(slot);
    double gain;
    if (!weighted) {
      gain = std::max(settings_.update_gain, 1.0 / (static_cast<double>(counted) + 1.0));
    } else {
      // A running mean weighted by precision: each sighting moves the anchor by
      // its share of everything the anchor now knows.
      const double worth = std::max(information(i), 1e-9);
      const double total = information_(slot) + worth;
      gain = worth / total;
      information_(slot) = total;
    }
    const double dx = x - position_(slot, 0);
    const double dy = y - position_(slot, 1);
    const double residual = dx * dx + dy * dy;
    variance_(slot) += gain * (residual - variance_(slot));
    position_(slot, 0) += gain * dx;
    position_(slot, 1) += gain * dy;
    observation_(slot) = counted + 1;
    seen_(slot) = frame_;
  }

  // Then births, up to whatever room the free list has.
  const size_t room = std::min(fresh.size(), free_.size());
  if (room > 0) {
    int64_t highest = 0;
    for (size_t n = 0; n < room; ++n) {
      highest = std::max(highest, ids(fresh[n]));
    }
    grow_table(highest);
    for (size_t n = 0; n < room; ++n) {
      const Eigen::Index i = fresh[n];
      const int64_t slot = free_.back();
      free_.pop_back();
      position_(slot, 0) = world_points(i, 0);
      position_(slot, 1) = world_points(i, 1);
      observation_(slot) = 1;
      variance_(slot) = settings_.initial_variance;
      seen_(slot) = frame_;
      identifier_(slot) = ids(i);
      information_(slot) = weighted ? std::max(information(i), 1e-9) : 1.0;
      by_id_[static_cast<size_t>(ids(i))] = slot;
      ++live_;
    }
  }
  prune();
}

void GroundAnchorMap::forget(int64_t slot)
{
  const int64_t gone = identifier_(slot);
  if (gone >= 0 && gone < static_cast<int64_t>(by_id_.size())) {
    by_id_[static_cast<size_t>(gone)] = -1;
  }
  identifier_(slot) = -1;
  observation_(slot) = 0;
  free_.push_back(slot);
  --live_;
}

void GroundAnchorMap::prune()
{
  for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
    if (identifier_(slot) < 0) {
      continue;
    }
    const bool stale = frame_ - seen_(slot) > settings_.max_age_frames;
    // Given a few sightings to settle, an anchor that still scatters is not a
    // landmark and should not be registered against.
    const bool scattered = settings_.select_by_consistency &&
      observation_(slot) >= settings_.trial_observations &&
      variance_(slot) > settings_.max_variance;
    if (!stale && !scattered) {
      continue;
    }
    if (scattered && !stale) {
      ++discarded_;
    }
    forget(slot);
  }

  const int surplus = live_ - settings_.max_anchors;
  if (surplus <= 0) {
    return;
  }
  // Drop the least trustworthy first; they carry the least history.
  std::vector<std::pair<double, int64_t>> scored;
  scored.reserve(static_cast<size_t>(live_));
  for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
    if (identifier_(slot) >= 0) {
      scored.emplace_back(weight_at(slot), slot);
    }
  }
  std::nth_element(
    scored.begin(), scored.begin() + surplus, scored.end(),
    [](const auto & a, const auto & b) {return a.first < b.first;});
  for (int n = 0; n < surplus; ++n) {
    forget(scored[static_cast<size_t>(n)].second);
  }
}

std::optional<Eigen::Vector2d> GroundAnchorMap::position_of(int64_t identity) const
{
  const int64_t slot = slot_of(identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return Eigen::Vector2d(position_(slot, 0), position_(slot, 1));
}

std::optional<int> GroundAnchorMap::observations_of(int64_t identity) const
{
  const int64_t slot = slot_of(identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return static_cast<int>(observation_(slot));
}

std::optional<double> GroundAnchorMap::variance_of(int64_t identity) const
{
  const int64_t slot = slot_of(identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return variance_(slot);
}

std::optional<double> GroundAnchorMap::weight_of(int64_t identity) const
{
  const int64_t slot = slot_of(identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return weight_at(slot);
}

std::optional<AnchorAlignment> align_to_anchors(
  const Points2 & body_points, const Points2 & world_points, const Weights & weights_in,
  double yaw, double threshold, int min_inliers, bool refine_yaw,
  const Eigen::Vector2d & origin, double radial_min_range)
{
  const Eigen::Index count = body_points.rows();
  if (count < std::max<Eigen::Index>(2, min_inliers) || world_points.rows() != count) {
    return std::nullopt;
  }
  const bool weighted = weights_in.size() == count;
  const auto weight_of = [&](Eigen::Index i) {return weighted ? weights_in(i) : 1.0;};

  double applied = std::numeric_limits<double>::infinity();
  Eigen::Vector2d centre = Eigen::Vector2d::Zero();
  bool have_centre = false;
  std::vector<char> inliers(static_cast<size_t>(count), 0);
  Points2 votes(count, 2);
  const double threshold_squared = threshold * threshold;

  for (int iteration = 0; iteration < 4; ++iteration) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    for (Eigen::Index i = 0; i < count; ++i) {
      const double bx = body_points(i, 0);
      const double by = body_points(i, 1);
      votes(i, 0) = world_points(i, 0) - (c * bx - s * by);
      votes(i, 1) = world_points(i, 1) - (s * bx + c * by);
    }
    if (!have_centre) {
      std::vector<double> xs(static_cast<size_t>(count));
      std::vector<double> ys(static_cast<size_t>(count));
      for (Eigen::Index i = 0; i < count; ++i) {
        xs[static_cast<size_t>(i)] = votes(i, 0);
        ys[static_cast<size_t>(i)] = votes(i, 1);
      }
      const auto median = [](std::vector<double> & values) {
          const size_t middle = values.size() / 2;
          std::nth_element(values.begin(), values.begin() + middle, values.end());
          const double upper = values[middle];
          if (values.size() % 2 == 1) {
            return upper;
          }
          const double lower = *std::max_element(values.begin(), values.begin() + middle);
          return 0.5 * (lower + upper);
        };
      centre = Eigen::Vector2d(median(xs), median(ys));
      have_centre = true;
    }

    // Three reweighting passes at most. The Python ran all three every time;
    // in practice the centre settles in one or two, and a pass that moves it by
    // less than a micrometre cannot change an inlier set gated at centimetres.
    for (int pass = 0; pass < 3; ++pass) {
      Eigen::Index kept = 0;
      for (Eigen::Index i = 0; i < count; ++i) {
        const double dx = votes(i, 0) - centre.x();
        const double dy = votes(i, 1) - centre.y();
        const bool inside = dx * dx + dy * dy <= threshold_squared;
        inliers[static_cast<size_t>(i)] = inside ? 1 : 0;
        kept += inside ? 1 : 0;
      }
      if (kept < min_inliers) {
        return std::nullopt;
      }
      double total = 0.0;
      Eigen::Vector2d sum = Eigen::Vector2d::Zero();
      for (Eigen::Index i = 0; i < count; ++i) {
        if (!inliers[static_cast<size_t>(i)]) {
          continue;
        }
        const double w = weight_of(i);
        total += w;
        sum.x() += w * votes(i, 0);
        sum.y() += w * votes(i, 1);
      }
      const Eigen::Vector2d moved = sum / total;
      const double step = (moved - centre).norm();
      centre = moved;
      if (step < 1e-6) {
        break;
      }
    }

    if (!refine_yaw || iteration == 3) {
      break;
    }

    // The rotation that best carries the inlier body points onto their anchors,
    // about each set's own weighted centroid. Closed form in 2D: no search, and
    // the translation falls out of it unchanged.
    double total = 0.0;
    Eigen::Vector2d body_centre = Eigen::Vector2d::Zero();
    Eigen::Vector2d world_centre = Eigen::Vector2d::Zero();
    for (Eigen::Index i = 0; i < count; ++i) {
      if (!inliers[static_cast<size_t>(i)]) {
        continue;
      }
      const double w = weight_of(i);
      total += w;
      body_centre += w * Eigen::Vector2d(body_points(i, 0), body_points(i, 1));
      world_centre += w * Eigen::Vector2d(world_points(i, 0), world_points(i, 1));
    }
    body_centre /= total;
    world_centre /= total;

    double cross = 0.0;
    double dot = 0.0;
    double lever_sum = 0.0;
    double fit_sum = 0.0;
    double kept = 0.0;
    for (Eigen::Index i = 0; i < count; ++i) {
      if (!inliers[static_cast<size_t>(i)]) {
        continue;
      }
      const double w = weight_of(i);
      const double bx = body_points(i, 0) - body_centre.x();
      const double by = body_points(i, 1) - body_centre.y();
      const double tx = world_points(i, 0) - world_centre.x();
      const double ty = world_points(i, 1) - world_centre.y();
      cross += w * (bx * ty - by * tx);
      dot += w * (bx * tx + by * ty);
      lever_sum += w * (bx * bx + by * by);
      const double rx = votes(i, 0) - centre.x();
      const double ry = votes(i, 1) - centre.y();
      fit_sum += rx * rx + ry * ry;
      kept += 1.0;
    }
    if (std::abs(cross) < 1e-12 && std::abs(dot) < 1e-12) {
      break;
    }
    const double proposed = wrap_pi(std::atan2(cross, dot));
    const double lever = std::sqrt(lever_sum / std::max(total, 1e-9));
    if (lever < 1e-6) {
      break;
    }
    // How well this fit pins a rotation down, which is the caller's to weigh: a
    // residual of `fit` metres seen across a lever arm of `lever` metres is an
    // angle, and averaging it over the inliers shrinks it by their root.
    const double fit = std::sqrt(fit_sum / std::max(kept, 1.0));
    applied = fit / (lever * std::sqrt(std::max(kept, 1.0)));
    const bool settled = std::abs(wrap_pi(proposed - yaw)) < 1e-6;
    yaw = proposed;
    if (settled) {
      break;
    }
  }

  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  AnchorAlignment result;
  result.inliers.resize(count);
  Eigen::Index kept = 0;
  double squared = 0.0;
  // The radial residuals, held until the reference range is known: normalising
  // by it is what keeps the two-term fit from spanning r to r^4.
  std::vector<double> ranges;
  std::vector<double> radial;
  std::vector<double> radial_weight;
  double range_total = 0.0;
  double range_weight = 0.0;
  ranges.reserve(static_cast<size_t>(count));
  radial.reserve(static_cast<size_t>(count));
  radial_weight.reserve(static_cast<size_t>(count));
  for (Eigen::Index i = 0; i < count; ++i) {
    const double rx = world_points(i, 0) - (c * body_points(i, 0) - s * body_points(i, 1)) -
      centre.x();
    const double ry = world_points(i, 1) - (s * body_points(i, 0) + c * body_points(i, 1)) -
      centre.y();
    const double residual_squared = rx * rx + ry * ry;
    const bool inside = residual_squared <= threshold_squared;
    result.inliers(i) = inside;
    if (!inside) {
      continue;
    }
    ++kept;
    squared += residual_squared;

    // The residual is in the world frame and the bearing is in the body's, so
    // one of them has to be carried across. Turning the residual back is the
    // cheaper way round: the rotation is the same for every point.
    const double bx = body_points(i, 0) - origin.x();
    const double by = body_points(i, 1) - origin.y();
    const double range = std::hypot(bx, by);
    if (!(range > 1e-6) || range < radial_min_range) {
      continue;
    }
    const double body_rx = c * rx + s * ry;
    const double body_ry = -s * rx + c * ry;
    const double w = weight_of(i);
    ranges.push_back(range);
    radial.push_back((bx * body_rx + by * body_ry) / range);
    radial_weight.push_back(w);
    range_total += w * range;
    range_weight += w;
  }
  if (kept < min_inliers) {
    return std::nullopt;
  }
  result.translation = centre;
  result.spread = std::sqrt(squared / static_cast<double>(kept));
  result.yaw = yaw;
  result.yaw_sigma = applied;

  // Weighted least squares of the radial residual on the normalised range and
  // its square, with no constant term: a projection that is wrong about where
  // the ground is is wrong by an amount that goes to zero underneath the lens,
  // so an offset there would be fitting something else.
  const double reference = range_weight > 0.0 ? range_total / range_weight : 0.0;
  if (reference > 1e-6 && ranges.size() >= 3) {
    double s11 = 0.0;
    double b1 = 0.0;
    for (size_t i = 0; i < ranges.size(); ++i) {
      const double x = ranges[i] / reference;
      const double w = radial_weight[i];
      s11 += w * x * x;
      b1 += w * x * radial[i];
    }
    if (s11 > 1e-12) {
      result.radial_linear = b1 / s11;
      result.radial_reference = reference;
    }
  }
  return result;
}

std::optional<CameraTranslation> fuse_by_precision(
  const std::vector<CameraTranslation> & estimates)
{
  if (estimates.empty()) {
    return std::nullopt;
  }
  double total = 0.0;
  double x = 0.0;
  double y = 0.0;
  int count = 0;
  for (const auto & estimate : estimates) {
    const double variance = std::pow(std::max(estimate.spread, 1e-3), 2);
    const double weight = std::max(static_cast<double>(estimate.count), 1.0) / variance;
    total += weight;
    x += weight * estimate.x;
    y += weight * estimate.y;
    count += estimate.count;
  }
  if (!(total > 0.0)) {
    return std::nullopt;
  }
  CameraTranslation fused;
  fused.x = x / total;
  fused.y = y / total;
  fused.count = count;
  return fused;
}

}  // namespace monoscale
