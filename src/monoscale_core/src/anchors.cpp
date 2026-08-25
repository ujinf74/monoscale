#include "monoscale_core/anchors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace monoscale
{

GroundAnchorMap::GroundAnchorMap(const AnchorSettings & settings, int sources)
: settings_(settings), sources_(std::max(sources, 1))
{
  by_id_.resize(static_cast<size_t>(sources_));
  const int capacity = std::max(settings_.max_anchors, 1) + 1;
  position_.setZero(capacity, 2);
  observation_.setZero(capacity);
  variance_.setZero(capacity);
  seen_.setZero(capacity);
  information_.setZero(capacity);
  identifier_.setConstant(capacity, -1);
  owner_.setConstant(capacity, sources_, -1);
  founder_.setConstant(capacity, -1);
  founded_path_.setZero(capacity);
  free_.reserve(static_cast<size_t>(capacity));
  for (int slot = capacity - 1; slot >= 0; --slot) {
    free_.push_back(slot);
  }
  for (auto & table : by_id_) {
    table.clear();
  }
}

int64_t GroundAnchorMap::slot_of(int source, int64_t identity) const
{
  if (source < 0 || source >= sources_) {
    return -1;
  }
  const auto & table = by_id_[static_cast<size_t>(source)];
  const auto found = table.find(identity);
  return found == table.end() ? -1 : found->second;
}

int64_t GroundAnchorMap::cell_of(double x, double y) const
{
  const double size = std::max(settings_.link_radius_m, 1e-3);
  const int64_t cx = static_cast<int64_t>(std::floor(x / size));
  const int64_t cy = static_cast<int64_t>(std::floor(y / size));
  // Cantor-style mix; the map is sparse so collisions only cost a comparison.
  return (cx << 32) ^ (cy & 0xffffffffLL);
}

void GroundAnchorMap::grid_insert(int64_t slot)
{
  if (settings_.link_radius_m <= 0.0) {
    return;
  }
  grid_[cell_of(position_(slot, 0), position_(slot, 1))].push_back(slot);
}

void GroundAnchorMap::grid_erase(int64_t slot)
{
  if (settings_.link_radius_m <= 0.0) {
    return;
  }
  const auto found = grid_.find(cell_of(position_(slot, 0), position_(slot, 1)));
  if (found == grid_.end()) {
    return;
  }
  auto & bucket = found->second;
  for (size_t i = 0; i < bucket.size(); ++i) {
    if (bucket[i] == slot) {
      bucket[i] = bucket.back();
      bucket.pop_back();
      break;
    }
  }
}

int64_t GroundAnchorMap::adoptable(int source, double x, double y) const
{
  if (settings_.link_radius_m <= 0.0) {
    return -1;
  }
  const double size = std::max(settings_.link_radius_m, 1e-3);
  const double limit = settings_.link_radius_m * settings_.link_radius_m;
  const int64_t cx = static_cast<int64_t>(std::floor(x / size));
  const int64_t cy = static_cast<int64_t>(std::floor(y / size));
  int64_t best = -1;
  double best_distance = limit;
  for (int64_t dy = -1; dy <= 1; ++dy) {
    for (int64_t dx = -1; dx <= 1; ++dx) {
      const auto found = grid_.find(((cx + dx) << 32) ^ ((cy + dy) & 0xffffffffLL));
      if (found == grid_.end()) {
        continue;
      }
      for (const int64_t slot : found->second) {
        if (identifier_(slot) < 0) {
          continue;
        }
        // An anchor this camera already wrote to *this frame* is spoken for --
        // taking it would fold two live features into one. One it owns from an
        // earlier frame is a different matter: that track has since been lost
        // and re-detected under a new identity, and rebinding is exactly what
        // reconnects them. Refusing that was costing more than the
        // cross-camera case it was written for.
        // "Not seen this frame" is not the same as "gone". A feature that
        // blinks out for a frame or two is still being tracked, and taking its
        // anchor leaves it with none when it comes back. Only rebind an anchor
        // this source has not touched for a while.
        if (owner_(slot, source) >= 0 &&
          frame_ - seen_(slot) < settings_.link_rebind_grace_frames)
        {
          continue;
        }
        const double ex = position_(slot, 0) - x;
        const double ey = position_(slot, 1) - y;
        const double distance = ex * ex + ey * ey;
        if (distance < best_distance) {
          best_distance = distance;
          best = slot;
        }
      }
    }
  }
  return best;
}

double GroundAnchorMap::weight_at(int64_t slot) const
{
  const double count = settings_.weight_by_information
    ? information_(slot)
    : static_cast<double>(
    std::min<int64_t>(observation_(slot), settings_.max_observations));
  if (!settings_.select_by_consistency) {
    return count;
  }
  return count / std::max(variance_(slot), 1e-4);
}

void GroundAnchorMap::anchored(int source, const Identities & ids, Mask & out) const
{
  out.resize(ids.size());
  for (Eigen::Index i = 0; i < ids.size(); ++i) {
    out(i) = slot_of(source, ids(i)) >= 0;
  }
}

void GroundAnchorMap::anchor_view(
  int source, const Identities & ids, Points2 & world_out, Weights & weights_out) const
{
  world_out.resize(ids.size(), 2);
  weights_out.resize(ids.size());
  for (Eigen::Index i = 0; i < ids.size(); ++i) {
    const int64_t slot = slot_of(source, ids(i));
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
  int source, const Identities & ids, const Points2 & world_points, bool allow_new,
  const Weights & information)
{
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
    int64_t slot = slot_of(source, ids(i));
    if (slot < 0) {
      // A camera meeting ground another camera already anchored adopts that
      // anchor instead of founding a rival one in the same place.
      const int64_t adopted = adoptable(source, x, y);
      if (adopted >= 0) {
        // The crossing, read as a length. Positive along-track means this
        // camera puts the ground further on than the anchor does, which is a
        // pose that ran long over the stretch between the two sightings.
        const double travel = path_ - founded_path_(adopted);
        if (travel > 1.0) {
          const double along = (x - position_(adopted, 0)) * std::cos(yaw_) +
            (y - position_(adopted, 1)) * std::sin(yaw_);
          along_ += along;
          gap_travel_ += travel;
          travel_squared_ += travel * travel;
          cross_ += along * travel;
          ++crossings_;
        }
      }
      if (adopted >= 0 && !settings_.link_measure_only) {
        // Release whatever identity this source had bound here before.
        auto & table = by_id_[static_cast<size_t>(source)];
        const int64_t previous = owner_(adopted, source);
        if (previous >= 0) {
          table.erase(previous);
        }
        table[ids(i)] = adopted;
        owner_(adopted, source) = ids(i);
        // What the two cameras disagreed by, and how that compares with how
        // far away the point was.
        const double gap = std::hypot(x - position_(adopted, 0), y - position_(adopted, 1));
        const double range = weighted
          ? 1.0 / std::sqrt(std::max(information(i), 1e-9)) : 0.0;
        link_gap_ += gap;
        link_range_ += range;
        link_ratio_ += range > 1e-6 ? gap / range : 0.0;
        ++adopted_;
        slot = adopted;
      } else {
        if (allow_new) {
          fresh.push_back(i);
        }
        continue;
      }
    }
    // A camera that merely adopted this anchor reads it and keeps it alive,
    // but does not move it: the disagreement is the pose's to answer for.
    if (!settings_.link_adopter_writes && founder_(slot) >= 0 &&
      founder_(slot) != static_cast<int64_t>(source))
    {
      seen_(slot) = frame_;
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
      // Hold the window finite. The sightings are not independent -- each one
      // is written in the world frame the estimate had just settled on -- so
      // averaging more of them does not keep buying precision, it only stops
      // the anchor from ever moving again.
      if (settings_.min_update_gain > 0.0 && gain < settings_.min_update_gain) {
        gain = settings_.min_update_gain;
        information_(slot) = worth / gain;
      }
    }
    const double dx = x - position_(slot, 0);
    const double dy = y - position_(slot, 1);
    const double residual = dx * dx + dy * dy;
    variance_(slot) += gain * (residual - variance_(slot));
    grid_erase(slot);
    position_(slot, 0) += gain * dx;
    position_(slot, 1) += gain * dy;
    grid_insert(slot);
    observation_(slot) = counted + 1;
    seen_(slot) = frame_;
  }

  // Then births. A full map has no free slots and nothing forces it to make
  // any: prune only evicts what is stale or over capacity, and at exactly
  // capacity it is neither. So the map fills once, early, and thereafter the
  // ground in front of the vehicle cannot get in -- measured, only 14-37% of
  // the points a solve could use had an anchor. Make room for what is arriving
  // by giving up what was seen longest ago; it is behind the vehicle and
  // cannot be registered against again.
  if (settings_.evict_for_new && allow_new && fresh.size() > free_.size()) {
    const size_t needed = fresh.size() - free_.size();
    std::vector<std::pair<int64_t, int64_t>> oldest;
    oldest.reserve(static_cast<size_t>(live_));
    for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
      if (identifier_(slot) >= 0 && seen_(slot) != frame_) {
        oldest.emplace_back(seen_(slot), slot);
      }
    }
    const size_t give = std::min(needed, oldest.size());
    if (give > 0) {
      std::nth_element(
        oldest.begin(), oldest.begin() + static_cast<std::ptrdiff_t>(give), oldest.end(),
        [](const auto & a, const auto & b) {return a.first < b.first;});
      for (size_t n = 0; n < give; ++n) {
        forget(oldest[n].second);
      }
    }
  }
  const size_t room = std::min(fresh.size(), free_.size());
  if (room > 0) {
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
      for (int other = 0; other < sources_; ++other) {
        owner_(slot, other) = -1;
      }
      owner_(slot, source) = ids(i);
      founder_(slot) = source;
      founded_path_(slot) = path_;
      by_id_[static_cast<size_t>(source)][ids(i)] = slot;
      grid_insert(slot);
      ++live_;
    }
  }
  prune();
}

void GroundAnchorMap::forget(int64_t slot)
{
  // Every source that had bound an identity to this slot loses it.
  for (int source = 0; source < sources_; ++source) {
    const int64_t gone = owner_(slot, source);
    if (gone >= 0) {
      by_id_[static_cast<size_t>(source)].erase(gone);
    }
    owner_(slot, source) = -1;
  }
  grid_erase(slot);
  founder_(slot) = -1;
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
  // What to give up when the map is full.
  //
  // Dropping the least trustworthy keeps the best-observed anchors, which
  // sounds right and is measurably wrong: those are the ones seen longest ago,
  // so the map fills with ground the vehicle has driven past while the ground
  // in front of it cannot get a slot. Measured, only 14-37% of the points a
  // solve could use had an anchor at all. Scoring by when an anchor was last
  // seen evicts the past instead, which is the half that can no longer be
  // registered against.
  std::vector<std::pair<double, int64_t>> scored;
  scored.reserve(static_cast<size_t>(live_));
  for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
    if (identifier_(slot) >= 0) {
      scored.emplace_back(
        settings_.evict_by_age ? static_cast<double>(seen_(slot)) : weight_at(slot),
        slot);
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
  const int64_t slot = slot_of(0, identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return Eigen::Vector2d(position_(slot, 0), position_(slot, 1));
}

std::optional<int> GroundAnchorMap::observations_of(int64_t identity) const
{
  const int64_t slot = slot_of(0, identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return static_cast<int>(observation_(slot));
}

std::optional<double> GroundAnchorMap::variance_of(int64_t identity) const
{
  const int64_t slot = slot_of(0, identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return variance_(slot);
}

std::optional<double> GroundAnchorMap::weight_of(int64_t identity) const
{
  const int64_t slot = slot_of(0, identity);
  if (slot < 0) {
    return std::nullopt;
  }
  return weight_at(slot);
}

std::optional<AnchorAlignment> align_to_anchors(
  const Points2 & body_points, const Points2 & world_points, const Weights & weights_in,
  double yaw, double threshold, int min_inliers, bool refine_yaw,
  const Eigen::Vector2d & origin, double radial_min_range, double softness,
  const Eigen::Vector2d * translation_prior, int restarts, double ambiguity,
  const Eigen::Vector2d * inertial_hop, double inertial_gate,
  const Weights & residual_scale)
{
  const Eigen::Index count = body_points.rows();
  if (count < std::max<Eigen::Index>(2, min_inliers) || world_points.rows() != count) {
    return std::nullopt;
  }
  const bool weighted = weights_in.size() == count;
  const auto weight_of = [&](Eigen::Index i) {return weighted ? weights_in(i) : 1.0;};
  // How wide this point's residual is allowed to be, relative to the rest.
  //
  // The gate and the Gaussian below are in metres and the measurement is an
  // angle. For a point on the road that is nearly the same thing -- the band
  // is a few metres deep, so every ground residual is amplified by about the
  // same factor. For a point that is not on the road it is not: the same
  // bearing error puts a 20 m landmark ten times further out of place than a
  // 2 m one, so a metre-wide gate silently discards exactly the distant,
  // long-lived structure it was worth adding.
  const bool scaled = residual_scale.size() == count;
  const auto scale_of = [&](Eigen::Index i) {
      return scaled ? std::max(residual_scale(i), 1e-3) : 1.0;
    };

  double applied = std::numeric_limits<double>::infinity();
  Eigen::Vector2d centre = Eigen::Vector2d::Zero();
  bool have_centre = false;
  std::vector<char> inliers(static_cast<size_t>(count), 0);
  // How much each point counts towards the fit. A hard gate makes the estimate
  // a step function of the threshold, which is why sweeping it moves the score
  // in jumps rather than smoothly; a Gaussian falloff of the same width lets a
  // point at the edge count for what it is worth instead of all or nothing.
  std::vector<double> robust(static_cast<size_t>(count), 1.0);
  Points2 votes(count, 2);
  const double threshold_squared = threshold * threshold;
  const double soft_squared = softness > 0.0 ? 2.0 * softness * softness : 0.0;

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

      // The median is one starting point, and a reweighted mean only ever
      // walks downhill from where it starts. When the hop is short the votes
      // spread wider than the hop itself, the median lands near zero, and the
      // iteration settles on "did not move" -- which is a real local optimum,
      // just not the right one. Starting the same iteration from several votes
      // and keeping whichever attracts the most support finds the mode the
      // data actually has, rather than the one nearest the median.
      if (restarts > 1 && count > 2) {
        const auto settle = [&](Eigen::Vector2d seed) {
            for (int pass = 0; pass < 3; ++pass) {
              double total = 0.0;
              Eigen::Vector2d sum = Eigen::Vector2d::Zero();
              for (Eigen::Index i = 0; i < count; ++i) {
                const double dx = votes(i, 0) - seed.x();
                const double dy = votes(i, 1) - seed.y();
                const double distance = dx * dx + dy * dy;
                const double widen = scale_of(i) * scale_of(i);
                double r;
                if (soft_squared > 0.0) {
                  r = std::exp(-distance / (soft_squared * widen));
                } else {
                  r = distance <= threshold_squared * widen ? 1.0 : 0.0;
                }
                const double w = weight_of(i) * r;
                total += w;
                sum += w * Eigen::Vector2d(votes(i, 0), votes(i, 1));
              }
              if (total <= 0.0) {
                return std::make_pair(seed, 0.0);
              }
              seed = sum / total;
            }
            double score = 0.0;
            for (Eigen::Index i = 0; i < count; ++i) {
              const double dx = votes(i, 0) - seed.x();
              const double dy = votes(i, 1) - seed.y();
              const double distance = dx * dx + dy * dy;
              const double widen = scale_of(i) * scale_of(i);
              const double r = soft_squared > 0.0
                ? std::exp(-distance / (soft_squared * widen))
                : (distance <= threshold_squared * widen ? 1.0 : 0.0);
              score += weight_of(i) * r;
            }
            return std::make_pair(seed, score);
          };
        // A mode further from the inertial expectation than the gate allows is
        // not a translation this vehicle can have made in this interval. The
        // commonest such mode is the one at zero: when the hop is short the
        // votes spread wider than it, and "did not move" collects as much
        // weight as the truth. The accelerometer is a poor guide to where the
        // answer is and an excellent one to where it is not.
        const auto admissible = [&](const Eigen::Vector2d & where) {
            if (inertial_hop == nullptr || inertial_gate <= 0.0) {
              return true;
            }
            return (where - *inertial_hop).norm() <= inertial_gate;
          };
        auto best = settle(centre);
        if (!admissible(best.first)) {
          best.second = -1.0;
        }
        // The previous hop is one more place to look, not a place to be
        // dragged to: it competes on the same score as every other start, so
        // it wins only where the data agrees with it.
        if (translation_prior != nullptr) {
          const auto seeded = settle(*translation_prior);
          if (seeded.second > best.second) {
            best = seeded;
          }
        }
        // Evenly spaced votes, so the choice is deterministic and spans the
        // spread rather than clustering wherever the list happens to start.
        const Eigen::Index stride = std::max<Eigen::Index>(1, count / (restarts - 1));
        std::vector<std::pair<Eigen::Vector2d, double>> found;
        found.push_back(best);
        for (Eigen::Index i = 0; i < count; i += stride) {
          auto candidate = settle(Eigen::Vector2d(votes(i, 0), votes(i, 1)));
          if (!admissible(candidate.first)) {
            candidate.second = -1.0;
          }
          found.push_back(candidate);
          if (candidate.second > best.second) {
            best = candidate;
          }
        }
        if (best.second < 0.0) {
          // Every mode was outside the gate. Nothing here is worth believing.
          return std::nullopt;
        }
        centre = best.first;
        have_centre = true;

        // Is the winner actually the answer, or just the first of several the
        // data supports equally? When the hop is short every mode attracts
        // about the same weight, and picking one of them is guessing. The
        // runner-up here is the best mode that is somewhere else, so the ratio
        // says how much the data prefers this translation to a different one.
        // Below the threshold the solve is refused and the filter coasts,
        // which is cheaper than a confident wrong hop.
        if (ambiguity > 0.0) {
          double rival = 0.0;
          const double separation = std::max(threshold, 1e-6);
          for (const auto & entry : found) {
            if ((entry.first - centre).norm() > separation) {
              rival = std::max(rival, entry.second);
            }
          }
          if (rival > 0.0 && best.second < ambiguity * rival) {
            return std::nullopt;
          }
        }
      } else if (translation_prior != nullptr) {
        centre = *translation_prior;
      }
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
        const double distance = dx * dx + dy * dy;
        const bool inside = distance <= threshold_squared;
        inliers[static_cast<size_t>(i)] = inside ? 1 : 0;
        robust[static_cast<size_t>(i)] = soft_squared > 0.0
          ? std::exp(-distance / soft_squared) : (inside ? 1.0 : 0.0);
        kept += inside ? 1 : 0;
      }
      if (kept < min_inliers) {
        return std::nullopt;
      }
      double total = 0.0;
      Eigen::Vector2d sum = Eigen::Vector2d::Zero();
      for (Eigen::Index i = 0; i < count; ++i) {
        const double r = robust[static_cast<size_t>(i)];
        if (r <= 0.0) {
          continue;
        }
        const double w = weight_of(i) * r;
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
      const double r = robust[static_cast<size_t>(i)];
      if (r <= 0.0) {
        continue;
      }
      const double w = weight_of(i) * r;
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
      const double r = robust[static_cast<size_t>(i)];
      if (r <= 0.0) {
        continue;
      }
      const double w = weight_of(i) * r;
      const double bx = body_points(i, 0) - body_centre.x();
      const double by = body_points(i, 1) - body_centre.y();
      const double tx = world_points(i, 0) - world_centre.x();
      const double ty = world_points(i, 1) - world_centre.y();
      cross += w * (bx * ty - by * tx);
      dot += w * (bx * tx + by * ty);
      lever_sum += w * (bx * bx + by * by);
      const double rx = votes(i, 0) - centre.x();
      const double ry = votes(i, 1) - centre.y();
      fit_sum += r * (rx * rx + ry * ry);
      kept += r;
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
