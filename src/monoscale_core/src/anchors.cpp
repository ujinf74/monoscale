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
  variance_slow_.setZero(capacity);
  pending_.assign(static_cast<size_t>(std::max(sources, 1)), {});
  seen_.setZero(capacity);
  information_.setZero(capacity);
  identifier_.setConstant(capacity, -1);
  owner_.setConstant(capacity, sources_, -1);
  sight_pose_.setConstant(capacity, kRemember, -1);
  sight_body_.setZero(capacity, 2 * kRemember);
  sight_weight_.setZero(capacity, kRemember);
  sight_source_.setConstant(capacity, kRemember, -1);
  best_pose_.setConstant(capacity, sources_, -1);
  best_body_.setZero(capacity, 2 * sources_);
  best_weight_.setZero(capacity, sources_);
  sight_count_.setZero(capacity);
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

// Bearing off the heading crossed with range, in the vehicle's frame.
int GroundAnchorMap::polar_cell_of(double x, double y) const
{
  const double c = std::cos(yaw_);
  const double s = std::sin(yaw_);
  const double dx = x - at_x_;
  const double dy = y - at_y_;
  const double f = c * dx + s * dy;
  const double l = -s * dx + c * dy;
  const double range = std::hypot(f, l);
  const double bearing = std::abs(std::atan2(l, f)) * 180.0 / M_PI;
  const int sectors = std::max(
    1, static_cast<int>(std::ceil(180.0 / std::max(settings_.polar_sector_deg, 1e-3))));
  const int rings = std::max(settings_.polar_rings, 1);
  const int sector = std::min(
    sectors - 1, static_cast<int>(bearing / std::max(settings_.polar_sector_deg, 1e-3)));
  const int ring = std::min(
    rings - 1, static_cast<int>(range / std::max(settings_.polar_ring_m, 1e-3)));
  return sector * rings + ring;
}

void GroundAnchorMap::rebuild_polar_counts()
{
  const int sectors = std::max(
    1, static_cast<int>(std::ceil(180.0 / std::max(settings_.polar_sector_deg, 1e-3))));
  const int rings = std::max(settings_.polar_rings, 1);
  polar_.assign(static_cast<size_t>(sectors * rings), 0);
  for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
    if (identifier_(slot) < 0) {
      continue;
    }
    const int cell = polar_cell_of(position_(slot, 0), position_(slot, 1));
    if (cell >= 0 && cell < static_cast<int>(polar_.size())) {
      ++polar_[static_cast<size_t>(cell)];
    }
  }
}

int64_t GroundAnchorMap::density_cell_of(double x, double y) const
{
  const double size = std::max(settings_.density_cell_m, 1e-3);
  const int64_t cx = static_cast<int64_t>(std::floor(x / size));
  const int64_t cy = static_cast<int64_t>(std::floor(y / size));
  return cx * 73856093LL ^ cy * 19349663LL;
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
        if (settings_.link_cross_source_only &&
          founder_(slot) == static_cast<int64_t>(source))
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

// What this anchor is worth for measuring motion along the heading.
double GroundAnchorMap::longitudinal_information_at(double x, double y) const
{
  if (!(lens_height_ > 1e-6)) {
    return 1.0;
  }
  const double c = std::cos(weight_yaw_);
  const double s = std::sin(weight_yaw_);
  const double dx = x - (at_x_ + look_f_ * c - look_l_ * s);
  const double dy = y - (at_y_ + look_f_ * s + look_l_ * c);
  const double f = c * dx + s * dy;
  const double l = -s * dx + c * dy;
  const double range = std::hypot(f, l);
  if (range < 1e-6) {
    return 1.0;
  }
  const double radial = (range * range + lens_height_ * lens_height_) / lens_height_;
  const double cosb = f / range;
  const double sinb = l / range;
  return cosb * cosb / (radial * radial) + sinb * sinb / (range * range);
}

double GroundAnchorMap::longitudinal_information(int64_t slot) const
{
  if (!(lens_height_ > 1e-6)) {
    return 1.0;
  }
  const double c = std::cos(weight_yaw_);
  const double s = std::sin(weight_yaw_);
  const double dx = position_(slot, 0) - (at_x_ + look_f_ * c - look_l_ * s);
  const double dy = position_(slot, 1) - (at_y_ + look_f_ * s + look_l_ * c);
  const double f = c * dx + s * dy;
  const double l = -s * dx + c * dy;
  const double range = std::hypot(f, l);
  if (range < 1e-6) {
    return 1.0;
  }
  const double radial = (range * range + lens_height_ * lens_height_) / lens_height_;
  const double cosb = f / range;
  const double sinb = l / range;
  return cosb * cosb / (radial * radial) + sinb * sinb / (range * range);
}

// What this anchor's next sighting is expected to scatter by, in m^2: the fast
// average carried one step along the trend the slow one reveals.
double GroundAnchorMap::predicted_variance(int64_t slot) const
{
  const double fast = std::max(variance_(slot), 1e-9);
  if (!(settings_.trend_power > 0.0)) {
    return fast;
  }
  const double slow = std::max(variance_slow_(slot), 1e-9);
  return fast * std::pow(fast / slow, settings_.trend_power);
}

double GroundAnchorMap::scatter_at(int64_t slot) const
{
  return slot >= 0 && slot < variance_.size() ? variance_(slot) : -1.0;
}

bool GroundAnchorMap::usable_at(int64_t slot) const
{
  if (slot < 0 || slot >= identifier_.size() || identifier_(slot) < 0) {
    return false;
  }
  if (observation_(slot) < settings_.trial_observations) {
    return false;
  }
  return !settings_.select_by_consistency || variance_(slot) <= settings_.max_variance;
}

double GroundAnchorMap::weight_at(int64_t slot) const
{
  if (settings_.weight_by_trend) {
    // No bearing. An anchor that sits where the geometry is poor scatters when
    // it is seen again, and that is already what this measures.
    const double sightings = static_cast<double>(
      std::max<int64_t>(
        std::min<int64_t>(observation_(slot), settings_.max_observations), 1));
    const double position = predicted_variance(slot) / (2.0 * sightings) +
      settings_.drift_variance_per_m * std::max(path_ - founded_path_(slot), 0.0);
    return 1.0 / std::max(position, 1e-12);
  }
  if (settings_.weight_by_variance) {
    // What this frame's sighting of the anchor is worth, in metres squared of
    // longitudinal uncertainty, plus what the anchor's own position is worth.
    const double geometric = std::max(longitudinal_information(slot), 1e-18);
    const double measured = settings_.bearing_variance / geometric;
    // `variance_` is the running mean of `dx^2 + dy^2` between a sighting and
    // the stored position -- the scatter of the sightings, not the variance of
    // the position they average to. The position is a mean, so its variance is
    // that over the number of sightings, and over two because the scatter is a
    // squared distance in the plane and this is per axis.
    //
    // `max_observations` is the right count to divide by and this is the first
    // use that earns it: the sightings are not independent (each is written in
    // the world frame the estimate had just settled on), so the window has to
    // be finite, which is exactly what that setting bounds.
    const double sightings = static_cast<double>(
      std::max<int64_t>(
        std::min<int64_t>(observation_(slot), settings_.max_observations), 1));
    const double position = std::max(variance_(slot), 0.0) / (2.0 * sightings) +
      settings_.drift_variance_per_m * std::max(path_ - founded_path_(slot), 0.0);
    return 1.0 / std::max(measured + position, 1e-12);
  }
  double count = settings_.weight_by_information
    ? information_(slot)
    : static_cast<double>(
    std::min<int64_t>(observation_(slot), settings_.max_observations));
  if (settings_.geometry_power > 0.0) {
    count *= std::pow(
      std::max(longitudinal_information(slot), 1e-18), settings_.geometry_power);
  }
  // What the anchor has stopped being sure of since it was written.
  //
  // An anchor is a world position, and the pose that put it there has moved on
  // since. The distance travelled since it was founded is what that costs: the
  // measured disagreement between the two cameras over ground one of them drove
  // across earlier runs at 8% of range, and it did not shrink when the pose got
  // three times better, so it is drift and it accumulates with the path.
  //
  // The map's only answer to this today is a hard age cutoff, and that cutoff
  // has a *sharp* optimum -- 120 / 180 / 250 / 350 / 500 frames score 0.3925 /
  // 0.2369 / 0.1895 / 0.2248 / 0.2396. A sharp optimum on a hard gate is what a
  // soft one looks like from the outside, which is the third time in this
  // estimator that has been true.
  if (settings_.drift_variance_per_m > 0.0) {
    const double travelled = std::max(path_ - founded_path_(slot), 0.0);
    count /= 1.0 + settings_.drift_variance_per_m * travelled;
  }
  if (!settings_.select_by_consistency) {
    return count;
  }
  return count / std::max(variance_(slot), 1e-4);
}

void GroundAnchorMap::polar(
  double x, double y, double yaw, std::vector<std::array<double, 7>> & out) const
{
  out.clear();
  out.reserve(static_cast<size_t>(live_));
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
    if (identifier_(slot) < 0) {
      continue;
    }
    const double dx = position_(slot, 0) - x;
    const double dy = position_(slot, 1) - y;
    const double f = c * dx + s * dy;
    const double l = -s * dx + c * dy;
    out.push_back(
      {std::hypot(f, l), std::atan2(l, f), weight_at(slot),
        static_cast<double>(observation_(slot)),
        static_cast<double>(identifier_(slot)),
        variance_(slot), usable_at(slot) ? 1.0 : 0.0});
  }
}

void GroundAnchorMap::extent(
  double x, double y, double yaw, double band, int & within, double & along,
  double & across) const
{
  within = 0;
  double sa = 0.0;
  double sc = 0.0;
  double sa2 = 0.0;
  double sc2 = 0.0;
  double n = 0.0;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
    if (identifier_(slot) < 0) {
      continue;
    }
    const double dx = position_(slot, 0) - x;
    const double dy = position_(slot, 1) - y;
    const double f = c * dx + s * dy;
    const double l = -s * dx + c * dy;
    if (std::hypot(dx, dy) <= band) {
      ++within;
    }
    sa += f; sc += l; sa2 += f * f; sc2 += l * l; n += 1.0;
  }
  if (n > 1.0) {
    // Mean offset first, spread second: a map that is a tight cluster far
    // behind and a map that is a window around the vehicle look the same in
    // the spread alone.
    along = sa / n;
    across = std::sqrt(std::max(sa2 / n - along * along, 0.0));
  }
}

void GroundAnchorMap::anchored(int source, const Identities & ids, Mask & out) const
{
  out.resize(ids.size());
  for (Eigen::Index i = 0; i < ids.size(); ++i) {
    const int64_t slot = slot_of(source, ids(i));
    // Maturity is only asked once the map is full. Before that it holds nothing
    // to be choosy with, and refusing its young anchors deadlocks the start:
    // `warming_up = !map_ready_ && !aligned_from_map` means the estimator
    // accepts no update until the map path has answered once, so a gate applied
    // from frame zero gives 899 solves and **zero** estimates.
    // Full means the map has ever reached capacity, not that no slot is
    // free right now: with eviction on, slots are freed and refilled every
    // update, so `free_.empty()` is false at the moment this is asked and
    // the gate never fires.
    const bool full = saturated_;
    out(i) = slot >= 0 &&
      (!full || settings_.anchored_min_observations <= 0 ||
      observation_(slot) >= settings_.anchored_min_observations);
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
  const Weights & information, const Points2 & body_points, int32_t pose_index,
  const Weights & clarity)
{
  update(source, ids, world_points, allow_new, information, clarity);
  const Eigen::Index count = ids.size();
  if (body_points.rows() != count) {
    return;
  }
  const bool weighted = information.size() == count;
  for (Eigen::Index i = 0; i < count; ++i) {
    const int64_t slot = slot_of(source, ids(i));
    if (slot < 0 || slot >= sight_pose_.rows()) {
      continue;
    }
    const int32_t at = sight_count_(slot) % kRemember;
    sight_pose_(slot, at) = pose_index;
    sight_body_(slot, 2 * at) = static_cast<float>(body_points(i, 0));
    sight_body_(slot, 2 * at + 1) = static_cast<float>(body_points(i, 1));
    sight_weight_(slot, at) = static_cast<float>(weighted ? information(i) : 1.0);
    sight_source_(slot, at) = source;
    const float strength = static_cast<float>(weighted ? information(i) : 1.0);
    if (best_pose_(slot, source) < 0 || strength > best_weight_(slot, source)) {
      best_pose_(slot, source) = pose_index;
      best_body_(slot, 2 * source) = static_cast<float>(body_points(i, 0));
      best_body_(slot, 2 * source + 1) = static_cast<float>(body_points(i, 1));
      best_weight_(slot, source) = strength;
    }
    sight_count_(slot) = sight_count_(slot) + 1;
    ++remembered_;
  }
}

void GroundAnchorMap::rebuild(const std::vector<std::array<double, 3>> & poses)
{
  for (Eigen::Index slot = 0; slot < position_.rows(); ++slot) {
    if (identifier_(slot) < 0 || sight_count_(slot) <= 0) {
      continue;
    }
    const int32_t held = std::min<int32_t>(sight_count_(slot), kRemember);
    double wx = 0.0;
    double wy = 0.0;
    double total = 0.0;
    for (int32_t k = 0; k < held; ++k) {
      const int32_t index = sight_pose_(slot, k);
      if (index < 0 || static_cast<size_t>(index) >= poses.size()) {
        continue;
      }
      const auto & p = poses[static_cast<size_t>(index)];
      const double c = std::cos(p[2]);
      const double s = std::sin(p[2]);
      const double bx = sight_body_(slot, 2 * k);
      const double by = sight_body_(slot, 2 * k + 1);
      const double w = std::max<double>(sight_weight_(slot, k), 1e-12);
      wx += w * (p[0] + c * bx - s * by);
      wy += w * (p[1] + s * bx + c * by);
      total += w;
    }
    if (total > 0.0) {
      const double nx = wx / total;
      const double ny = wy / total;
      rebuild_shift_ += std::hypot(nx - position_(slot, 0), ny - position_(slot, 1));
      ++rebuild_slots_;
      if (!settings_.rebuild_measure_only) {
        position_(slot, 0) = nx;
        position_(slot, 1) = ny;
      }
    }
  }
}

void GroundAnchorMap::update(
  int source, const Identities & ids, const Points2 & world_points, bool allow_new,
  const Weights & information, const Weights & clarity)
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
          // Earn the slot first. See `found_after_observations`.
          if (settings_.found_after_observations > 1) {
            auto & table = pending_[static_cast<size_t>(source)];
            auto held = table.find(ids(i));
            int32_t run = 1;
            if (held != table.end()) {
              run = held->second.second + 1 == frame_ ? held->second.first + 1 : 1;
            }
            if (run < settings_.found_after_observations) {
              table[ids(i)] = {run, frame_};
              continue;
            }
            table.erase(ids(i));
          }
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
    // The slow companion. Its gain is fixed rather than shared with the fast
    // one, because what the ratio has to measure is the change over a longer
    // window than the fast average can remember.
    if (settings_.weight_by_trend) {
      const double slow = std::min(settings_.trend_gain, gain);
      variance_slow_(slot) += slow * (residual - variance_slow_(slot));
    }
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
    std::vector<std::pair<double, int64_t>> ranked;
    ranked.reserve(static_cast<size_t>(live_));
    for (Eigen::Index slot = 0; slot < identifier_.size(); ++slot) {
      if (identifier_(slot) >= 0 &&
        frame_ - seen_(slot) >= std::max<int64_t>(settings_.evict_unseen_solves, 1))
      {
        double rank = static_cast<double>(seen_(slot));
        if (settings_.evict_by_information) {
          rank = weight_at(slot) * longitudinal_information(slot);
        } else if (settings_.evict_by_weight) {
          rank = weight_at(slot);
        }
        ranked.emplace_back(rank, slot);
      }
    }
    const size_t give = std::min(needed, ranked.size());
    if (give > 0) {
      std::nth_element(
        ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(give), ranked.end(),
        [](const auto & a, const auto & b) {return a.first < b.first;});
      for (size_t n = 0; n < give; ++n) {
        forget(ranked[n].second);
      }
    }
  }
  if (live_ >= settings_.max_anchors) {
    saturated_ = true;
  }
  // Clearest first, if asked and if the tracker gave us the means.
  if (settings_.admit_by_clarity && clarity.size() == count && fresh.size() > 1) {
    std::stable_sort(
      fresh.begin(), fresh.end(),
      [&](Eigen::Index a, Eigen::Index b) {return clarity(a) > clarity(b);});
  }
  // Best first, if asked. Sorting before the identity partition so that the
  // partition still wins where both are on.
  if (settings_.admit_by_information && fresh.size() > 1) {
    std::stable_sort(
      fresh.begin(), fresh.end(),
      [&](Eigen::Index a, Eigen::Index b) {
        return longitudinal_information_at(world_points(a, 0), world_points(a, 1)) >
        longitudinal_information_at(world_points(b, 0), world_points(b, 1));
      });
  }
  // Priority candidates first, stably, so the rest keep the order the frame
  // gave them. See `priority_identity_floor`.
  if (settings_.priority_identity_floor > 0 && fresh.size() > 1) {
    std::stable_partition(
      fresh.begin(), fresh.end(),
      [&](Eigen::Index i) {return ids(i) >= settings_.priority_identity_floor;});
  }
  size_t room = std::min(fresh.size(), free_.size());
  if (settings_.admit_per_update > 0) {
    room = std::min(room, static_cast<size_t>(settings_.admit_per_update));
  }
  if (room > 0) {
    for (size_t n = 0; n < room; ++n) {
      const Eigen::Index i = fresh[n];
      // Refuse a birth into a cell that already holds its share. Nothing is
      // removed to make space; the ground here is already described.
      if (settings_.density_quota > 0) {
        const int64_t cell = density_cell_of(world_points(i, 0), world_points(i, 1));
        const auto held = density_.find(cell);
        if (held != density_.end() && held->second >= settings_.density_quota) {
          // The cell is full. Refusing here keeps whatever it already has,
          // which after a few seconds of driving is ground the vehicle has
          // passed -- the least informative place an anchor can be. Weigh the
          // candidate against the cell's poorest instead.
          bool admitted = false;
          if (settings_.density_replace_margin > 0.0) {
            const auto list = density_slots_.find(cell);
            if (list != density_slots_.end() && !list->second.empty()) {
              int64_t worst = -1;
              double worst_value = std::numeric_limits<double>::infinity();
              for (const int64_t held_slot : list->second) {
                if (identifier_(held_slot) < 0) {
                  continue;
                }
                const double value = longitudinal_information(held_slot);
                if (value < worst_value) {
                  worst_value = value;
                  worst = held_slot;
                }
              }
              const double candidate =
                longitudinal_information_at(world_points(i, 0), world_points(i, 1));
              if (worst >= 0 &&
                candidate > settings_.density_replace_margin * worst_value)
              {
                forget(worst);
                free_.push_back(worst);
                --live_;
                admitted = true;
              }
            }
          }
          if (!admitted) {
            continue;
          }
        }
      }
      int polar_cell = -1;
      if (settings_.polar_quota > 0 && !polar_.empty()) {
        polar_cell = polar_cell_of(world_points(i, 0), world_points(i, 1));
        if (polar_cell >= 0 && polar_cell < static_cast<int>(polar_.size()) &&
          polar_[static_cast<size_t>(polar_cell)] >= settings_.polar_quota)
        {
          continue;
        }
      }
      const int64_t slot = free_.back();
      free_.pop_back();
      position_(slot, 0) = world_points(i, 0);
      position_(slot, 1) = world_points(i, 1);
      observation_(slot) = 1;
      // What the sighting that founded it was worth, not a chosen number. The
      // anchor's position is that one measurement, so its variance is that
      // measurement's -- `sigma_b^2 / I` at the geometry it was seen from,
      // doubled back into the squared-distance convention `variance_` keeps.
      // An anchor born at the far edge of the band therefore starts believed
      // to a fraction of one born close, which is the difference the fixed
      // `initial_variance` could not express.
      variance_slow_(slot) = settings_.initial_variance;
      variance_(slot) = settings_.weight_by_variance
        ? 2.0 * settings_.bearing_variance /
        std::max(
          longitudinal_information_at(world_points(i, 0), world_points(i, 1)), 1e-18)
        : settings_.initial_variance;
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
      const int64_t born_cell = density_cell_of(position_(slot, 0), position_(slot, 1));
      ++density_[born_cell];
      if (settings_.density_replace_margin > 0.0) {
        density_slots_[born_cell].push_back(slot);
      }
      if (polar_cell >= 0 && polar_cell < static_cast<int>(polar_.size())) {
        ++polar_[static_cast<size_t>(polar_cell)];
      }
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
  if (settings_.density_quota > 0) {
    const auto held = density_.find(
      density_cell_of(position_(slot, 0), position_(slot, 1)));
    if (held != density_.end() && --held->second <= 0) {
      density_.erase(held);
    }
    if (settings_.density_replace_margin > 0.0) {
      const auto list = density_slots_.find(
        density_cell_of(position_(slot, 0), position_(slot, 1)));
      if (list != density_slots_.end()) {
        auto & slots = list->second;
        slots.erase(std::remove(slots.begin(), slots.end(), slot), slots.end());
        if (slots.empty()) {
          density_slots_.erase(list);
        }
      }
    }
  }
  founder_(slot) = -1;
  identifier_(slot) = -1;
  observation_(slot) = 0;
  // The remembered sightings belong to the anchor that just died, not to
  // whatever is put in this slot next. Leaving them makes a rebuild place the
  // new anchor where the old one was, which measured as eighteen metres.
  if (sight_count_.size() > slot) {
    sight_count_(slot) = 0;
    sight_pose_.row(slot).setConstant(-1);
    sight_source_.row(slot).setConstant(-1);
    best_pose_.row(slot).setConstant(-1);
    best_weight_.row(slot).setZero();
  }
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
    // Behind and receding: it cannot be observed again, and it measures nothing
    // about motion along the heading.
    bool astern = false;
    if (settings_.forget_beyond_bearing_deg > 0.0) {
      const double c = std::cos(yaw_);
      const double s = std::sin(yaw_);
      const double dx = position_(slot, 0) - at_x_;
      const double dy = position_(slot, 1) - at_y_;
      const double f = c * dx + s * dy;
      const double l = -s * dx + c * dy;
      const double reach = std::hypot(f, l);
      astern = reach > std::max(settings_.forget_beyond_range_m, 1e-6) &&
        std::abs(std::atan2(l, f)) >
        settings_.forget_beyond_bearing_deg * M_PI / 180.0;
    }
    // Given a few sightings to settle, an anchor that still scatters is not a
    // landmark and should not be registered against.
    bool scattered = settings_.select_by_consistency &&
      observation_(slot) >= settings_.trial_observations &&
      variance_(slot) > settings_.max_variance;
    // The weight-threshold eviction. An anchor whose predicted scatter has
    // passed this is not a landmark, and waiting for it to go stale or for the
    // map to overflow is waiting for the wrong thing.
    if (settings_.trend_evict_variance > 0.0 && observation_(slot) >= 2 &&
      predicted_variance(slot) > settings_.trend_evict_variance)
    {
      scattered = true;
    }
    if (!stale && !scattered && !astern) {
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
  const Eigen::Vector2d & origin, double radial_min_range, double lens_height,
  double softness,
  const Eigen::Vector2d * translation_prior, int restarts, double ambiguity,
  const Eigen::Vector2d * inertial_hop, double inertial_gate,
  const Weights & residual_scale, bool bearing_nonholonomic)
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
  // The bearing-domain normal equations, accumulated in the same pass. Five
  // unknowns: three of rotation and the two in-plane translations, which are
  // carried so that whatever the ground fit left behind lands there instead of
  // leaking into the angles.
  Eigen::Matrix<double, 5, 5> bearing_normal = Eigen::Matrix<double, 5, 5>::Zero();
  Eigen::Matrix<double, 5, 1> bearing_rhs = Eigen::Matrix<double, 5, 1>::Zero();
  int bearing_terms = 0;
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

    if (lens_height > 1e-6) {
      const double rho = std::hypot(range, lens_height);
      const Eigen::Vector3d bearing(bx / rho, by / rho, -lens_height / rho);
      // Only the part of a ground residual that lies across the line of sight
      // moves a bearing; the part along it is a range error and is invisible
      // here, which is exactly the separation being used.
      const Eigen::Vector3d ground(body_rx, body_ry, 0.0);
      const Eigen::Vector3d residual =
        (ground - bearing * bearing.dot(ground)) / rho;
      Eigen::Matrix<double, 3, 5> basis;
      for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        axis(k) = 1.0;
        basis.col(k) = -axis.cross(bearing);
      }
      for (int k = 0; k < 2; ++k) {
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        axis(k) = 1.0;
        basis.col(3 + k) = -(axis - bearing * bearing.dot(axis)) / rho;
      }
      if (bearing_nonholonomic) {
        // A yaw about the axle carries the lens sideways by its own lever, so
        // the two columns become one; the sideways column then has nothing left
        // to describe and is held at zero rather than fitted.
        basis.col(2) += origin.x() * basis.col(4);
        basis.col(4).setZero();
      }
      bearing_normal += w * basis.transpose() * basis;
      bearing_rhs += w * basis.transpose() * residual;
      ++bearing_terms;
    }
  }
  if (kept < min_inliers) {
    return std::nullopt;
  }
  result.translation = centre;
  result.spread = std::sqrt(squared / static_cast<double>(kept));
  result.yaw = yaw;
  result.yaw_sigma = applied;

  // The angles, from the sphere. Fitted before the ground-domain regression
  // below because it is the one that is entitled to them: that one reports a
  // pitch too, but off a pair of bases that are 98-99 per cent collinear.
  if (bearing_terms >= 8) {
    if (bearing_nonholonomic) {
      // The zeroed column leaves a singular direction; pin it so the solve is
      // of the four that remain.
      bearing_normal(4, 4) = 1.0;
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 5, 5>> solver(bearing_normal);
    const double smallest = solver.eigenvalues()(0);
    const double largest = solver.eigenvalues()(4);
    // A fit standing on a direction the anchors do not span is not a
    // measurement of anything. The population turns over constantly, so this
    // fires on its own where the geometry is thin rather than needing a count.
    if (smallest > 1e-12 && largest / smallest < 1e6) {
      const Eigen::Matrix<double, 5, 1> fit = bearing_normal.ldlt().solve(bearing_rhs);
      if (fit.allFinite()) {
        // Negated on the way out. The bases above are `-w x b`, so the fit
        // solves for the attitude the assumption is wrong BY; what a caller
        // wants is the correction to apply, which is its opposite. Confirmed
        // against the geometry rather than assumed: applying -0.0163 deg moves
        // the front camera's hop bias by +0.054% where a direct numerical
        // projection of that angle predicts +0.042%.
        result.bearing_roll = -fit(0);
        result.bearing_pitch = -fit(1);
        result.bearing_yaw = -fit(2);
        result.bearing_terms = bearing_terms;
        result.bearing_tx = -fit(3);
        result.bearing_ty = -fit(4);
      }
    }
  }

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
    // And again with both bases, so a pitch is not read as a height.
    if (lens_height > 1e-6) {
      double a11 = 0.0;
      double a12 = 0.0;
      double a22 = 0.0;
      double c1 = 0.0;
      double c2 = 0.0;
      for (size_t i = 0; i < ranges.size(); ++i) {
        const double w = radial_weight[i];
        const double u = ranges[i] / reference;
        const double v = (ranges[i] * ranges[i] + lens_height * lens_height) /
          (lens_height * reference);
        a11 += w * u * u;
        a12 += w * u * v;
        a22 += w * v * v;
        c1 += w * u * radial[i];
        c2 += w * v * radial[i];
      }
      const double det = a11 * a22 - a12 * a12;
      if (std::abs(det) > 1e-12) {
        result.radial_height = (c1 * a22 - c2 * a12) / det;
        result.radial_pitch = (a11 * c2 - a12 * c1) / det;
      }
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


double GroundAnchorMap::sighting_span() const
{
  double total = 0.0;
  int64_t n = 0;
  for (Eigen::Index slot = 0; slot < sight_count_.size(); ++slot) {
    if (identifier_(slot) < 0 || sight_count_(slot) < 2) {
      continue;
    }
    const int32_t held = std::min<int32_t>(sight_count_(slot), kRemember);
    int32_t low = std::numeric_limits<int32_t>::max();
    int32_t high = -1;
    for (int32_t k = 0; k < held; ++k) {
      const int32_t index = sight_pose_(slot, k);
      if (index < 0) {continue;}
      low = std::min(low, index);
      high = std::max(high, index);
    }
    if (high >= low) {total += high - low; ++n;}
  }
  return n > 0 ? total / static_cast<double>(n) : 0.0;
}

std::vector<GroundAnchorMap::Revisit> GroundAnchorMap::revisits() const
{
  std::vector<Revisit> out;
  if (sources_ < 2 || sight_count_.size() == 0) {
    return out;
  }
  for (Eigen::Index slot = 0; slot < sight_count_.size(); ++slot) {
    if (identifier_(slot) < 0 || sight_count_(slot) < 2) {
      continue;
    }
    // Two sources have bound this slot only if the link found the same ground
    // twice. `owner_` records who; the sightings themselves do not carry a
    // source, so the pair is taken as the earliest and latest remembered
    // sighting, which is the widest baseline the slot holds.
    int bound = 0;
    for (int source = 0; source < sources_; ++source) {
      bound += owner_(slot, source) >= 0 ? 1 : 0;
    }
    if (bound < 2) {
      continue;
    }
    // The two cameras' strongest sightings of this slot, ordered in time.
    int32_t first_source = -1;
    int32_t second_source = -1;
    for (int source = 0; source < sources_; ++source) {
      if (best_pose_(slot, source) < 0) {continue;}
      if (first_source < 0 || best_pose_(slot, source) < best_pose_(slot, first_source)) {
        second_source = first_source;
        first_source = source;
      } else if (second_source < 0 || best_pose_(slot, source) > best_pose_(slot, second_source)) {
        second_source = source;
      }
    }
    if (first_source < 0 || second_source < 0 ||
      best_pose_(slot, first_source) >= best_pose_(slot, second_source))
    {
      continue;
    }
    Revisit r;
    r.from = best_pose_(slot, first_source);
    r.to = best_pose_(slot, second_source);
    r.bx_from = best_body_(slot, 2 * first_source);
    r.by_from = best_body_(slot, 2 * first_source + 1);
    r.bx_to = best_body_(slot, 2 * second_source);
    r.by_to = best_body_(slot, 2 * second_source + 1);
    r.weight = std::min<double>(best_weight_(slot, first_source),
      best_weight_(slot, second_source));
    out.push_back(r);
  }
  return out;
}

}  // namespace monoscale
