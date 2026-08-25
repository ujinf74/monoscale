#include "monoscale_core/occupancy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace monoscale
{

LogOddsGrid::LogOddsGrid(const GridSettings & settings)
: settings_(settings)
{
  settings_.tile_size_cells = std::max(4, settings_.tile_size_cells);
  settings_.roi_tiles_x = std::max(1, settings_.roi_tiles_x);
  settings_.roi_tiles_y = std::max(1, settings_.roi_tiles_y);
}

int32_t LogOddsGrid::floor_div(const int32_t value, const int32_t by)
{
  const int32_t quotient = value / by;
  return (value % by != 0 && ((value < 0) != (by < 0))) ? quotient - 1 : quotient;
}

int32_t LogOddsGrid::cell_of(const double world, const double origin) const
{
  return static_cast<int32_t>(std::floor((world - origin) / settings_.resolution));
}

LogOddsGrid::Tile & LogOddsGrid::tile_for(
  const int32_t cell_x, const int32_t cell_y, size_t & at)
{
  const int32_t size = settings_.tile_size_cells;
  const int32_t tx = floor_div(cell_x, size);
  const int32_t ty = floor_div(cell_y, size);
  at = static_cast<size_t>(cell_y - ty * size) * static_cast<size_t>(size) +
    static_cast<size_t>(cell_x - tx * size);

  auto found = tiles_.find(TileKey{tx, ty});
  if (found == tiles_.end()) {
    Tile fresh;
    fresh.log_odds.assign(static_cast<size_t>(size) * static_cast<size_t>(size), 0.0f);
    fresh.observed.assign(static_cast<size_t>(size) * static_cast<size_t>(size), 0);
    fresh.last_update = now_;
    found = tiles_.emplace(TileKey{tx, ty}, std::move(fresh)).first;
  }
  return found->second;
}

const LogOddsGrid::Tile * LogOddsGrid::tile_at(const TileKey & key) const
{
  const auto found = tiles_.find(key);
  return found == tiles_.end() ? nullptr : &found->second;
}

float LogOddsGrid::strongest(const Tile & tile) const
{
  float most = 0.0f;
  for (size_t at = 0; at < tile.log_odds.size(); ++at) {
    if (tile.observed[at]) {
      most = std::max(most, std::abs(tile.log_odds[at]));
    }
  }
  return most;
}

void LogOddsGrid::advance(const double now)
{
  const double was = now_;
  now_ = now;
  if (settings_.decay_log_odds_per_sec <= 0.0) {
    return;
  }
  const double step = settings_.decay_log_odds_per_sec * std::max(0.0, now - was);
  if (step <= 0.0) {
    return;
  }
  for (auto & entry : tiles_) {
    Tile & tile = entry.second;
    for (size_t at = 0; at < tile.log_odds.size(); ++at) {
      if (!tile.observed[at]) {
        continue;
      }
      float & value = tile.log_odds[at];
      value = value > 0.0f
        ? static_cast<float>(std::max(0.0, value - step))
        : static_cast<float>(std::min(0.0, value + step));
    }
  }
}

void LogOddsGrid::collect()
{
  for (auto it = tiles_.begin(); it != tiles_.end();) {
    const double age = now_ - it->second.last_update;
    const bool nothing_worth_keeping =
      strongest(it->second) < static_cast<float>(settings_.collect_min_abs_log_odds);
    const bool settled = age >= settings_.collect_min_age_sec;
    const bool forgotten = settings_.tile_forget_sec > 0.0 && age > settings_.tile_forget_sec;
    if (nothing_worth_keeping && (settled || forgotten)) {
      it = tiles_.erase(it);
    } else {
      ++it;
    }
  }
}

void LogOddsGrid::mark(const int32_t cell_x, const int32_t cell_y, const bool occupied)
{
  size_t at = 0;
  Tile & tile = tile_for(cell_x, cell_y, at);
  tile.last_update = now_;
  float & value = tile.log_odds[at];
  value = occupied
    ? std::min(4.0f, value + static_cast<float>(settings_.occupied_update))
    : std::max(static_cast<float>(settings_.floor), value - static_cast<float>(settings_.free_update));
  tile.observed[at] = 1;
}

void LogOddsGrid::integrate_point(const double x, const double y)
{
  mark(cell_of(x, settings_.origin_x), cell_of(y, settings_.origin_y), true);
}

void LogOddsGrid::integrate_ray(
  const double from_x, const double from_y, const double to_x, const double to_y,
  const bool occupied)
{
  int32_t x0 = cell_of(from_x, settings_.origin_x);
  int32_t y0 = cell_of(from_y, settings_.origin_y);
  const int32_t x1 = cell_of(to_x, settings_.origin_x);
  const int32_t y1 = cell_of(to_y, settings_.origin_y);

  const int32_t dx = std::abs(x1 - x0);
  const int32_t dy = -std::abs(y1 - y0);
  const int32_t sx = x0 < x1 ? 1 : -1;
  const int32_t sy = y0 < y1 ? 1 : -1;
  int32_t error = dx + dy;

  while (true) {
    const bool last = x0 == x1 && y0 == y1;
    if (last && occupied) {
      mark(x0, y0, true);
      return;
    }
    mark(x0, y0, false);
    if (last) {
      return;
    }
    const int32_t twice = 2 * error;
    if (twice >= dy) {
      error += dy;
      x0 += sx;
    }
    if (twice <= dx) {
      error += dx;
      y0 += sy;
    }
  }
}

LogOddsGrid::Window LogOddsGrid::window(const double x, const double y) const
{
  const int32_t size = settings_.tile_size_cells;
  const int32_t tx = floor_div(cell_of(x, settings_.origin_x), size);
  const int32_t ty = floor_div(cell_of(y, settings_.origin_y), size);
  const int32_t reach_x = settings_.roi_tiles_x / 2;
  const int32_t reach_y = settings_.roi_tiles_y / 2;

  // The reach is the ceiling; what is published is the box the tiles inside it
  // actually occupy, so an empty map starts small and grows with the drive.
  int32_t x0 = 0;
  int32_t y0 = 0;
  int32_t x1 = 0;
  int32_t y1 = 0;
  bool any = false;
  for (int32_t i = tx - reach_x; i <= tx - reach_x + settings_.roi_tiles_x - 1; ++i) {
    for (int32_t j = ty - reach_y; j <= ty - reach_y + settings_.roi_tiles_y - 1; ++j) {
      const Tile * tile = tile_at(TileKey{i, j});
      if (tile == nullptr ||
        strongest(*tile) < static_cast<float>(settings_.collect_min_abs_log_odds))
      {
        continue;
      }
      if (!any) {
        x0 = x1 = i;
        y0 = y1 = j;
        any = true;
      } else {
        x0 = std::min(x0, i);
        x1 = std::max(x1, i);
        y0 = std::min(y0, j);
        y1 = std::max(y1, j);
      }
    }
  }
  if (!any) {
    return Window{};
  }

  // Room for the dilation of what the box holds. Without it an obstacle seen at
  // the edge of the mapped area has its inflation cut off by the crop.
  const int32_t pad = static_cast<int32_t>(
    std::ceil(std::max(settings_.inflation_radius_m, 0.0) / settings_.resolution));

  Window window;
  window.cell_x = x0 * size - pad;
  window.cell_y = y0 * size - pad;
  window.width = static_cast<int>((x1 - x0 + 1) * size + 2 * pad);
  window.height = static_cast<int>((y1 - y0 + 1) * size + 2 * pad);
  window.origin_x = settings_.origin_x + window.cell_x * settings_.resolution;
  window.origin_y = settings_.origin_y + window.cell_y * settings_.resolution;
  return window;
}

std::vector<int8_t> LogOddsGrid::message_values(const Window & window) const
{
  if (window.width <= 0 || window.height <= 0) {
    return {};
  }
  const size_t cells = static_cast<size_t>(window.width) * static_cast<size_t>(window.height);
  std::vector<int8_t> values(cells, -1);
  cv::Mat occupied(window.height, window.width, CV_8UC1, cv::Scalar(0));
  bool any_occupied = false;

  const int32_t size = settings_.tile_size_cells;
  for (int row = 0; row < window.height; ++row) {
    const int32_t cell_y = window.cell_y + row;
    const int32_t ty = floor_div(cell_y, size);
    const size_t within_row = static_cast<size_t>(cell_y - ty * size) * static_cast<size_t>(size);
    for (int column = 0; column < window.width; ++column) {
      const int32_t cell_x = window.cell_x + column;
      const int32_t tx = floor_div(cell_x, size);
      const Tile * tile = tile_at(TileKey{tx, ty});
      if (tile == nullptr) {
        continue;
      }
      const size_t at = within_row + static_cast<size_t>(cell_x - tx * size);
      if (!tile->observed[at]) {
        continue;
      }
      const size_t out = static_cast<size_t>(row) * static_cast<size_t>(window.width) +
        static_cast<size_t>(column);
      const double probability = 1.0 / (1.0 + std::exp(-tile->log_odds[at]));
      if (probability < settings_.free_probability) {
        values[out] = 0;
      } else if (probability <= settings_.occupied_probability) {
        values[out] = 50;
      } else {
        occupied.data[out] = 1;
        any_occupied = true;
      }
    }
  }

  const int radius = static_cast<int>(
    std::ceil(std::max(settings_.inflation_radius_m, 0.0) / settings_.resolution));
  if (radius > 0 && any_occupied) {
    const int kernel_size = 2 * radius + 1;
    const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::dilate(occupied, occupied, kernel);
  }
  for (size_t at = 0; at < cells; ++at) {
    if (occupied.data[at] != 0) {
      values[at] = 100;
    }
  }
  return values;
}

}  // namespace monoscale
