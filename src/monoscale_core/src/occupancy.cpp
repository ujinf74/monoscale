#include "monoscale_core/occupancy.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace monoscale
{

LogOddsGrid::LogOddsGrid(const GridSettings & settings)
: settings_(settings),
  log_odds_(static_cast<size_t>(settings.width) * settings.height, 0.0f),
  observed_(static_cast<size_t>(settings.width) * settings.height, 0)
{
}

void LogOddsGrid::mark_free(int x, int y)
{
  if (!inside(x, y)) {
    return;
  }
  const size_t at = static_cast<size_t>(y) * settings_.width + x;
  log_odds_[at] = std::max(
    static_cast<float>(settings_.floor),
    log_odds_[at] - static_cast<float>(settings_.free_update));
  observed_[at] = 1;
}

void LogOddsGrid::mark_occupied(int x, int y)
{
  if (!inside(x, y)) {
    return;
  }
  const size_t at = static_cast<size_t>(y) * settings_.width + x;
  log_odds_[at] = std::min(
    4.0f, log_odds_[at] + static_cast<float>(settings_.occupied_update));
  observed_[at] = 1;
}

void LogOddsGrid::integrate_point(double x, double y)
{
  mark_occupied(
    static_cast<int>(std::floor((x - settings_.origin_x) / settings_.resolution)),
    static_cast<int>(std::floor((y - settings_.origin_y) / settings_.resolution)));
}

void LogOddsGrid::integrate_ray(
  double from_x, double from_y, double to_x, double to_y, bool occupied)
{
  int x0 = static_cast<int>(std::floor((from_x - settings_.origin_x) / settings_.resolution));
  int y0 = static_cast<int>(std::floor((from_y - settings_.origin_y) / settings_.resolution));
  const int x1 = static_cast<int>(std::floor((to_x - settings_.origin_x) / settings_.resolution));
  const int y1 = static_cast<int>(std::floor((to_y - settings_.origin_y) / settings_.resolution));

  const int dx = std::abs(x1 - x0);
  const int dy = -std::abs(y1 - y0);
  const int sx = x0 < x1 ? 1 : -1;
  const int sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;

  while (true) {
    const bool last = x0 == x1 && y0 == y1;
    if (last && occupied) {
      mark_occupied(x0, y0);
      return;
    }
    mark_free(x0, y0);
    if (last) {
      return;
    }
    const int twice = 2 * error;
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

std::vector<int8_t> LogOddsGrid::message_values() const
{
  const size_t cells = log_odds_.size();
  std::vector<int8_t> values(cells, -1);
  cv::Mat occupied(settings_.height, settings_.width, CV_8UC1, cv::Scalar(0));
  bool any_occupied = false;

  for (size_t at = 0; at < cells; ++at) {
    if (!observed_[at]) {
      continue;
    }
    const double probability = 1.0 / (1.0 + std::exp(-log_odds_[at]));
    if (probability < settings_.free_probability) {
      values[at] = 0;
    } else if (probability <= settings_.occupied_probability) {
      values[at] = 50;
    } else {
      occupied.data[at] = 1;
      any_occupied = true;
    }
  }

  const int radius = static_cast<int>(
    std::ceil(std::max(settings_.inflation_radius_m, 0.0) / settings_.resolution));
  if (radius > 0 && any_occupied) {
    const int size = 2 * radius + 1;
    const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
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
