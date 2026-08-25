// The log-odds grid, from
// src/monoscale_occupancy_grid_map/test/test_occupancy.py.

#include <cmath>

#include <gtest/gtest.h>

#include "monoscale_core/occupancy.hpp"

using monoscale::GridSettings;
using monoscale::LogOddsGrid;

namespace
{

GridSettings small(double free_update, double occupied_update)
{
  GridSettings settings;
  settings.resolution = 1.0;
  settings.origin_x = 0.0;
  settings.origin_y = 0.0;
  settings.tile_size_cells = 4;
  settings.free_update = free_update;
  settings.occupied_update = occupied_update;
  settings.inflation_radius_m = 0.0;
  return settings;
}

// Read a published window by world position, which is the only frame both the
// caller and the tiles agree on once the extent moves with the car.
int8_t at(
  const GridSettings & settings, const LogOddsGrid::Window & window,
  const std::vector<int8_t> & values, double x, double y)
{
  const int column =
    static_cast<int>(std::floor((x - settings.origin_x) / settings.resolution)) - window.cell_x;
  const int row =
    static_cast<int>(std::floor((y - settings.origin_y) / settings.resolution)) - window.cell_y;
  if (column < 0 || row < 0 || column >= window.width || row >= window.height) {
    return -1;
  }
  return values[static_cast<size_t>(row) * static_cast<size_t>(window.width) +
    static_cast<size_t>(column)];
}

}  // namespace

TEST(Occupancy, RayMarksFreeSpaceAndOccupiedEndpoint)
{
  const GridSettings settings = small(1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_ray(1.5, 1.5, 4.5, 1.5, true);
  const auto window = grid.window(1.5, 1.5);
  const auto values = grid.message_values(window);

  EXPECT_EQ(at(settings, window, values, 1.5, 1.5), 0);
  EXPECT_EQ(at(settings, window, values, 2.5, 1.5), 0);
  EXPECT_EQ(at(settings, window, values, 3.5, 1.5), 0);
  EXPECT_EQ(at(settings, window, values, 4.5, 1.5), 100);
}

TEST(Occupancy, OccupiedPointDoesNotCarveFreeSpaceOnTheWayIn)
{
  const GridSettings settings = small(1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_ray(1.5, 1.5, 4.5, 1.5, false);
  grid.integrate_point(2.5, 1.5);
  grid.integrate_point(2.5, 1.5);
  const auto window = grid.window(1.5, 1.5);
  const auto values = grid.message_values(window);

  EXPECT_EQ(at(settings, window, values, 2.5, 1.5), 100);
  // The cells the free ray swept stay free; only the endpoint cell changed.
  EXPECT_EQ(at(settings, window, values, 1.5, 1.5), 0);
  EXPECT_EQ(at(settings, window, values, 3.5, 1.5), 0);
}

TEST(Occupancy, GroundFarFromTheStartIsKeptRatherThanDropped)
{
  // What this replaced was a 60 m box anchored where the run began. Half the
  // points of a road drive landed outside it and were discarded in silence.
  const GridSettings settings = small(1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_point(500.5, -400.5);
  const auto window = grid.window(500.5, -400.5);

  ASSERT_GT(window.width, 0);
  EXPECT_EQ(at(settings, window, grid.message_values(window), 500.5, -400.5), 100);
}

TEST(Occupancy, ThePublishedWindowIsCroppedToWhatHasBeenSeen)
{
  const GridSettings settings = small(1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_point(1.5, 1.5);
  const auto one = grid.window(1.5, 1.5);
  EXPECT_EQ(one.width, settings.tile_size_cells);

  // A second tile's worth of ground widens it; the extent is the map's, not a
  // constant.
  grid.integrate_point(9.5, 1.5);
  const auto two = grid.window(1.5, 1.5);
  EXPECT_GT(two.width, one.width);
}

TEST(Occupancy, TheWindowFollowsTheCarAndLeavesTheRestBehind)
{
  const GridSettings settings = small(1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_point(1.5, 1.5);
  grid.integrate_point(500.5, 500.5);

  const auto near_start = grid.window(1.5, 1.5);
  const auto values = grid.message_values(near_start);
  EXPECT_EQ(at(settings, near_start, values, 1.5, 1.5), 100);
  EXPECT_EQ(at(settings, near_start, values, 500.5, 500.5), -1);
}

TEST(Occupancy, DecayGivesEvidenceBackAndTheEmptiedTileIsCollected)
{
  GridSettings settings = small(1.0, 1.0);
  settings.decay_log_odds_per_sec = 0.5;
  settings.collect_min_age_sec = 0.0;
  LogOddsGrid grid(settings);

  grid.advance(0.0);
  grid.integrate_point(1.5, 1.5);
  grid.collect();
  EXPECT_EQ(grid.tile_count(), 1u);

  // Ten seconds at half a unit a second is more than the single vote held.
  grid.advance(10.0);
  grid.collect();
  EXPECT_EQ(grid.tile_count(), 0u);
}

TEST(Occupancy, ATileStillCarryingEvidenceSurvivesCollection)
{
  GridSettings settings = small(1.0, 1.0);
  settings.collect_min_age_sec = 0.0;
  LogOddsGrid grid(settings);

  grid.advance(0.0);
  grid.integrate_point(1.5, 1.5);
  grid.advance(1000.0);
  grid.collect();

  EXPECT_EQ(grid.tile_count(), 1u);
}

TEST(Occupancy, RaisingTheThresholdDemandsASecondObservation)
{
  GridSettings strict_settings = small(0.45, 0.85);
  strict_settings.occupied_probability = 0.8;
  GridSettings lenient_settings = small(0.45, 0.85);
  lenient_settings.occupied_probability = 0.65;

  LogOddsGrid strict(strict_settings);
  LogOddsGrid lenient(lenient_settings);
  strict.integrate_point(1.5, 1.5);
  lenient.integrate_point(1.5, 1.5);

  const auto lenient_window = lenient.window(1.5, 1.5);
  EXPECT_EQ(
    at(lenient_settings, lenient_window, lenient.message_values(lenient_window), 1.5, 1.5), 100);
  const auto strict_window = strict.window(1.5, 1.5);
  EXPECT_NE(
    at(strict_settings, strict_window, strict.message_values(strict_window), 1.5, 1.5), 100);

  strict.integrate_point(1.5, 1.5);
  const auto again = strict.window(1.5, 1.5);
  EXPECT_EQ(at(strict_settings, again, strict.message_values(again), 1.5, 1.5), 100);
}

TEST(Occupancy, InflationSpreadsAnObstacleWithoutInventingOne)
{
  GridSettings settings = small(1.0, 1.0);
  settings.tile_size_cells = 20;
  settings.inflation_radius_m = 2.0;
  LogOddsGrid grid(settings);

  grid.integrate_point(10.5, 10.5);
  const auto window = grid.window(10.5, 10.5);
  const auto values = grid.message_values(window);

  EXPECT_EQ(at(settings, window, values, 10.5, 10.5), 100);
  // Two cells out is inside the radius; five is not.
  EXPECT_EQ(at(settings, window, values, 12.5, 10.5), 100);
  EXPECT_NE(at(settings, window, values, 15.5, 10.5), 100);
}
