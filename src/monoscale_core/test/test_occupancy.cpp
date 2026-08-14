// The log-odds grid, from
// src/monoscale_occupancy_grid_map/test/test_occupancy.py.

#include <gtest/gtest.h>

#include "monoscale_core/occupancy.hpp"

using monoscale::GridSettings;
using monoscale::LogOddsGrid;

namespace
{

GridSettings small(int side, double free_update, double occupied_update)
{
  GridSettings settings;
  settings.resolution = 1.0;
  settings.width = side;
  settings.height = side;
  settings.origin_x = 0.0;
  settings.origin_y = 0.0;
  settings.free_update = free_update;
  settings.occupied_update = occupied_update;
  settings.inflation_radius_m = 0.0;
  return settings;
}

int8_t at(const std::vector<int8_t> & values, const GridSettings & settings, int row, int column)
{
  return values[static_cast<size_t>(row) * settings.width + column];
}

}  // namespace

TEST(Occupancy, RayMarksFreeSpaceAndOccupiedEndpoint)
{
  const GridSettings settings = small(10, 1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_ray(1.5, 1.5, 4.5, 1.5, true);
  const auto values = grid.message_values();

  EXPECT_EQ(at(values, settings, 1, 1), 0);
  EXPECT_EQ(at(values, settings, 1, 2), 0);
  EXPECT_EQ(at(values, settings, 1, 3), 0);
  EXPECT_EQ(at(values, settings, 1, 4), 100);
}

TEST(Occupancy, OccupiedPointDoesNotCarveFreeSpaceOnTheWayIn)
{
  const GridSettings settings = small(10, 1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_ray(1.5, 1.5, 4.5, 1.5, false);
  grid.integrate_point(2.5, 1.5);
  grid.integrate_point(2.5, 1.5);
  const auto values = grid.message_values();

  EXPECT_EQ(at(values, settings, 1, 2), 100);
  // The cells the free ray swept stay free; only the endpoint cell changed.
  EXPECT_EQ(at(values, settings, 1, 1), 0);
  EXPECT_EQ(at(values, settings, 1, 3), 0);
}

TEST(Occupancy, PointOutsideTheGridIsIgnored)
{
  const GridSettings settings = small(4, 1.0, 1.0);
  LogOddsGrid grid(settings);

  grid.integrate_point(99.0, 99.0);

  for (int8_t value : grid.message_values()) {
    EXPECT_EQ(value, -1);
  }
}

TEST(Occupancy, RaisingTheThresholdDemandsASecondObservation)
{
  GridSettings strict_settings = small(4, 0.45, 0.85);
  strict_settings.occupied_probability = 0.8;
  GridSettings lenient_settings = small(4, 0.45, 0.85);
  lenient_settings.occupied_probability = 0.65;

  LogOddsGrid strict(strict_settings);
  LogOddsGrid lenient(lenient_settings);
  strict.integrate_point(1.5, 1.5);
  lenient.integrate_point(1.5, 1.5);

  EXPECT_EQ(at(lenient.message_values(), lenient_settings, 1, 1), 100);
  EXPECT_NE(at(strict.message_values(), strict_settings, 1, 1), 100);

  strict.integrate_point(1.5, 1.5);
  EXPECT_EQ(at(strict.message_values(), strict_settings, 1, 1), 100);
}

TEST(Occupancy, InflationSpreadsAnObstacleWithoutInventingOne)
{
  GridSettings settings = small(20, 1.0, 1.0);
  settings.inflation_radius_m = 2.0;
  LogOddsGrid grid(settings);

  grid.integrate_point(10.5, 10.5);
  const auto values = grid.message_values();

  EXPECT_EQ(at(values, settings, 10, 10), 100);
  // Two cells out is inside the radius; five is not.
  EXPECT_EQ(at(values, settings, 10, 12), 100);
  EXPECT_NE(at(values, settings, 10, 15), 100);
}
