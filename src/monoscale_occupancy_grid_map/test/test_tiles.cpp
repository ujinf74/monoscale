// The tile storage under the sweep's grid.
//
// The dense accumulate and publish chain was measured and is unchanged; what
// is new is that it now runs against a window copied out of tiles and copied
// back. So what these tests cover is exactly that seam -- the round trip, the
// cell arithmetic that places a window in the world, and the legacy extent
// `sweep_offline` depends on -- rather than the sweep arithmetic itself.

#include <cmath>

#include <gtest/gtest.h>

#include "monoscale_occupancy_grid_map/sweep.hpp"

using monoscale_occupancy::CameraGrid;
using monoscale_occupancy::GridView;
using monoscale_occupancy::GridWindow;
using monoscale_occupancy::SweepSettings;

namespace
{

SweepSettings small()
{
  SweepSettings settings;
  settings.resolution = 0.1;
  settings.origin_x = 0.0;
  settings.origin_y = 0.0;
  settings.tile_size_cells = 8;
  return settings;
}

}  // namespace

TEST(Tiles, WindowRoundTripsThroughStorage)
{
  const SweepSettings s = small();
  CameraGrid grid;
  // A window deliberately straddling tile boundaries in both axes.
  GridWindow window;
  window.cell_x = -5;
  window.cell_y = 3;
  window.width = 21;
  window.height = 19;
  window.origin_x = s.origin_x + window.cell_x * s.resolution;
  window.origin_y = s.origin_y + window.cell_y * s.resolution;

  GridView view = grid.view(s, window, true);
  ASSERT_EQ(view.log_odds.cols, window.width);
  ASSERT_EQ(view.log_odds.rows, window.height);
  for (int r = 0; r < window.height; ++r) {
    for (int c = 0; c < window.width; ++c) {
      view.log_odds.at<float>(r, c) = static_cast<float>(r * 100 + c);
      view.observed.at<uint8_t>(r, c) = 1;
      view.slab_free[0].at<int16_t>(r, c) = static_cast<int16_t>(r);
      view.slab_free[1].at<int16_t>(r, c) = static_cast<int16_t>(c);
    }
  }
  grid.commit(s, view);

  const GridView back = grid.view(s, window, false);
  for (int r = 0; r < window.height; ++r) {
    for (int c = 0; c < window.width; ++c) {
      ASSERT_FLOAT_EQ(back.log_odds.at<float>(r, c), static_cast<float>(r * 100 + c));
      ASSERT_EQ(back.observed.at<uint8_t>(r, c), 1);
      ASSERT_EQ(back.slab_free[0].at<int16_t>(r, c), static_cast<int16_t>(r));
      ASSERT_EQ(back.slab_free[1].at<int16_t>(r, c), static_cast<int16_t>(c));
    }
  }
}

TEST(Tiles, AShiftedWindowReadsTheSameWorldCells)
{
  // What a cell holds must depend on where it is in the world, not on which
  // window happened to read it. This is the property the publish path relies
  // on when its extent moves with the drive.
  const SweepSettings s = small();
  CameraGrid grid;
  GridWindow first;
  first.cell_x = 0;
  first.cell_y = 0;
  first.width = 16;
  first.height = 16;
  first.origin_x = 0.0;
  first.origin_y = 0.0;

  GridView view = grid.view(s, first, true);
  view.log_odds.at<float>(6, 9) = 2.5f;   // world cell (9, 6)
  view.observed.at<uint8_t>(6, 9) = 1;
  grid.commit(s, view);

  GridWindow second;
  second.cell_x = 4;
  second.cell_y = 2;
  second.width = 16;
  second.height = 16;
  second.origin_x = s.origin_x + second.cell_x * s.resolution;
  second.origin_y = s.origin_y + second.cell_y * s.resolution;
  const GridView shifted = grid.view(s, second, false);

  // The same world cell, at its offset in the shifted window.
  EXPECT_FLOAT_EQ(shifted.log_odds.at<float>(6 - 2, 9 - 4), 2.5f);
  EXPECT_EQ(shifted.observed.at<uint8_t>(6 - 2, 9 - 4), 1);
}

TEST(Tiles, AReadOnlyViewDoesNotMintTiles)
{
  const SweepSettings s = small();
  CameraGrid grid;
  GridWindow window;
  window.cell_x = 1000;
  window.cell_y = -4000;
  window.width = 32;
  window.height = 32;

  GridView view = grid.view(s, window, false);
  EXPECT_EQ(grid.tile_count(), 0u);
  // Writing into it and committing must still leave nothing behind: a publish
  // reading empty ground must not turn that ground into stored map.
  view.log_odds.at<float>(0, 0) = 1.0f;
  grid.commit(s, view);
  EXPECT_EQ(grid.tile_count(), 0u);
}

TEST(Tiles, GroundFarFromTheStartIsKeptRatherThanDropped)
{
  // The fault this replaced: a 60 m box anchored where the run began, which
  // silently dropped everything beyond it. Measured 09-08 on the driving rig
  // with the car at x = 71.7 m, the whole map was outside the box.
  const SweepSettings s = small();
  CameraGrid grid;
  const GridWindow far = CameraGrid::around(s, 500.0, -400.0, 1.0);
  GridView view = grid.view(s, far, true);
  view.log_odds.at<float>(view.log_odds.rows / 2, view.log_odds.cols / 2) = 3.0f;
  view.observed.at<uint8_t>(view.log_odds.rows / 2, view.log_odds.cols / 2) = 1;
  grid.commit(s, view);

  const GridWindow held = grid.extent(s);
  ASSERT_FALSE(held.empty());
  // Placed through the global lattice, not through the window's world origin:
  // that origin is a large product and subtracting it loses a bit, which at
  // 500 m reads back one cell short. The production path takes the same route
  // for the same reason.
  const int column =
    static_cast<int>(std::floor((500.0 - s.origin_x) / s.resolution)) - held.cell_x;
  const int row =
    static_cast<int>(std::floor((-400.0 - s.origin_y) / s.resolution)) - held.cell_y;
  ASSERT_GE(column, 0);
  ASSERT_LT(column, held.width);
  ASSERT_GE(row, 0);
  ASSERT_LT(row, held.height);
  EXPECT_EQ(grid.view(s, held, false).observed.at<uint8_t>(row, column), 1);
}

TEST(Tiles, TheExtentGrowsWithWhatHasBeenMapped)
{
  const SweepSettings s = small();
  CameraGrid grid;
  GridView first = grid.view(s, CameraGrid::around(s, 0.0, 0.0, 0.2), true);
  grid.commit(s, first);
  const GridWindow one = grid.extent(s);
  ASSERT_FALSE(one.empty());

  GridView second = grid.view(s, CameraGrid::around(s, 20.0, 0.0, 0.2), true);
  grid.commit(s, second);
  const GridWindow two = grid.extent(s);
  EXPECT_GT(two.width, one.width);
  // 20 m at 0.1 m is 200 cells away; the extent has to reach it.
  EXPECT_GE(two.width, 200);
}

TEST(Tiles, TheLegacyWindowIsTheExtentOfflineWrites)
{
  // sweep_offline writes a bare .npy with no origin in it, and the scoring
  // reference reads it as grid_width x grid_height at (origin_x, origin_y).
  SweepSettings s = small();
  s.grid_width = 600;
  s.grid_height = 600;
  s.origin_x = -30.0;
  s.origin_y = -30.0;
  const GridWindow legacy = monoscale_occupancy::legacy_window(s);
  EXPECT_EQ(legacy.width, 600);
  EXPECT_EQ(legacy.height, 600);
  EXPECT_DOUBLE_EQ(legacy.origin_x, -30.0);
  EXPECT_DOUBLE_EQ(legacy.origin_y, -30.0);
  EXPECT_EQ(legacy.cell_x, 0);
  EXPECT_EQ(legacy.cell_y, 0);
}

TEST(Tiles, PublishOverEmptyStorageSaysNothing)
{
  const SweepSettings s = small();
  CameraGrid grid;
  const auto published = monoscale_occupancy::publish(s, {&grid});
  EXPECT_TRUE(published.window.empty());
}

TEST(Tiles, PublishAtTheLegacyExtentKeepsThatExtent)
{
  SweepSettings s = small();
  s.grid_width = 64;
  s.grid_height = 64;
  s.origin_x = -3.2;
  s.origin_y = -3.2;
  CameraGrid grid;
  GridView view = grid.view(s, CameraGrid::around(s, 0.0, 0.0, 0.5), true);
  view.observed.setTo(1);
  grid.commit(s, view);

  const auto published =
    monoscale_occupancy::publish(s, {&grid}, monoscale_occupancy::legacy_window(s));
  EXPECT_EQ(published.window.width, 64);
  EXPECT_EQ(published.window.height, 64);
  EXPECT_EQ(published.values.cols, 64);
  EXPECT_EQ(published.values.rows, 64);
}
