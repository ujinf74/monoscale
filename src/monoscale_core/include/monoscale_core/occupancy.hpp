// An occupancy grid built from what the odometry saw on the road.
//
// The estimator publishes the points its solve already produced, labelled: a
// feature that survived the ground registration is road, one whose projection
// slid against the travel is standing on it. Both arrive in the map frame with
// the camera position they were seen from, which is everything this needs --
// no extrinsics, no transform tree, and no opinion about how the pose was
// arrived at.
//
// The cells live in tiles held in a hash, not in one array sized up front. A
// fixed window was sized for a parking manoeuvre and silently dropped half the
// points of a road drive the moment the car left it. Tiles are allocated where
// something is seen, forgotten when they carry nothing, and what gets published
// is the part of them near the car -- so the map grows and shrinks with the
// drive. The structure and its settled parameters come from the accumulator in
// ioniq-autopark-ws, which was validated against a LiDAR reference; the scales
// are converted to the log odds this grid already uses.

#ifndef MONOSCALE_CORE__OCCUPANCY_HPP_
#define MONOSCALE_CORE__OCCUPANCY_HPP_

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace monoscale
{

struct GridSettings
{
  double resolution = 0.1;
  // Where the cell lattice is anchored. Not a bound any more: it only fixes
  // which world point falls on a cell corner.
  double origin_x = 0.0;
  double origin_y = 0.0;
  // 64 cells is 6.4 m at 0.1 m, the same ground a tile covers in the reference
  // accumulator at its coarser resolution.
  int tile_size_cells = 64;
  // The most that is ever published, in tiles: 15 x 64 cells is 96 m across.
  // The message is cropped to the tiles that actually carry something, so this
  // is a ceiling on bandwidth rather than the usual size. Raising it costs
  // bytes on the wire quadratically.
  int roi_tiles_x = 15;
  int roi_tiles_y = 15;
  double free_update = 0.45;
  double occupied_update = 0.9;
  // Where a cell stops being unknown. Raising the occupied threshold buys
  // precision at the cost of recall, and vice versa. It ships at 0.65, where a
  // single triangulation is enough: for a system that warns a driver, missing
  // an obstacle is worse than mentioning one that is not there.
  double occupied_probability = 0.65;
  double free_probability = 0.35;
  // How much free history a cell may bank. At -4 the bank is seven occupied
  // votes deep: a cell driven past and carved free a hundred times flips from
  // eight misplaced close-range votes, which is how ground already mapped free
  // turned into obstacles as the car passed.
  double floor = -4.0;
  // Casting a Bresenham ray per ground inlier does not fit in a 20 Hz budget,
  // so only every nth inlier carves free space.
  int free_ray_stride = 4;
  double inflation_radius_m = 0.25;

  // Evidence bleeding back towards unknown. The reference runs this at 1.0 on a
  // scale where a free observation is worth 20, so a single one fades in twenty
  // seconds; the same ratio here is 0.0225. It ships off, because that rate was
  // settled against a LiDAR returning thousands of points at 10 Hz and vision
  // carves far fewer -- turn it on for a moving scene, where stale free space
  // is a hazard rather than a memory.
  double decay_log_odds_per_sec = 0.0;
  // A tile with nothing worth keeping is dropped once it has gone this long
  // without an update. 0.135 is the reference's 6 on the same 0.0225 scale:
  // low enough that a fraction of one observation still counts as something.
  double collect_min_abs_log_odds = 0.135;
  double collect_min_age_sec = 3.0;
  // The same, for a tile that has been quiet far longer.
  double tile_forget_sec = 60.0;
};

class LogOddsGrid
{
public:
  explicit LogOddsGrid(const GridSettings & settings);

  const GridSettings & settings() const {return settings_;}

  // The published extent: a rectangle of cells in the global lattice, with the
  // world position of its lower-left corner.
  struct Window
  {
    int32_t cell_x = 0;
    int32_t cell_y = 0;
    int width = 0;
    int height = 0;
    double origin_x = 0.0;
    double origin_y = 0.0;
  };

  // Carry the clock forward, letting evidence decay. Every integrate after this
  // stamps its tiles with `now`.
  void advance(double now);

  // Drop tiles that hold nothing worth publishing. Kept separate from advance
  // so a whole cloud lands before anything is reconsidered.
  void collect();

  // Add occupied evidence to one cell, carving nothing on the way in. A
  // triangulated obstacle is only as trustworthy as its depth, and depth from a
  // 10 cm baseline is not trustworthy enough to declare everything in front of
  // it empty. Carving from these rays erased real obstacles.
  void integrate_point(double x, double y);

  // Walk the cells between two points. `occupied` marks the far end and leaves
  // everything before it free; otherwise the whole line is carved.
  void integrate_ray(double from_x, double from_y, double to_x, double to_y, bool occupied);

  // The tiles worth drawing within reach of (x, y), cropped to their bounding
  // box. Empty width and height when there is nothing to say.
  Window window(double x, double y) const;

  // The window as an OccupancyGrid carries it: -1 unknown, 0 free, 50
  // undecided, 100 occupied, with occupied cells dilated by the inflation
  // radius.
  std::vector<int8_t> message_values(const Window & window) const;

  size_t tile_count() const {return tiles_.size();}

private:
  struct TileKey
  {
    int32_t x = 0;
    int32_t y = 0;
    bool operator==(const TileKey & other) const {return x == other.x && y == other.y;}
  };
  struct TileHash
  {
    size_t operator()(const TileKey & key) const
    {
      return static_cast<size_t>(static_cast<uint32_t>(key.x)) * 0x9E3779B97F4A7C15ull ^
             static_cast<size_t>(static_cast<uint32_t>(key.y));
    }
  };
  struct Tile
  {
    std::vector<float> log_odds;
    std::vector<uint8_t> observed;
    double last_update = 0.0;
  };

  static int32_t floor_div(int32_t value, int32_t by);
  int32_t cell_of(double world, double origin) const;
  // The tile this cell belongs to, created if it is the first thing seen there.
  Tile & tile_for(int32_t cell_x, int32_t cell_y, size_t & at);
  const Tile * tile_at(const TileKey & key) const;
  float strongest(const Tile & tile) const;
  void mark(int32_t cell_x, int32_t cell_y, bool occupied);

  GridSettings settings_;
  std::unordered_map<TileKey, Tile, TileHash> tiles_;
  double now_ = 0.0;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__OCCUPANCY_HPP_
