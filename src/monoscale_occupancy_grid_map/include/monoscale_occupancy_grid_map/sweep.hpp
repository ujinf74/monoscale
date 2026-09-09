// Occupancy by plane sweep, ported from tools/plane_sweep.py at the adopted
// operating point.
//
// The python original grew for a month under measurement; this port carries
// only what survived: the horizontal height ladder, ZNCC over a wide-short
// window, SGM aggregation, the believed/clear verdicts, sub-cell soft votes,
// the margin-tiered free carve, the two-slab column test, and the publish
// chain. Everything measured to lose -- vertical families, shadow projection,
// bridge-gap, ray votes, subpixel refinement -- is not here, and the file
// that explains why each one lost is the python original beside its flag.
//
// Defaults are the operating point measured on approach_hd60_occ_b
// (2026-09-02): G1 0 / G2 2 / G3 0.831 / G4 36 / path ghosts 0 on truth
// poses. Anything retuned should be retuned against that scoreboard.

#ifndef MONOSCALE_SWEEP__SWEEP_HPP_
#define MONOSCALE_SWEEP__SWEEP_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace monoscale_occupancy
{

// Equidistant fisheye with a fixed mount. The same three pieces the python
// CameraModel carried; the pinhole branch is not ported because every camera
// this runs on is the assembled fisheye.
struct Lens
{
  double focal = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  Eigen::Matrix3d rotation_base_from_camera = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_base_from_camera = Eigen::Vector3d::Zero();
};

// (x, y, yaw, roll, pitch) in the map frame. Roll and pitch matter: a drive
// pitching under throttle moves the ground ten metres out by six.
struct Pose5
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
};

Eigen::Matrix3d attitude(const Pose5 & pose);

// Per-pixel verdicts and vote weights the integrate step consumes.
struct Verdict
{
  cv::Mat believed;    // CV_8U: a surface stands here
  cv::Mat clear;       // CV_8U: the sweep called this pixel road
  cv::Mat confidence;  // CV_32F: decisive * quality, the occupied vote weight
  cv::Mat surface;     // CV_32F: the winning height, subplane-refined
};

struct SweepSettings
{
  // The ladder. -0.15 gives the match-failure pixels planes to drain into;
  // 0.9 is the camera height, above which a hypothesis only manufactures
  // ghosts (28-7).
  double height_min = -0.15;
  double height_step = 0.05;
  double obstacle_max_height = 0.9;
  // Below this a winner is road furniture, not an obstacle; the kerbs here
  // stand at 0.17 and a 0.20 floor made them undetectable.
  double obstacle_min_height = 0.10;

  // Matching. 7x3: wide beats tall because a horizontal plane is wrong about
  // a vertical surface in the vertical image direction (20-3).
  int window_x = 7;
  int window_y = 3;
  double seen_fraction = 0.25;
  // SGM steps, on the ZNCC-x100 cost scale.
  double small_step = 128.0;
  double big_step = 1280.0;

  // Verdicts.
  double uniqueness = 0.65;
  double margin = 800.0;
  double min_contrast = 4.0;
  double min_crosswise = 0.10;
  // 160 is what plane_sweep.py was tuned to, on its own float64 `believed`
  // mask. This port's mask comes off the fp32 CUDA cost volume and differs by
  // a few pixels per blob, which lands far obstacles either side of that
  // threshold -- and a far obstacle is small in pixels by construction, so the
  // filter prunes preferentially at range. Measured on approach_hd60_occ_b,
  // truth poses, 674 keyframes:
  //
  //   min_blob  160    G1 0  G2 14  G3 0.831  G4 29
  //   min_blob   80    G1 0  G2  3  G3 0.831  G4 29
  //   min_blob   60    G1 0  G2  3  G3 0.831  G4 31
  //   (python reference G1 0  G2  2  G3 0.829  G4 39)
  //
  // Eleven of the fourteen blocked-cells-called-free were carried by blobs
  // between 80 and 160 pixels; below 80 nothing further is bought and the
  // false-positive count starts to pay. The port at 80 beats the reference it
  // was checked against on both coverage and false positives.
  int min_blob = 80;
  int pixel_stride = 2;
  double min_distance = 1.0;
  double max_range = 30.0;
  bool subplane = true;

  // Occupied votes.
  bool subcell = true;
  bool soft_votes = true;
  double vote_scale = 300.0;
  double vote_cap = 1.0;
  double obstacle_range = 6.0;
  double far_weight = 0.3;
  double free_to_occupied = 0.2;

  // Free carve.
  double free_uniqueness = 0.9;
  double free_cost_max = 60.0;
  double free_margin = 0.0;
  double occlusion_aware = 0.98;
  int free_ray_stride = 1;
  double free_cap = 8.0;
  // The margin tiering: the confident half of the road verdicts carves at
  // `boost` times the base rate, the doubtful tail at `weight` times it.
  double free_margin_weight = 0.25;
  double free_margin_quantile = 0.25;
  double free_margin_boost = 2.0;

  // Two-slab column test.
  bool slab_carve = true;
  int slab_min = 1;
  double slab_local = 1.0;
  double slab_split = 0.35;
  int slab_samples = 48;

  // Grid and publish.
  double resolution = 0.1;
  // Cells live in tiles allocated where something is seen, so the map is not
  // bounded by an extent chosen up front. What these four values still do is
  // fix the lattice (which world point falls on a cell corner) and describe
  // the legacy fixed extent that `sweep_offline` asks for, so its .npy stays
  // the size and origin the scoring reference expects. The live node asks for
  // no extent and gets what has actually been mapped.
  int grid_width = 600;
  int grid_height = 600;
  double origin_x = -30.0;
  double origin_y = -30.0;
  // 64 cells is 6.4 m at 0.1 m. Tiles are created on demand and never
  // reclaimed: the sweep's own grid is the drive's record.
  int tile_size_cells = 64;
  double free_update = 0.45;
  double occupied_update = 0.9;
  double free_probability = 0.35;
  double occupied_probability = 0.97;
  double log_odds_floor = -8.0;
  int free_erode = 1;

  // Keyframing and sources.
  double keyframe_travel = 0.05;
  std::vector<double> source_offsets{0.6, -0.6, 1.2, -1.2, 2.4, -2.4};
  double source_tolerance = 0.15;
  int threads = 0;

  // Use the monoscale_fast CUDA kernel for the whole match-aggregate-reduce
  // when it is linked in; false forces the CPU path (the parity reference).
  bool use_cuda = true;
};

// A rectangle of the global cell lattice, and where its lower-left corner
// sits in the world.
struct GridWindow
{
  int32_t cell_x = 0;
  int32_t cell_y = 0;
  int width = 0;
  int height = 0;
  double origin_x = 0.0;
  double origin_y = 0.0;
  bool empty() const {return width <= 0 || height <= 0;}
};

// A dense working copy of one window of a camera's grid. Every accumulation
// and publish step runs against one of these, which is what lets the whole
// validated dense chain -- the subcell votes, the ray carve, the slab counts,
// distanceTransform, erode -- stay exactly as it was measured while the
// storage underneath became sparse.
struct GridView
{
  GridWindow window;
  cv::Mat log_odds;      // CV_32F
  cv::Mat observed;      // CV_8U
  cv::Mat slab_free[2];  // CV_16S, counts per z-slab
};

// One camera's accumulation state: tiles of cells, keyed by tile coordinate,
// created where the sweep puts something and held for the life of the drive.
struct CameraGrid
{
  struct Key
  {
    int32_t x = 0;
    int32_t y = 0;
    bool operator==(const Key & other) const {return x == other.x && y == other.y;}
  };
  struct KeyHash
  {
    std::size_t operator()(const Key & key) const
    {
      return static_cast<std::size_t>(static_cast<uint32_t>(key.x)) * 0x9E3779B97F4A7C15ull ^
             static_cast<std::size_t>(static_cast<uint32_t>(key.y));
    }
  };
  struct Tile
  {
    cv::Mat log_odds;      // CV_32F
    cv::Mat observed;      // CV_8U
    cv::Mat slab_free[2];  // CV_16S
  };

  void reset(const SweepSettings & settings);

  // A dense copy of the cells in `window`. With `create`, tiles the window
  // covers are allocated so a later commit has somewhere to land; without it,
  // absent tiles read as zeros and are not stored.
  GridView view(const SweepSettings & s, const GridWindow & window, bool create);

  // The window that covers `centre` out to `reach` metres, in whole cells.
  static GridWindow around(
    const SweepSettings & s, double x, double y, double reach);

  // Write a view's cells back into the tiles. Cells outside any existing tile
  // are dropped, so a read-only view commits only where it read.
  void commit(const SweepSettings & s, const GridView & view);

  // The window covering every tile held. Empty when nothing has been mapped.
  GridWindow extent(const SweepSettings & s) const;

  std::size_t tile_count() const {return tiles.size();}

  std::unordered_map<Key, Tile, KeyHash> tiles;
};

// What `publish` produced, and the piece of the world it covers.
struct Published
{
  cv::Mat values;   // CV_8S
  GridWindow window;
};

class Sweep
{
public:
  Sweep(const SweepSettings & settings, const Lens & lens);

  // Feed one keyframe: the reference gray image, its pose, and the source
  // frames with theirs. Updates the grid in place.
  void keyframe(
    const cv::Mat & reference_gray, const Pose5 & reference_pose,
    const std::vector<cv::Mat> & source_grays,
    const std::vector<Pose5> & source_poses,
    CameraGrid & grid) const;

  const SweepSettings & settings() const {return settings_;}

private:
  // Where each reference pixel is found in `source`, for the world-horizontal
  // plane at `offset`. The per-pixel replacement for the homography a fisheye
  // does not have.
  void warp_maps(
    const Pose5 & reference, const Pose5 & source, const Eigen::Vector3d & normal,
    double offset, const cv::Mat & dot_normal, cv::Mat & map_x, cv::Mat & map_y) const;

  // Semi-global aggregation over the height volume, four directional passes
  // summed, in place. Parity target: plane_sweep.aggregate.
  void aggregate(std::vector<cv::Mat> & volume, double small_step, double big_step) const;

  // Winner index, its cost, the best cost at or below the road rung, and the
  // second best with the winner's shoulder (+/-1 rung) excluded.
  void reduce_volume(
    const std::vector<cv::Mat> & volume, int road, cv::Mat & best,
    cv::Mat & best_cost, cv::Mat & road_cost, cv::Mat & second_cost) const;

  // The believed/clear masks, the soft-vote confidence, and the subplane
  // surface. Parity target: the `believed`, `confidence`, `textured`,
  // `judgeable`, min-blob and subplane blocks of plane_sweep.run.
  void build_verdict(
    const cv::Mat & reference32, const cv::Mat & best, const cv::Mat & best_cost,
    const cv::Mat & road_cost, const cv::Mat & second_cost,
    const std::vector<double> & heights, int road,
    const Pose5 & reference_pose, const std::vector<Pose5> & source_poses,
    const cv::Mat & dot_normal, const Eigen::Vector3d & normal,
    Verdict & out) const;

  // Occupied votes, margin-tiered free carve, and the two-slab bookkeeping,
  // for one keyframe. Parity target: the mark_occupied / carve_road /
  // slab_carve body of plane_sweep.run.
  void integrate(
    const Pose5 & reference_pose, const std::vector<double> & heights, int road,
    const cv::Mat & best, const cv::Mat & best_cost, const cv::Mat & second_cost,
    const Verdict & verdict, CameraGrid & grid) const;

  SweepSettings settings_;
  Lens lens_;
  int width_ = 0;
  int height_ = 0;
  // Unit rays through every pixel, in base_link; built on first use for the
  // frame size seen.
  mutable cv::Mat ray_base_;  // CV_64FC3
  void ensure_rays(int width, int height) const;
};

// The publish chain: neutralise the slab-failing free belief, combine the
// cameras by union, apply the column gate, erode the free rind. Returns the
// ternary map (-1 unknown / 0 free / 50 undecided / 100 occupied) and the
// window it covers.
//
// `request` names the extent to publish. Leave it empty and the map is
// cropped to the tiles that carry something, which is what the live node
// wants: the message then grows with the drive instead of being a box chosen
// before it started. `sweep_offline` passes the legacy fixed extent, because
// it writes a bare .npy with no origin in it and the scoring reference reads
// that file as 600 x 600 at (-30, -30).
Published publish(
  const SweepSettings & settings, std::vector<CameraGrid *> grids,
  const GridWindow & request = GridWindow());

// The legacy fixed extent named by grid_width/grid_height/origin_x/origin_y.
GridWindow legacy_window(const SweepSettings & settings);

}  // namespace monoscale_occupancy

#endif  // MONOSCALE_SWEEP__SWEEP_HPP_
