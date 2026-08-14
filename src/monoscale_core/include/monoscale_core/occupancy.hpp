// An occupancy grid built from what the odometry saw on the road.
//
// The estimator publishes the points its solve already produced, labelled: a
// feature that survived the ground registration is road, one whose projection
// slid against the travel is standing on it. Both arrive in the map frame with
// the camera position they were seen from, which is everything this needs --
// no extrinsics, no transform tree, and no opinion about how the pose was
// arrived at.

#ifndef MONOSCALE_CORE__OCCUPANCY_HPP_
#define MONOSCALE_CORE__OCCUPANCY_HPP_

#include <cstdint>
#include <vector>

namespace monoscale
{

struct GridSettings
{
  double resolution = 0.1;
  int width = 600;
  int height = 600;
  double origin_x = -30.0;
  double origin_y = -30.0;
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
};

class LogOddsGrid
{
public:
  explicit LogOddsGrid(const GridSettings & settings);

  const GridSettings & settings() const {return settings_;}

  // Add occupied evidence to one cell, carving nothing on the way in. A
  // triangulated obstacle is only as trustworthy as its depth, and depth from a
  // 10 cm baseline is not trustworthy enough to declare everything in front of
  // it empty. Carving from these rays erased real obstacles.
  void integrate_point(double x, double y);

  // Walk the cells between two points. `occupied` marks the far end and leaves
  // everything before it free; otherwise the whole line is carved.
  void integrate_ray(double from_x, double from_y, double to_x, double to_y, bool occupied);

  // The grid as an OccupancyGrid carries it: -1 unknown, 0 free, 50 undecided,
  // 100 occupied, with occupied cells dilated by the inflation radius.
  std::vector<int8_t> message_values() const;

private:
  bool inside(int x, int y) const
  {
    return x >= 0 && x < settings_.width && y >= 0 && y < settings_.height;
  }
  void mark_free(int x, int y);
  void mark_occupied(int x, int y);

  GridSettings settings_;
  std::vector<float> log_odds_;
  std::vector<uint8_t> observed_;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__OCCUPANCY_HPP_
