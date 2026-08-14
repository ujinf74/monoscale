// Accumulates the labelled points the odometry publishes into an occupancy
// grid.
//
// Keeping the grid here rather than inside the estimator is what lets the map
// be re-tuned, or replaced, without re-running the odometry: the same recorded
// cloud plays back into whatever accumulates it. The cloud carries the camera
// position each point was seen from, so this node needs no extrinsics, no
// transform tree and no opinion about how the pose was arrived at.

#include <cstring>
#include <memory>
#include <vector>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "monoscale_core/occupancy.hpp"

namespace
{

constexpr uint32_t kPointStep = 24;
constexpr float kGroundLabel = 0.0f;

struct Row
{
  float x;
  float y;
  float z;
  float label;
  float origin_x;
  float origin_y;
};

}  // namespace

class OccupancyGridMap : public rclcpp::Node
{
public:
  OccupancyGridMap()
  : rclcpp::Node("occupancy_grid_map")
  {
    monoscale::GridSettings settings;
    const auto points_topic = declare_parameter<std::string>(
      "ground_points_topic", "/perception/ground_points");
    const auto grid_topic = declare_parameter<std::string>(
      "occupancy_topic", "/perception/occupancy_grid_map");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");

    // 60 x 60 m at 10 cm, centred on where the run started. A parking
    // manoeuvre stays well inside it.
    settings.resolution = declare_parameter<double>("map_resolution", 0.1);
    settings.width = static_cast<int>(declare_parameter<int>("map_width", 600));
    settings.height = static_cast<int>(declare_parameter<int>("map_height", 600));
    settings.origin_x = declare_parameter<double>("map_origin_x", -30.0);
    settings.origin_y = declare_parameter<double>("map_origin_y", -30.0);

    // Evidence per observation, and the probabilities a cell has to reach
    // before it is called one thing or the other.
    settings.free_update = declare_parameter<double>("free_log_odds_update", 0.45);
    settings.occupied_update = declare_parameter<double>("occupied_log_odds_update", 0.9);
    settings.occupied_probability =
      declare_parameter<double>("occupied_probability_threshold", 0.65);
    settings.free_probability =
      declare_parameter<double>("free_probability_threshold", 0.35);
    settings.inflation_radius_m =
      declare_parameter<double>("obstacle_inflation_radius_m", 0.25);
    // Carving a ray per ground point costs more than it adds; every fourth one
    // sweeps the same free space.
    settings.free_ray_stride =
      static_cast<int>(declare_parameter<int>("free_ray_stride", 4));
    const double rate = declare_parameter<double>("occupancy_publish_rate_hz", 5.0);
    const int depth = static_cast<int>(declare_parameter<int>("input_queue_depth", 50));

    grid_ = std::make_unique<monoscale::LogOddsGrid>(settings);

    rclcpp::QoS input(static_cast<size_t>(std::max(depth, 1)));
    input.reliable();
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic, input,
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
        on_points(*message);
      });

    rclcpp::QoS latched(1);
    latched.reliable().transient_local();
    publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(grid_topic, latched);

    publish_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(rate, 0.1)),
      [this]() {publish();});
    // Said out loud while it runs, not only on the way out: a node killed by a
    // signal it never sees reports nothing at all.
    report_ = create_wall_timer(std::chrono::seconds(2), [this]() {report();});
  }

private:
  void on_points(const sensor_msgs::msg::PointCloud2 & message)
  {
    if (message.point_step != kPointStep) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000, "Expected %u byte points, got %u", kPointStep,
        message.point_step);
      return;
    }
    const size_t count = message.data.size() / kPointStep;
    std::vector<Row> rows(count);
    std::memcpy(rows.data(), message.data.data(), count * kPointStep);

    last_stamp_ = message.header.stamp;
    have_stamp_ = true;
    ++clouds_;
    points_ += count;

    // Free space is carved from where the camera stood to where the road was
    // seen; an obstacle marks its own cell and nothing else.
    const int stride = std::max(grid_->settings().free_ray_stride, 1);
    int ground_index = 0;
    for (const Row & row : rows) {
      if (row.label == kGroundLabel) {
        if (ground_index++ % stride == 0) {
          grid_->integrate_ray(row.origin_x, row.origin_y, row.x, row.y, false);
        }
      } else {
        ++obstacles_;
        grid_->integrate_point(row.x, row.y);
      }
    }
  }

  void report()
  {
    if (clouds_ == 0) {
      return;
    }
    RCLCPP_INFO(
      get_logger(), "occupancy: clouds=%ld points=%ld obstacles=%ld", clouds_, points_,
      obstacles_);
  }

  void publish()
  {
    if (!have_stamp_) {
      return;
    }
    const auto & settings = grid_->settings();
    nav_msgs::msg::OccupancyGrid message;
    message.header.stamp = last_stamp_;
    message.header.frame_id = map_frame_;
    message.info.resolution = static_cast<float>(settings.resolution);
    message.info.width = static_cast<uint32_t>(settings.width);
    message.info.height = static_cast<uint32_t>(settings.height);
    message.info.origin.position.x = settings.origin_x;
    message.info.origin.position.y = settings.origin_y;
    message.info.origin.orientation.w = 1.0;
    message.data = grid_->message_values();
    publisher_->publish(message);
  }

  std::string map_frame_;
  std::unique_ptr<monoscale::LogOddsGrid> grid_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr publish_;
  rclcpp::TimerBase::SharedPtr report_;
  builtin_interfaces::msg::Time last_stamp_;
  bool have_stamp_ = false;
  int64_t clouds_ = 0;
  int64_t points_ = 0;
  int64_t obstacles_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OccupancyGridMap>());
  rclcpp::shutdown();
  return 0;
}
