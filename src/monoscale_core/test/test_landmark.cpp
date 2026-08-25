// The unified filter on a world it is told the truth about.
//
// What is being checked is that bearings alone pull the pose back when the
// step it was given is wrong -- which is the whole claim: the ground and the
// structure above it are the same measurement, and the only thing separating
// them is how tightly each one's range starts out.
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "monoscale_core/landmark.hpp"

using monoscale::LandmarkFilter;
using monoscale::LandmarkSettings;

namespace
{

Eigen::Matrix3d turn_of(double yaw)
{
  Eigen::Matrix3d out = Eigen::Matrix3d::Identity();
  out(0, 0) = std::cos(yaw); out(0, 1) = -std::sin(yaw);
  out(1, 0) = std::sin(yaw); out(1, 1) = std::cos(yaw);
  return out;
}

// The front mount of the shipped rig, near enough.
const Eigen::Vector3d kMount(3.694, 0.0, 0.89);

std::vector<Eigen::Vector3d> world()
{
  std::vector<Eigen::Vector3d> out;
  // Road under the camera, and structure standing beside it. Nothing in the
  // filter tells them apart; only the range prior differs.
  for (int i = -8; i <= 60; ++i) {
    for (int j = -6; j <= 6; ++j) {
      out.emplace_back(0.5 * i, 0.5 * j, 0.0);
    }
  }
  for (int i = 0; i <= 40; ++i) {
    out.emplace_back(0.8 * i, 6.0, 1.5);
    out.emplace_back(0.8 * i, -6.0, 2.5);
  }
  return out;
}

// Everything visible from a pose, as the filter wants it.
std::vector<LandmarkFilter::Sighting> look(
  const std::vector<Eigen::Vector3d> & points, const Eigen::Vector3d & pose)
{
  std::vector<LandmarkFilter::Sighting> out;
  const Eigen::Matrix3d turn = turn_of(pose.z());
  const Eigen::Vector3d centre =
    Eigen::Vector3d(pose.x(), pose.y(), 0.0) + turn * kMount;
  for (size_t n = 0; n < points.size(); ++n) {
    const Eigen::Vector3d direction = turn.transpose() * (points[n] - centre);
    const double range = direction.norm();
    // A camera that looks forward and down, out to a few metres of road.
    if (direction.x() <= 0.2 || range > 12.0) {
      continue;
    }
    LandmarkFilter::Sighting sighting;
    sighting.identity = static_cast<int64_t>(n);
    sighting.mount = kMount;
    sighting.bearing = direction / range;
    // The plane answers for anything on it, and says nothing about the rest.
    if (std::abs(points[n].z()) < 1e-9 && range < 6.0) {
      sighting.ground_range = range;
    }
    out.push_back(sighting);
  }
  return out;
}

}  // namespace

TEST(LandmarkFilter, BearingsPullThePoseBackWhenTheStepIsWrong)
{
  const auto points = world();
  LandmarkSettings settings;
  settings.bearing_noise_rad = 0.002;
  settings.hop_process_noise_m = 0.10;
  settings.ground_in_state = true;
  LandmarkFilter filter(settings);

  Eigen::Vector3d truth = Eigen::Vector3d::Zero();
  const double step = 0.12;
  // Twenty steps of the truth, so the map is built and the filter is settled.
  for (int i = 0; i < 20; ++i) {
    truth.x() += step;
    filter.predict(Eigen::Vector2d(step, 0.0), 0.0);
    filter.observe(look(points, truth), i);
    filter.retire(i);
  }
  ASSERT_GT(filter.landmarks(), 100u);
  EXPECT_NEAR(filter.pose().x(), truth.x(), 0.02);

  // Now lie to it: tell it the vehicle went half as far as it did, for ten
  // steps. Nothing but the bearings can know otherwise.
  for (int i = 20; i < 30; ++i) {
    truth.x() += step;
    filter.predict(Eigen::Vector2d(0.5 * step, 0.0), 0.0);
    filter.observe(look(points, truth), i);
    filter.retire(i);
  }
  // Ten steps of half a hop is 0.6 m of lie.
  EXPECT_NEAR(filter.pose().x(), truth.x(), 0.05)
    << "the bearings did not recover the step the prediction withheld";
  EXPECT_NEAR(filter.pose().y(), truth.y(), 0.03);
  EXPECT_NEAR(filter.pose().z(), truth.z(), 0.01);
}

TEST(LandmarkFilter, StructureOffTheGroundCarriesTheSameWeight)
{
  // With the plane's prior withheld entirely, the filter has only the points
  // standing beside the road -- initialised from their own parallax -- and it
  // still has to hold the pose.
  auto points = world();
  LandmarkSettings settings;
  settings.bearing_noise_rad = 0.002;
  settings.range_from_plane = false;
  settings.initialise_min_views = 4;
  settings.ground_in_state = true;
  LandmarkFilter filter(settings);

  Eigen::Vector3d truth = Eigen::Vector3d::Zero();
  const double step = 0.15;
  for (int i = 0; i < 40; ++i) {
    truth.x() += step;
    filter.predict(Eigen::Vector2d(i < 25 ? step : 0.6 * step, 0.0), 0.0);
    filter.observe(look(points, truth), i);
    filter.retire(i);
  }
  ASSERT_GT(filter.landmarks(), 50u);
  EXPECT_NEAR(filter.pose().x(), truth.x(), 0.10)
    << "triangulated structure alone could not hold the step";
}

TEST(LandmarkFilter, GroundFeaturesTakeNoSeatAndAreNotMissed)
{
  // The deployed shape: the road is still seen and still handed in, but it does
  // not enter the state. Only what stands beside it takes a seat, so the state
  // stays small enough to update, and the step still has to be recovered.
  const auto points = world();
  LandmarkSettings settings;
  settings.bearing_noise_rad = 0.002;
  settings.ground_in_state = false;
  settings.max_landmarks = 200;
  LandmarkFilter filter(settings);

  // How many of the sightings are road, so the saving is stated rather than
  // assumed.
  size_t road = 0;
  for (const auto & sighting : look(points, Eigen::Vector3d::Zero())) {
    road += sighting.ground_range.has_value() ? 1 : 0;
  }
  ASSERT_GT(road, 50u) << "the fixture is not handing the filter any road";

  Eigen::Vector3d truth = Eigen::Vector3d::Zero();
  const double step = 0.15;
  for (int i = 0; i < 40; ++i) {
    truth.x() += step;
    // Honest for twenty-five steps, then two-thirds of the hop for fifteen.
    filter.predict(Eigen::Vector2d(i < 25 ? step : 0.6 * step, 0.0), 0.0);
    filter.observe(look(points, truth), i);
    filter.retire(i);
  }
  EXPECT_LE(filter.landmarks(), 200u);
  EXPECT_GT(filter.landmarks(), 20u) << "no structure was held at all";
  EXPECT_NEAR(filter.pose().x(), truth.x(), 0.10)
    << "structure alone could not recover the step the prediction withheld";
}

TEST(LandmarkFilter, AFullStateTurnsOverInsteadOfRefusingTheNewcomer)
{
  // Seats are finite and the vehicle drives away from whatever filled them
  // first. With the newcomer simply refused, the state is whatever was in view
  // at the start; with the seats compared, it has to follow the vehicle.
  const auto points = world();
  LandmarkSettings settings;
  settings.bearing_noise_rad = 0.002;
  settings.max_landmarks = 40;
  settings.retire_unseen_frames = 10000;   // the clock must not do this for us
  settings.evict_by_contribution = true;
  LandmarkFilter filter(settings);

  Eigen::Vector3d truth = Eigen::Vector3d::Zero();
  const double step = 0.15;
  for (int i = 0; i < 60; ++i) {
    truth.x() += step;
    filter.predict(Eigen::Vector2d(step, 0.0), 0.0);
    filter.observe(look(points, truth), i);
    filter.retire(i);
  }
  EXPECT_EQ(filter.landmarks(), 40u) << "the seats should be full";
  EXPECT_GT(filter.evicted(), 0) << "nothing was ever displaced from a full state";
  EXPECT_NEAR(filter.pose().x(), truth.x(), 0.15);
}
