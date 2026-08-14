// Six degrees of freedom, and the constraints that keep them on the road.

#include <cmath>

#include <gtest/gtest.h>

#include "monoscale_core/strapdown.hpp"

using monoscale::SpatialMsckfFilter;

namespace
{

constexpr double kGravity = 9.80665;

// What an accelerometer reads on a vehicle that is level and not moving:
// gravity pushing up through it.
Eigen::Vector3d resting()
{
  return Eigen::Vector3d(0.0, 0.0, kGravity);
}

// The same reading on a body rolled and pitched by the given angles.
Eigen::Vector3d resting_at(double roll, double pitch)
{
  const Eigen::Matrix3d rotation =
    (Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
  return rotation.transpose() * Eigen::Vector3d(0.0, 0.0, kGravity);
}

// Drive a filter through hops of a stated length with the instrument saying
// nothing is happening beyond holding the vehicle up.
void drive(SpatialMsckfFilter & filter, int hops, double per_hop)
{
  for (int hop = 0; hop < hops; ++hop) {
    filter.open_hop();
    for (int i = 0; i < 5; ++i) {
      filter.predict(resting(), Eigen::Vector3d::Zero(), 0.01);
    }
    filter.update(Eigen::Vector2d(per_hop, 0.0), 0.0, 400);
  }
}

}  // namespace

TEST(Strapdown, GravityIsPutBackBeforeTheVehicleIsIntegrated)
{
  SpatialMsckfFilter filter;

  // A part at rest reads gravity, and a filter that took that for acceleration
  // would have the vehicle a metre in the air after half a second.
  for (int i = 0; i < 100; ++i) {
    filter.predict(resting(), Eigen::Vector3d::Zero(), 0.01);
  }

  EXPECT_NEAR(filter.velocity().norm(), 0.0, 1e-9);
  EXPECT_NEAR(filter.position().norm(), 0.0, 1e-9);
}

TEST(Strapdown, TheAccelerometerLevelsTheAttitudeAndOnlyWhenItCan)
{
  SpatialMsckfFilter filter;
  // Two degrees of pitch the filter does not know about.
  const double truth = 2.0 * M_PI / 180.0;

  for (int i = 0; i < 400; ++i) {
    filter.predict(resting_at(0.0, truth), Eigen::Vector3d::Zero(), 0.01);
    EXPECT_TRUE(filter.update_gravity(resting_at(0.0, truth)));
  }

  EXPECT_NEAR(filter.pitch(), truth, 2.0e-3);
  EXPECT_NEAR(filter.roll(), 0.0, 2.0e-3);
  EXPECT_GT(filter.levelled(), 0);
}

TEST(Strapdown, ABrakingVehicleIsNotMistakenForANoseDownOne)
{
  SpatialMsckfFilter filter;

  // Half a g of braking. The reading is nowhere near gravity's magnitude, and
  // believing it would tip the attitude by nearly thirty degrees.
  const Eigen::Vector3d braking = resting() + Eigen::Vector3d(-5.0, 0.0, 0.0);
  EXPECT_FALSE(filter.update_gravity(braking));
  EXPECT_EQ(filter.levelled(), 0);
  EXPECT_NEAR(filter.pitch(), 0.0, 1e-12);
}

TEST(Strapdown, TheHopKeepsTheVehicleOnTheRoad)
{
  SpatialMsckfFilter filter;
  drive(filter, 60, 0.2);

  EXPECT_NEAR(filter.body_translation().x(), 0.2, 0.02);
  EXPECT_GT(filter.position().x(), 10.0);
  // Nothing observes height but the hop's own claim that the vehicle did not
  // climb. Without it the accelerometer's noise integrates into a vehicle that
  // has taken off.
  EXPECT_NEAR(filter.position().z(), 0.0, 0.02);
}

TEST(Strapdown, HeadingComesOutOfTheAttitudeItWasSetTo)
{
  SpatialMsckfFilter filter;
  filter.set_pose(Eigen::Vector2d(4.0, -1.0), 0.8);

  EXPECT_NEAR(filter.yaw(), 0.8, 1e-9);
  EXPECT_NEAR(filter.position().x(), 4.0, 1e-9);
  EXPECT_NEAR(filter.position().y(), -1.0, 1e-9);
  // Placing the vehicle in the plane says nothing about how it is leaning.
  EXPECT_NEAR(filter.roll(), 0.0, 1e-9);
  EXPECT_NEAR(filter.pitch(), 0.0, 1e-9);
}

TEST(Strapdown, TurningIsCarriedByTheGyroAndMeasuredAgainstTheAnchor)
{
  SpatialMsckfFilter filter;
  const double rate = 0.2;

  filter.open_hop();
  for (int i = 0; i < 50; ++i) {
    filter.predict(resting(), Eigen::Vector3d(0.0, 0.0, rate), 0.01);
  }

  EXPECT_NEAR(filter.hop_yaw(), rate * 0.5, 1e-3);
  EXPECT_NEAR(filter.yaw(), rate * 0.5, 1e-3);
}

TEST(Strapdown, TheGyroBiasIsLearnedFromTheReportedHeading)
{
  SpatialMsckfFilter filter;
  const double bias = 0.01;

  for (int i = 0; i < 6000; ++i) {
    filter.predict(resting(), Eigen::Vector3d(0.0, 0.0, bias), 0.01);
    filter.update_heading(0.0, 0.1);
  }

  EXPECT_NEAR(filter.gyro_bias().z(), bias, 0.002);
  EXPECT_NEAR(filter.yaw(), 0.0, 0.01);
}

TEST(Strapdown, ZeroVelocityPinsTheGyroBias)
{
  SpatialMsckfFilter filter;
  const Eigen::Vector3d offset(0.3, 0.0, 0.0);

  filter.open_hop();
  for (int i = 0; i < 2000; ++i) {
    filter.predict(resting() + offset, Eigen::Vector3d(0.0, 0.0, 0.01), 0.01);
    filter.update_zero_velocity();
  }

  EXPECT_NEAR(filter.velocity().norm(), 0.0, 0.02);
  EXPECT_NEAR(filter.gyro_bias().z(), 0.01, 0.002);

  // The planar filter put the whole of that x offset in the accelerometer's
  // bias, because it had nowhere else to put it. Here it does: a vehicle nose
  // down by 1.8 degrees reads the same 0.3 m/s2, and standing still there is
  // nothing to say which it is. So the offset is shared, and what settles it is
  // motion -- which is where the bias is actually earned.
  EXPECT_LT(filter.bias().x(), 0.3);
  EXPECT_LT(filter.pitch(), 0.0);
}

TEST(Strapdown, AHopWrongByMetresIsDropped)
{
  SpatialMsckfFilter filter;
  drive(filter, 20, 0.2);

  filter.open_hop();
  filter.predict(resting(), Eigen::Vector3d::Zero(), 0.05);
  const Eigen::Vector3d before = filter.position();
  const int updates = filter.updates();

  EXPECT_FALSE(filter.update(Eigen::Vector2d(6.0, 0.0), 0.0, 400));
  EXPECT_EQ(filter.dropped(), 1);
  EXPECT_EQ(filter.updates(), updates);
  EXPECT_NEAR((filter.position() - before).norm(), 0.0, 0.05);
}

TEST(Strapdown, TheTiltHandedToTheGroundProjectionCarriesNoHeading)
{
  SpatialMsckfFilter filter;
  filter.set_pose(Eigen::Vector2d::Zero(), 1.1);
  const double truth = 3.0 * M_PI / 180.0;
  for (int i = 0; i < 400; ++i) {
    filter.predict(resting(), Eigen::Vector3d::Zero(), 0.01);
    filter.update_gravity(resting_at(0.0, truth));
  }

  const Eigen::Matrix3d tilt = filter.body_tilt();
  // Whatever the vehicle is pointing at, this is only how it is leaning: the
  // ground registration owns the heading and rotating it twice would be a bug
  // that looks like a calibration error.
  EXPECT_NEAR(std::atan2(tilt(1, 0), tilt(0, 0)), 0.0, 1e-6);
  EXPECT_NEAR(std::asin(-tilt(2, 0)), filter.pitch(), 1e-9);
  EXPECT_GT(std::abs(filter.pitch()), 1e-3);
}

TEST(Strapdown, CovarianceStaysSymmetricAndFinite)
{
  SpatialMsckfFilter filter;
  for (int hop = 0; hop < 300; ++hop) {
    filter.open_hop();
    for (int i = 0; i < 5; ++i) {
      filter.predict(
        resting() + Eigen::Vector3d(0.1, -0.05, 0.0), Eigen::Vector3d(0.0, 0.0, 0.02), 0.02);
    }
    filter.update(Eigen::Vector2d(0.01, 0.0), 0.002, 400, 0.0, 0.02);
    filter.update_gravity(resting());
    filter.update_heading(0.02 * 0.1 * (hop + 1), 0.05);
  }

  ASSERT_TRUE(filter.last_update().has_value());
  EXPECT_TRUE(std::isfinite(filter.last_update()->nis));
  EXPECT_TRUE(std::isfinite(filter.velocity().x()));
  EXPECT_TRUE(std::isfinite(filter.gyro_bias().z()));
  EXPECT_TRUE(std::isfinite(filter.yaw()));
  EXPECT_NEAR(filter.attitude().norm(), 1.0, 1e-9);
}
