// What the heading filter has to do that a gain could not, from
// src/monoscale_odometry/test/test_heading_filter.py.
//
// Two things were tried before this and measured on real drives. A hard bound
// on how far the ground solve may move the heading recovered a drifting gyro at
// 8 m/s and ruined a 2.5 m/s drive, because a push of up to the bound on every
// solve is a random walk and a slower vehicle takes more solves per metre.
// Weighting the share by precision -- the ordinary scalar gain -- was no better
// anywhere.
//
// The reason both failed is the same. A bias is not noise. Inflate its variance
// and pay that off against a measurement and the next interval earns it
// straight back. So the property to test for is not that the heading gets
// corrected, but that the rate behind it is learned and stops coming back.

#include <cmath>

#include <gtest/gtest.h>

#include "monoscale_core/attitude.hpp"

using monoscale::AttitudeFilter;
using monoscale::HeadingBiasFilter;

namespace
{

// Box-Muller over a fixed sequence, so the drive is repeatable without pulling
// in a generator whose stream we would have to match.
class Noise
{
public:
  explicit Noise(unsigned int seed)
  : state_(seed) {}

  double normal(double sigma)
  {
    const double u1 = std::max(uniform(), 1e-12);
    const double u2 = uniform();
    return sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  }

private:
  double uniform()
  {
    state_ = state_ * 1103515245u + 12345u;
    return static_cast<double>((state_ >> 16) & 0x7fffu) / 32768.0;
  }

  unsigned int state_;
};

// Drive the filter with a heading that drifts at `bias` radians a second.
// Returns how far the heading was out at the end, after every correction the
// filter asked for has been applied to it.
double run(
  HeadingBiasFilter & filter, double bias, int steps, double dt = 0.02,
  double sigma = 0.01, unsigned int seed = 0)
{
  Noise noise(seed);
  double error = 0.0;
  for (int i = 0; i < steps; ++i) {
    error += bias * dt;
    filter.predict(dt);
    // What the ground says the heading should have been, seen through a solve
    // of finite precision.
    const double measured = -error + noise.normal(sigma);
    error += filter.update(measured, sigma);
  }
  return error;
}

}  // namespace

TEST(Heading, DisabledByDefaultTheHeadingIsLeftAlone)
{
  HeadingBiasFilter quiet(0.0, 1e-5, 1e-3);

  EXPECT_FALSE(quiet.enabled());
  EXPECT_DOUBLE_EQ(quiet.update(0.5, 0.01), 0.0);
  EXPECT_NEAR(run(quiet, 0.005, 200), 0.005 * 0.02 * 200, 1e-12);
}

TEST(Heading, AConstantBiasIsLearnedAndStopsAccumulating)
{
  const double bias = 0.005;
  HeadingBiasFilter filter(0.01, 1e-6, 1e-4);

  const double left = run(filter, bias, 600);

  // The rate it settled on is the one that was there.
  EXPECT_NEAR(filter.rate(), -bias, 0.2 * bias);
  // And what the heading has left over is a fraction of a milliradian, against
  // the 60 mrad it would have accumulated untouched.
  EXPECT_LT(std::abs(left), 0.002);
}

TEST(Heading, LearningTheRateIsWhatMakesTheDifference)
{
  // The same filter, denied its second state, cannot keep up. Pinning the rate
  // to zero leaves something that can only chase the error it can already see,
  // which is the shape both earlier attempts had.
  const double bias = 0.005;
  HeadingBiasFilter full(0.01, 1e-6, 1e-4);
  HeadingBiasFilter rateless(0.01, 1e-6, 1e-4);
  rateless.covariance()(1, 1) = 0.0;

  const double with_rate = std::abs(run(full, bias, 600));
  const double without = std::abs(run(rateless, bias, 600));

  EXPECT_LT(with_rate, 0.25 * without);
}

TEST(Heading, AQuietInstrumentIsNotTalkedOutOfItsHeading)
{
  // Nothing to find, so nothing should be done. This is the case the simulator
  // provides and a vehicle never will, and the filter costing accuracy here is
  // the price of it working elsewhere -- but the price has to stay small.
  HeadingBiasFilter filter(0.01, 1e-6, 1e-4);

  const double left = std::abs(run(filter, 0.0, 600, 0.02, 0.01, 3));

  EXPECT_LT(left, 0.002);
  EXPECT_LT(std::abs(filter.rate()), 0.001);
}

TEST(Heading, TheCovarianceStaysSymmetricAndPositive)
{
  HeadingBiasFilter filter(0.01, 1e-6, 1e-4);

  run(filter, 0.005, 300);

  const Eigen::Matrix2d & covariance = filter.covariance();
  EXPECT_NEAR(covariance(0, 1), covariance(1, 0), 1e-15);
  EXPECT_GT(covariance(0, 0), 0.0);
  EXPECT_GT(covariance(1, 1), 0.0);
  EXPECT_GT(covariance.determinant(), 0.0);
  EXPECT_TRUE(covariance.allFinite());
}

TEST(Attitude, LevelIsTheStartingPriorNotTheFirstSample)
{
  // Seeding from the first sample of a drive that opens under throttle starts
  // the pitch 15 degrees out, and a sixty second trim never recovers inside a
  // thirty second drive. It scored 18.9 m.
  AttitudeFilter filter(60.0, 0.3);

  filter.update(Eigen::Vector3d::Zero(), Eigen::Vector3d(-2.5, 0.0, 9.5), 0.01);

  EXPECT_TRUE(filter.started());
  EXPECT_DOUBLE_EQ(filter.roll(), 0.0);
  EXPECT_DOUBLE_EQ(filter.pitch(), 0.0);
}

TEST(Attitude, TheGyroCarriesTheAttitudeAndGravityOnlyTrimsIt)
{
  AttitudeFilter filter(60.0, 0.3);
  filter.update(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 9.80665), 0.01);

  // A steady roll rate for a second, with the accelerometer reporting level
  // gravity throughout. The trim is sixty seconds long, so the gyro wins.
  for (int i = 0; i < 100; ++i) {
    filter.update(
      Eigen::Vector3d(0.1, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 9.80665), 0.01);
  }

  EXPECT_NEAR(filter.roll(), 0.1, 0.01);
}

TEST(Attitude, AShakenInstrumentIsNotAllowedToTrim)
{
  AttitudeFilter filter(1.0, 0.3);
  filter.update(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 9.80665), 0.01);
  const int before = filter.corrections();

  // Braking hard: the magnitude is nowhere near gravity, so this says nothing
  // about which way is down.
  for (int i = 0; i < 50; ++i) {
    filter.update(Eigen::Vector3d::Zero(), Eigen::Vector3d(-6.0, 0.0, 9.80665), 0.01);
  }

  EXPECT_EQ(filter.corrections(), before);
}

TEST(Attitude, TheTiltIsARotationIntoALevelFrame)
{
  AttitudeFilter filter(60.0, 0.3);
  filter.update(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 9.80665), 0.01);
  for (int i = 0; i < 100; ++i) {
    filter.update(
      Eigen::Vector3d(0.0, 0.05, 0.0), Eigen::Vector3d(0.0, 0.0, 9.80665), 0.01);
  }

  const Eigen::Matrix3d tilt = filter.body_tilt();

  // Orthonormal, and it carries the body vertical back towards world vertical.
  EXPECT_NEAR((tilt * tilt.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
  EXPECT_NEAR(tilt.determinant(), 1.0, 1e-12);
  const Eigen::Vector3d up = tilt.transpose() * Eigen::Vector3d(0.0, 0.0, 1.0);
  EXPECT_LT(up.z(), 1.0);
  EXPECT_GT(up.z(), 0.99);
}
