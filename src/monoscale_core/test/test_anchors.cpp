// The anchor map and the registration against it, from
// src/monoscale_odometry/test/test_tracks.py.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "monoscale_core/anchors.hpp"

using monoscale::AnchorSettings;
using monoscale::CameraTranslation;
using monoscale::GroundAnchorMap;
using monoscale::Identities;
using monoscale::Points2;
using monoscale::Weights;

namespace
{

Identities ids_of(std::initializer_list<int64_t> values)
{
  Identities ids(static_cast<Eigen::Index>(values.size()));
  Eigen::Index i = 0;
  for (int64_t value : values) {
    ids(i++) = value;
  }
  return ids;
}

Points2 points_of(std::initializer_list<std::pair<double, double>> values)
{
  Points2 points(static_cast<Eigen::Index>(values.size()), 2);
  Eigen::Index i = 0;
  for (const auto & value : values) {
    points(i, 0) = value.first;
    points(i, 1) = value.second;
    ++i;
  }
  return points;
}

void see(GroundAnchorMap & anchors, int64_t identity, double x, double y = 0.0)
{
  anchors.update(ids_of({identity}), points_of({{x, y}}), true, Weights());
}

}  // namespace

TEST(Anchors, AnchorIsTheRunningAverageOfItsObservations)
{
  AnchorSettings settings;
  settings.update_gain = 0.5;
  GroundAnchorMap anchors(settings);

  see(anchors, 1, 2.0);
  see(anchors, 1, 4.0);

  // Second sighting uses gain 1/2, landing halfway.
  const auto position = anchors.position_of(1);
  ASSERT_TRUE(position.has_value());
  EXPECT_NEAR(position->x(), 3.0, 1e-12);
  EXPECT_EQ(anchors.observations_of(1).value(), 2);
}

TEST(Anchors, UnseenIdsAreReportedUnanchored)
{
  GroundAnchorMap anchors;
  see(anchors, 7, 1.0, 1.0);

  monoscale::Mask anchored;
  anchors.anchored(ids_of({7, 8}), anchored);

  ASSERT_EQ(anchored.size(), 2);
  EXPECT_TRUE(anchored(0));
  EXPECT_FALSE(anchored(1));
}

TEST(Anchors, StaleAnchorsAreDropped)
{
  AnchorSettings settings;
  settings.max_age_frames = 2;
  GroundAnchorMap anchors(settings);
  see(anchors, 1, 0.0);
  for (int i = 0; i < 4; ++i) {
    see(anchors, 2, 5.0, 5.0);
  }

  EXPECT_FALSE(anchors.position_of(1).has_value());
  EXPECT_TRUE(anchors.position_of(2).has_value());
}

TEST(Anchors, MapIsCappedByDroppingTheLeastSeen)
{
  AnchorSettings settings;
  settings.max_anchors = 2;
  settings.max_age_frames = 1000;
  GroundAnchorMap anchors(settings);
  for (int64_t identifier : {1, 2}) {
    for (int i = 0; i < 5; ++i) {
      see(anchors, identifier, static_cast<double>(identifier));
    }
  }
  see(anchors, 3, 3.0);

  EXPECT_EQ(anchors.size(), 2);
  EXPECT_FALSE(anchors.position_of(3).has_value());
}

TEST(Anchors, AlignmentRecoversAPureTranslation)
{
  const Points2 body = points_of({{3.0, 1.0}, {5.0, -2.0}, {8.0, 0.5}, {4.0, 2.0}});
  Points2 world = body;
  world.col(0).array() += 10.0;
  world.col(1).array() -= 4.0;

  const auto result = monoscale::align_to_anchors(
    body, world, Weights::Ones(body.rows()), 0.0, 0.05, 3, false);

  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->translation.x(), 10.0, 1e-9);
  EXPECT_NEAR(result->translation.y(), -4.0, 1e-9);
  EXPECT_EQ(result->inliers.count(), body.rows());
  // A perfect fit has no spread, which is what makes it precise.
  EXPECT_NEAR(result->spread, 0.0, 1e-9);
}

TEST(Anchors, AlignmentUsesTheKnownYaw)
{
  const double yaw = 0.4;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const Points2 body = points_of({{3.0, 1.0}, {5.0, -2.0}, {8.0, 0.5}, {4.0, 2.0}});
  Points2 world(body.rows(), 2);
  for (Eigen::Index i = 0; i < body.rows(); ++i) {
    world(i, 0) = c * body(i, 0) - s * body(i, 1) + 2.0;
    world(i, 1) = s * body(i, 0) + c * body(i, 1) + 1.0;
  }

  const auto result = monoscale::align_to_anchors(
    body, world, Weights::Ones(body.rows()), yaw, 0.05, 3, false);

  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->translation.x(), 2.0, 1e-9);
  EXPECT_NEAR(result->translation.y(), 1.0, 1e-9);
}

TEST(Anchors, AMistrackedFeatureIsRejectedNotAveragedIn)
{
  const Points2 body =
    points_of({{3.0, 1.0}, {5.0, -2.0}, {8.0, 0.5}, {4.0, 2.0}, {6.0, 1.0}});
  Points2 world = body;
  world.col(0).array() += 1.0;
  // One feature claims the vehicle barely moved, the failure that used to win.
  world(4, 0) = body(4, 0) + 0.05;

  const auto result = monoscale::align_to_anchors(
    body, world, Weights::Ones(body.rows()), 0.0, 0.1, 3, false);

  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->translation.x(), 1.0, 1e-6);
  EXPECT_NEAR(result->translation.y(), 0.0, 1e-6);
  EXPECT_FALSE(result->inliers(4));
}

TEST(Anchors, LongLivedFeaturesOutweighNewOnes)
{
  const Points2 body = points_of({{3.0, 0.0}, {5.0, 0.0}, {7.0, 0.0}, {9.0, 0.0}});
  Points2 world = body;
  world.col(0).array() += 1.0;
  // Two fresh features are slightly off; the well seen ones should dominate.
  world(2, 0) += 0.04;
  world(3, 0) += 0.04;

  Weights weights(4);
  weights << 20.0, 20.0, 1.0, 1.0;

  const auto weighted =
    monoscale::align_to_anchors(body, world, weights, 0.0, 0.2, 3, false);
  const auto unweighted =
    monoscale::align_to_anchors(body, world, Weights::Ones(4), 0.0, 0.2, 3, false);

  ASSERT_TRUE(weighted.has_value());
  ASSERT_TRUE(unweighted.has_value());
  EXPECT_NEAR(unweighted->translation.x(), 1.02, 1e-6);
  EXPECT_NEAR(weighted->translation.x(), 1.0019, 1e-3);
}

TEST(Anchors, TooFewAgreeingPointsReturnsNothing)
{
  const Points2 body = points_of({{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}});
  const Points2 world = points_of({{9.0, 0.0}, {2.5, 3.0}, {-4.0, 1.0}});

  EXPECT_FALSE(
    monoscale::align_to_anchors(body, world, Weights::Ones(3), 0.0, 0.05, 3, false)
    .has_value());
}

TEST(Anchors, PrecisionFusionPrefersTheTighterCameraOverTheBiggerOne)
{
  // The rear sees far more ground but scatters; the front sees little and is
  // tight. Counting points alone hands it to the rear twelve to one.
  const std::vector<CameraTranslation> estimates{
    CameraTranslation{1.00, 0.0, 26, 0.01},
    CameraTranslation{1.20, 0.0, 300, 0.10}};

  const auto fused = monoscale::fuse_by_precision(estimates);

  ASSERT_TRUE(fused.has_value());
  EXPECT_NEAR(fused->x, 1.0, 0.05);
  EXPECT_LT(std::abs(fused->x - 1.0), 0.25 * std::abs(fused->x - 1.2));
  EXPECT_EQ(fused->count, 326);
}

TEST(Anchors, PrecisionFusionFallsBackToOneCamera)
{
  const auto single =
    monoscale::fuse_by_precision({CameraTranslation{2.0, 1.0, 50, 0.02}});
  ASSERT_TRUE(single.has_value());
  EXPECT_NEAR(single->x, 2.0, 1e-12);
  EXPECT_FALSE(monoscale::fuse_by_precision({}).has_value());
}

TEST(Anchors, AConsistentAnchorOutweighsAScatteringOne)
{
  AnchorSettings settings;
  settings.update_gain = 0.5;
  GroundAnchorMap anchors(settings);
  for (int i = 0; i < 8; ++i) {
    see(anchors, 1, 5.0);
  }
  // Same number of sightings, but this one never lands twice in one place.
  for (double offset : {0.0, 0.2, -0.2, 0.25, -0.25, 0.2, -0.2, 0.25}) {
    see(anchors, 2, 9.0 + offset);
  }

  const auto steady = anchors.weight_of(1);
  const auto jittery = anchors.weight_of(2);
  ASSERT_TRUE(steady.has_value());
  ASSERT_TRUE(jittery.has_value());
  EXPECT_GT(*steady, 5.0 * *jittery);
}

TEST(Anchors, AScatteringAnchorIsEventuallyDiscarded)
{
  AnchorSettings settings;
  settings.update_gain = 0.5;
  settings.max_variance = 0.02;
  settings.trial_observations = 3;
  GroundAnchorMap anchors(settings);
  for (double offset : {0.0, 0.6, -0.6, 0.6, -0.6, 0.6}) {
    see(anchors, 3, 2.0 + offset);
  }

  EXPECT_FALSE(anchors.position_of(3).has_value());
  EXPECT_GE(anchors.discarded(), 1);
}

TEST(Anchors, ASteadyAnchorSurvivesTheSameTreatment)
{
  AnchorSettings settings;
  settings.update_gain = 0.5;
  settings.max_variance = 0.02;
  settings.trial_observations = 3;
  GroundAnchorMap anchors(settings);
  for (int i = 0; i < 6; ++i) {
    see(anchors, 4, 2.0);
  }

  EXPECT_TRUE(anchors.position_of(4).has_value());
  EXPECT_EQ(anchors.discarded(), 0);
}

TEST(Anchors, CapacityPruningKeepsTheTrustworthyNotMerelyTheFrequent)
{
  AnchorSettings settings;
  settings.max_anchors = 1;
  settings.max_age_frames = 1000;
  settings.update_gain = 0.5;
  GroundAnchorMap anchors(settings);
  for (int i = 0; i < 10; ++i) {
    see(anchors, 5, 1.0);
  }
  for (double offset : {0.0, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3}) {
    see(anchors, 6, 4.0 + offset);
  }

  // The jittery one has been seen more often, and still loses.
  EXPECT_TRUE(anchors.position_of(5).has_value());
  EXPECT_FALSE(anchors.position_of(6).has_value());
}

TEST(Anchors, YawIsBelievedWhenRefinementIsOff)
{
  // The heading handed in stands, even when the ground disagrees with it. This
  // is the behaviour every measurement so far was taken under.
  const Points2 body =
    points_of({{3.0, 1.0}, {5.0, -2.0}, {8.0, 0.5}, {4.0, 2.0}, {6.0, -1.0}});
  const double true_yaw = 0.02;
  const double c = std::cos(true_yaw);
  const double s = std::sin(true_yaw);
  Points2 world(body.rows(), 2);
  for (Eigen::Index i = 0; i < body.rows(); ++i) {
    world(i, 0) = c * body(i, 0) - s * body(i, 1) + 10.0;
    world(i, 1) = s * body(i, 0) + c * body(i, 1) - 4.0;
  }

  const auto result = monoscale::align_to_anchors(
    body, world, Weights::Ones(body.rows()), 0.0, 0.5, 3, false);

  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result->yaw, 0.0);
}

TEST(Anchors, AHeadingErrorIsRecoveredWhenRefinementIsOn)
{
  // A gyro bias shows up exactly like this: the heading handed in is wrong by a
  // small constant and every anchor disagrees with it in the same rotational
  // sense. Solving translation alone absorbs that as a sideways offset, which
  // is what makes the bias leak into position.
  const Points2 body =
    points_of({{3.0, 1.0}, {5.0, -2.0}, {8.0, 0.5}, {4.0, 2.0}, {6.0, -1.0}});
  const double true_yaw = 0.02;
  const double c = std::cos(true_yaw);
  const double s = std::sin(true_yaw);
  Points2 world(body.rows(), 2);
  for (Eigen::Index i = 0; i < body.rows(); ++i) {
    world(i, 0) = c * body(i, 0) - s * body(i, 1) + 10.0;
    world(i, 1) = s * body(i, 0) + c * body(i, 1) - 4.0;
  }

  const auto result = monoscale::align_to_anchors(
    body, world, Weights::Ones(body.rows()), 0.0, 0.5, 3, true);

  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->yaw, true_yaw, 1e-6);
  EXPECT_NEAR(result->translation.x(), 10.0, 1e-5);
  EXPECT_NEAR(result->translation.y(), -4.0, 1e-5);
  EXPECT_EQ(result->inliers.count(), body.rows());
  // A fit this clean claims to be precise, and has to, or nothing downstream
  // can tell it apart from a loose one.
  EXPECT_LT(result->yaw_sigma, 1e-6);
}

TEST(Anchors, ALooserFitReportsALooserHeading)
{
  // Scattering the anchors leaves the same rotation recoverable on average and
  // much less worth believing, and only the second half of that shows up in the
  // number the solve returns for yaw.
  const int count = 60;
  Points2 body(count, 2);
  unsigned int state = 12345u;
  const auto next = [&state]() {
      state = state * 1103515245u + 12345u;
      return static_cast<double>((state >> 16) & 0x7fffu) / 32767.0;
    };
  for (int i = 0; i < count; ++i) {
    body(i, 0) = -8.0 + 16.0 * next();
    body(i, 1) = -8.0 + 16.0 * next();
  }

  const double true_yaw = 0.02;
  const double c = std::cos(true_yaw);
  const double s = std::sin(true_yaw);
  Points2 clean(count, 2);
  Points2 noisy(count, 2);
  for (int i = 0; i < count; ++i) {
    clean(i, 0) = c * body(i, 0) - s * body(i, 1) + 10.0;
    clean(i, 1) = s * body(i, 0) + c * body(i, 1) - 4.0;
    // Zero mean scatter of a few centimetres, deterministic.
    noisy(i, 0) = clean(i, 0) + 0.05 * (next() - 0.5) * 2.0;
    noisy(i, 1) = clean(i, 1) + 0.05 * (next() - 0.5) * 2.0;
  }

  const auto tight =
    monoscale::align_to_anchors(body, clean, Weights::Ones(count), 0.0, 0.5, 3, true);
  const auto loose =
    monoscale::align_to_anchors(body, noisy, Weights::Ones(count), 0.0, 0.5, 3, true);

  ASSERT_TRUE(tight.has_value());
  ASSERT_TRUE(loose.has_value());
  EXPECT_LT(tight->yaw_sigma, loose->yaw_sigma);
  EXPECT_NEAR(loose->yaw, true_yaw, 0.01);
}

TEST(Anchors, ANearSightingOutweighsAFarOne)
{
  // Precision, not count. The same angular error is worth eight times as many
  // centimetres at eight metres as at one, and averaging the two as equals
  // leaves the anchor displaced along its own bearing.
  AnchorSettings settings;
  settings.update_gain = 0.0;
  GroundAnchorMap anchors(settings);
  const Identities ids = ids_of({1});

  Weights far(1);
  far << 1.0 / 64.0;
  Weights near(1);
  near << 1.0;

  anchors.update(ids, points_of({{10.4, 0.0}}), true, far);
  anchors.update(ids, points_of({{10.0, 0.0}}), true, near);

  // Within a centimetre of the close sighting, not halfway between them.
  EXPECT_NEAR(anchors.position_of(1)->x(), 10.006, 0.01);
}

TEST(Anchors, EqualWeightsStillAverageEqually)
{
  AnchorSettings settings;
  settings.update_gain = 0.0;
  GroundAnchorMap anchors(settings);

  see(anchors, 1, 10.4);
  see(anchors, 1, 10.0);

  EXPECT_NEAR(anchors.position_of(1)->x(), 10.2, 1e-9);
}
