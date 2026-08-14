import math

import numpy as np
import pytest

from monoscale_odometry.tracks import align_to_anchors
from monoscale_odometry.tracks import GroundAnchorMap


def test_anchor_is_the_running_average_of_its_observations():
    anchors = GroundAnchorMap(update_gain=0.5)

    anchors.update(np.array([1]), np.array([[2.0, 0.0]]))
    anchors.update(np.array([1]), np.array([[4.0, 0.0]]))

    # Second sighting uses gain 1/2, landing halfway.
    assert anchors.world_points(np.array([1]))[0] == pytest.approx([3.0, 0.0])
    assert anchors.observations[1] == 2


def test_unseen_ids_are_reported_unanchored():
    anchors = GroundAnchorMap()
    anchors.update(np.array([7]), np.array([[1.0, 1.0]]))

    assert anchors.anchored(np.array([7, 8])).tolist() == [True, False]


def test_stale_anchors_are_dropped():
    anchors = GroundAnchorMap(max_age_frames=2)
    anchors.update(np.array([1]), np.array([[0.0, 0.0]]))

    for _ in range(4):
        anchors.update(np.array([2]), np.array([[5.0, 5.0]]))

    assert 1 not in anchors.positions
    assert 2 in anchors.positions


def test_map_is_capped_by_dropping_the_least_seen():
    anchors = GroundAnchorMap(max_anchors=2, max_age_frames=1000)
    for identifier in (1, 2):
        for _ in range(5):
            anchors.update(np.array([identifier]), np.array([[float(identifier), 0.0]]))
    anchors.update(np.array([3]), np.array([[3.0, 0.0]]))

    assert len(anchors) == 2
    assert 3 not in anchors.positions


def test_alignment_recovers_a_pure_translation():
    body = np.array([[3.0, 1.0], [5.0, -2.0], [8.0, 0.5], [4.0, 2.0]])
    world = body + np.array([10.0, -4.0])

    result = align_to_anchors(body, world, np.ones(len(body)), 0.0, 0.05, 3)

    assert result is not None
    translation, inliers, spread, _, _ = result
    assert translation == pytest.approx([10.0, -4.0])
    assert inliers.all()
    # A perfect fit has no spread, which is what makes it precise.
    assert spread == pytest.approx(0.0, abs=1e-9)


def test_alignment_uses_the_known_yaw():
    yaw = 0.4
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    body = np.array([[3.0, 1.0], [5.0, -2.0], [8.0, 0.5], [4.0, 2.0]])
    world = (rotation @ body.T).T + np.array([2.0, 1.0])

    result = align_to_anchors(body, world, np.ones(len(body)), yaw, 0.05, 3)

    assert result is not None
    assert result[0] == pytest.approx([2.0, 1.0])


def test_a_mistracked_feature_is_rejected_not_averaged_in():
    body = np.array([[3.0, 1.0], [5.0, -2.0], [8.0, 0.5], [4.0, 2.0], [6.0, 1.0]])
    world = body + np.array([1.0, 0.0])
    # One feature claims the vehicle barely moved, the failure that used to win.
    world[4] = body[4] + np.array([0.05, 0.0])

    result = align_to_anchors(body, world, np.ones(len(body)), 0.0, 0.1, 3)

    assert result is not None
    translation, inliers, _, _, _ = result
    assert translation == pytest.approx([1.0, 0.0], abs=1e-6)
    assert not inliers[4]


def test_long_lived_features_outweigh_new_ones():
    body = np.array([[3.0, 0.0], [5.0, 0.0], [7.0, 0.0], [9.0, 0.0]])
    world = body + np.array([1.0, 0.0])
    # Two fresh features are slightly off; the well seen ones should dominate.
    world[2:] += np.array([0.04, 0.0])
    weights = np.array([20.0, 20.0, 1.0, 1.0])

    weighted = align_to_anchors(body, world, weights, 0.0, 0.2, 3)
    unweighted = align_to_anchors(body, world, np.ones(4), 0.0, 0.2, 3)

    assert weighted is not None and unweighted is not None
    # Unweighted the two fresh outliers pull the answer to 1.02; weighted they
    # barely register.
    assert unweighted[0][0] == pytest.approx(1.02, abs=1e-6)
    assert weighted[0][0] == pytest.approx(1.0019, abs=1e-3)


def test_too_few_agreeing_points_returns_nothing():
    body = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    world = np.array([[9.0, 0.0], [2.5, 3.0], [-4.0, 1.0]])

    assert align_to_anchors(body, world, np.ones(3), 0.0, 0.05, 3) is None


def test_precision_fusion_prefers_the_tighter_camera_over_the_bigger_one():
    from monoscale_odometry.tracks import fuse_by_precision

    # The rear sees far more ground but scatters; the front sees little and is
    # tight. Counting points alone hands it to the rear twelve to one.
    tight_small = (1.00, 0.0, 26, 0.01)
    loose_large = (1.20, 0.0, 300, 0.10)

    fused = fuse_by_precision([tight_small, loose_large])

    # Lands next to the tight camera, not the numerous one.
    assert fused[0] == pytest.approx(1.0, abs=0.05)
    assert abs(fused[0] - 1.0) < 0.25 * abs(fused[0] - 1.2)
    assert fused[2] == 326


def test_precision_fusion_falls_back_to_one_camera():
    from monoscale_odometry.tracks import fuse_by_precision

    assert fuse_by_precision([(2.0, 1.0, 50, 0.02)])[0] == pytest.approx(2.0)
    assert fuse_by_precision([]) is None


def test_a_consistent_anchor_outweighs_a_scattering_one():
    anchors = GroundAnchorMap(update_gain=0.5)
    for _ in range(8):
        anchors.update(np.array([1]), np.array([[5.0, 0.0]]))
    # Same number of sightings, but this one never lands twice in one place.
    for offset in (0.0, 0.2, -0.2, 0.25, -0.25, 0.2, -0.2, 0.25):
        anchors.update(np.array([2]), np.array([[9.0 + offset, 0.0]]))

    steady, jittery = anchors.weights(np.array([1, 2]))

    assert steady > 5.0 * jittery


def test_a_scattering_anchor_is_eventually_discarded():
    anchors = GroundAnchorMap(update_gain=0.5, max_variance=0.02, trial_observations=3)
    for offset in (0.0, 0.6, -0.6, 0.6, -0.6, 0.6):
        anchors.update(np.array([3]), np.array([[2.0 + offset, 0.0]]))

    assert 3 not in anchors.positions
    assert anchors.discarded >= 1


def test_a_steady_anchor_survives_the_same_treatment():
    anchors = GroundAnchorMap(update_gain=0.5, max_variance=0.02, trial_observations=3)
    for _ in range(6):
        anchors.update(np.array([4]), np.array([[2.0, 0.0]]))

    assert 4 in anchors.positions
    assert anchors.discarded == 0


def test_capacity_pruning_keeps_the_trustworthy_not_merely_the_frequent():
    anchors = GroundAnchorMap(max_anchors=1, max_age_frames=1000, update_gain=0.5)
    for _ in range(10):
        anchors.update(np.array([5]), np.array([[1.0, 0.0]]))
    for index, offset in enumerate((0.0, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3,
                                    -0.3, 0.3, -0.3, 0.3)):
        anchors.update(np.array([6]), np.array([[4.0 + offset, 0.0]]))

    # The jittery one has been seen more often, and still loses.
    assert 5 in anchors.positions
    assert 6 not in anchors.positions


def test_yaw_is_believed_when_refinement_is_off():
    """The heading handed in stands, even when the ground disagrees with it.

    This is the behaviour every measurement so far was taken under, so it has
    to keep working: refinement is off by default and the solve must not
    quietly start moving yaw.
    """
    body = np.array([[3.0, 1.0], [5.0, -2.0], [8.0, 0.5], [4.0, 2.0], [6.0, -1.0]])
    true_yaw = 0.02
    c, s = np.cos(true_yaw), np.sin(true_yaw)
    world = (np.array([[c, -s], [s, c]]) @ body.T).T + np.array([10.0, -4.0])

    result = align_to_anchors(body, world, np.ones(len(body)), 0.0, 0.5, 3)

    assert result is not None
    assert result[3] == pytest.approx(0.0)


def test_a_heading_error_is_recovered_when_refinement_is_on():
    """With refinement the solve finds the heading the ground implies.

    A gyro bias shows up exactly like this: the heading handed in is wrong by
    a small constant and every anchor disagrees with it in the same rotational
    sense. Solving translation alone cannot see that -- it absorbs the
    disagreement as a sideways offset, which is what makes the bias leak into
    position.
    """
    body = np.array([[3.0, 1.0], [5.0, -2.0], [8.0, 0.5], [4.0, 2.0], [6.0, -1.0]])
    true_yaw = 0.02
    c, s = np.cos(true_yaw), np.sin(true_yaw)
    world = (np.array([[c, -s], [s, c]]) @ body.T).T + np.array([10.0, -4.0])

    result = align_to_anchors(
        body, world, np.ones(len(body)), 0.0, 0.5, 3, refine_yaw=True)

    assert result is not None
    translation, inliers, _, yaw, sigma = result
    assert yaw == pytest.approx(true_yaw, abs=1e-6)
    assert translation == pytest.approx([10.0, -4.0], abs=1e-5)
    assert inliers.all()
    # A fit this clean claims to be precise, and has to, or nothing
    # downstream can tell it apart from a loose one.
    assert sigma < 1e-6


def test_a_looser_fit_reports_a_looser_heading():
    """The reported uncertainty is what the caller weighs, so it has to move.

    Scattering the anchors leaves the same rotation recoverable on average and
    much less worth believing, and only the second half of that shows up in
    the number the solve returns for yaw.
    """
    rng = np.random.default_rng(0)
    body = rng.uniform(-8.0, 8.0, size=(60, 2))
    true_yaw = 0.02
    c, s = np.cos(true_yaw), np.sin(true_yaw)
    clean = (np.array([[c, -s], [s, c]]) @ body.T).T + np.array([10.0, -4.0])
    noisy = clean + rng.normal(scale=0.05, size=clean.shape)

    tight = align_to_anchors(
        body, clean, np.ones(len(body)), 0.0, 0.5, 3, refine_yaw=True)
    loose = align_to_anchors(
        body, noisy, np.ones(len(body)), 0.0, 0.5, 3, refine_yaw=True)

    assert tight is not None and loose is not None
    assert tight[4] < loose[4]
    assert loose[3] == pytest.approx(true_yaw, abs=0.01)


def test_a_near_sighting_outweighs_a_far_one():
    """Precision, not count. The same angular error is worth eight times as
    many centimetres at eight metres as at one, and averaging the two as
    equals leaves the anchor displaced along its own bearing. Over a frame of
    features that displacement reads as a rotation.
    """
    anchors = GroundAnchorMap(update_gain=0.0)
    ids = np.array([1])

    # Seen far away first, and wrongly; then close up, and correctly.
    anchors.update(ids, np.array([[10.4, 0.0]]), information=np.array([1.0 / 64.0]))
    anchors.update(ids, np.array([[10.0, 0.0]]), information=np.array([1.0 / 1.0]))

    # Within a centimetre of the close sighting, not halfway between them.
    assert anchors.positions[1][0] == pytest.approx(10.006, abs=0.01)


def test_equal_weights_still_average_equally():
    """The old behaviour is what callers without weights still get."""
    anchors = GroundAnchorMap(update_gain=0.0)
    ids = np.array([1])

    anchors.update(ids, np.array([[10.4, 0.0]]))
    anchors.update(ids, np.array([[10.0, 0.0]]))

    assert anchors.positions[1][0] == pytest.approx(10.2, abs=1e-9)
