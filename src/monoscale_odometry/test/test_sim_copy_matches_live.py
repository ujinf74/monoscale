"""The vendored recording recipe must not drift from the tree that runs.

`<repo>/sim` is a copy of `~/ros2_ws/hero-release/sim`, kept so the conditions
the bags were recorded under live in the repository -- and so
`test_sensor_kit_consistency` has a kit to read on a machine that is not this
one, where it used to `pytest.skip` and take its guarantee with it.

A copy can go stale, and a stale `sensor_kit_calibration.yaml` is the worst kind
of stale: the odometry's extrinsics are derived from it, so the tests would
agree with the copy while the recordings came out of different geometry. That is
the same shape as a launch file missing a parameter the benchmark had.

So: where the live tree exists, every vendored file must match it byte for byte.
Where it does not -- any machine but the one that records -- these skip, and the
vendored copy is simply what the repository knows.
"""
import hashlib
import os

import pytest

LIVE = '/home/i/ros2_ws/hero-release/sim'
VENDORED = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sim')


def _digest(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _pairs():
    root = os.path.normpath(VENDORED)
    for directory, _, names in os.walk(root):
        for name in sorted(names):
            if name == 'README.md':
                continue
            here = os.path.join(directory, name)
            yield os.path.relpath(here, root), here


@pytest.mark.parametrize('relative,vendored', list(_pairs()), ids=lambda v: v if isinstance(v, str) else '')
def test_the_vendored_copy_matches_the_live_tree(relative, vendored):
    if not os.path.isdir(LIVE):
        pytest.skip(f'no live tree at {LIVE}; the vendored copy is the record')
    live = os.path.join(LIVE, relative)
    assert os.path.exists(live), (
        f'{relative} is vendored but gone from {LIVE} -- either it was renamed '
        f'there and the copy was not updated, or it should be dropped here')
    assert _digest(vendored) == _digest(live), (
        f'{relative} has drifted from {LIVE}. The live tree is what records, so '
        f'copy it over -- and if the change touches sensor_kit_calibration.yaml, '
        f'move vision_fisheye.param.yaml in the same commit')


def test_the_kit_the_odometry_reads_is_the_vendored_one():
    """The search order matters, not just the file's presence.

    `test_sensor_kit_consistency._load_kit` tries the built package, then
    `src/ioniq_carla_bridge/config`, then `<repo>/sim/ioniq_carla_bridge/config`,
    then this machine's absolute path. The vendored copy has to come before the
    absolute path or a machine with both would check the wrong one.
    """
    vendored = os.path.normpath(
        os.path.join(VENDORED, 'ioniq_carla_bridge', 'config',
                     'sensor_kit_calibration.yaml'))
    assert os.path.exists(vendored), 'the kit is not vendored'

    source = open(
        os.path.join(os.path.dirname(__file__), 'test_sensor_kit_consistency.py'),
        'r').read()
    sim_at = source.index("'..', 'sim', KIT_PACKAGE")
    absolute_at = source.index('/home/i/ros2_ws/hero-release/sim')
    assert sim_at < absolute_at, (
        'the absolute path is searched before the vendored copy, so this '
        "machine would read the live kit and every other machine the copy")
