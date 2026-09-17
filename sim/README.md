# Recording conditions

What produced the benchmark's bags. **This directory is a copy of
`~/ros2_ws/hero-release/sim`** -- that tree is what actually runs, and this is
the record, so the conditions live in the repository. Taken 2026-09-12.

## Why it is in the repository

Two reasons.

**Reproduction.** The mount heights and rotations in
`vision_fisheye.param.yaml` are derived from `sensor_kit_calibration.yaml` here.
With that file outside the repository the provenance of those numbers existed on
one machine only. Correcting `actor_origin_height_m` from 0.0372 to 0.03635 on
2026-09-11 is exactly that class of change: the repository's config and the kit's
value have to move in the same commit, and they cannot if half the pair is
untracked.

**A test that was quietly not running.**
`monoscale_odometry/test/test_sensor_kit_consistency.py` reads the kit and holds
it against the odometry's extrinsics. Its search order puts
`<repo>/sim/ioniq_carla_bridge/config` ahead of this machine's absolute path.
Without a vendored copy it did not fail on another machine -- it skipped, and
took its guarantee with it.

## Drift

A copy can go stale. `test_sim_copy_matches_live.py` compares it against the
live tree where that tree exists and skips where it does not. **Edit both
sides.** If the change touches `sensor_kit_calibration.yaml`, move
`vision_fisheye.param.yaml` in the same commit.

## What is here

| | |
| --- | --- |
| `ioniq_carla_bridge/config/sensor_kit_calibration.yaml` | where each sensor attaches to the actor. The source of the odometry's extrinsics. |
| `ioniq_carla_bridge/config/sensor_mapping_fisheye_live.yaml` | the sensor kit the bench recorded with: resolution, FOV, `motion_blur_intensity: 0.0`, `shutter_speed: 200`. |
| `record_run.sh` | records one drive: brings up the bridge, assembles the fisheye, runs the pilot, writes the bag. |
| `rec_slalom.sh` `rec_straight.sh` `rec_rest.sh` | the recipe per drive type. |
| `reset_world.py` `set_sun.py` `spawn_obstacles.py` `wait_sim_seconds.py` | what `record_run.sh` calls. |

The scripts carry absolute paths into the live tree. That is deliberate: they
are a record of what ran, not a second copy to run, and the byte-for-byte test
above would fail if they were edited to be portable.

## Still outside the repository

Without these the recordings cannot be reproduced.

- **The CARLA server**: `./CarlaUE4.sh -quality-level=Low -carla-rpc-port=2000`.
  **`-quality-level=Low` is not optional.** Recorded at Epic, the same drive's
  length bias moves by a factor of sixty (a straight reads −6.826 % against
  +0.111 %), because the road texture is what the photometric alignment reads.
- **The PhysX substep**: `settings.max_substep_delta_time = 0.002` in
  `autoware_carla_interface/src/autoware_carla_interface/carla_autoware.py`.
  CARLA's default of 0.01 puts two 8.33 ms substeps inside a 1/60 s step and the
  reported angular velocity carries a turn-on bias of up to 0.9 deg/s. That was
  read as a gyro bias for weeks; it was neither the IMU nor the bridge.
  `monoscale_evaluation/bag_gate.py` checks a recorded bag for the trace.
- **The `ioniq_carla_bridge` package itself** (launch files, nodes). Only its
  configuration is vendored here.

## Recording a drive

```bash
cd ~/ros2_ws/hero-release/sim
./rec_slalom.sh <name> <seconds> <steer> [segment_seconds] [speed]
```

`rec_slalom.sh` sets its own `FISHEYE_ARGS` and appends whatever the caller
exported, so the fisheye assembler's parameters can be passed through:

```bash
FISHEYE_ARGS="-p interpolation:=cubic" ./rec_slalom.sh curve_s27_cubic 90 0.265 4.0 1.4
```

Until 2026-09-11 it overwrote instead of appending and the caller's arguments
were discarded silently.

The bag names and conditions of the benchmark set are in
`src/monoscale_evaluation/README.md`.
