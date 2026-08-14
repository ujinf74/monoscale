# monoscale_odometry

Vision-only odometry and occupancy mapping on CARLA, scored against the
simulator ground truth. This is `vision_only_autopark` moved off Isaac Sim,
with the parts that made the Isaac run hard to trust replaced by measurement.

The vehicle and sensor kit live in `ioniq_carla_bridge`, which holds the
canonical copy of the Isaac rig. This package adds estimation and measurement
only, and brings that kit up itself.

No controller, no vehicle interface, no trajectory follower. The stack
estimates, maps and reports, which is the coaching-only direction the HERO
schedule settled on.

The current odometry-only architecture, two-fisheye-plus-IMU input contract,
C++ tracker workflow, velocity filtering, Orin budget and reproducible
parameter sweep are documented in
[`docs/오도메트리_인수인계.md`](../../docs/오도메트리_인수인계.md). The high-resolution `approach_fish8`
benchmark is in the 0.034--0.040 m ATE RMSE range. The clean 30 Hz camera /
60 Hz IMU release bag currently measures 0.108 m: acceleration improves its
own OFF baseline of 0.175 m, but direct assembly at 640 loses accuracy versus
assembling at higher resolution and resizing in the tracker. The target below
0.03 m has not yet been met.

## What changed from the Isaac package

| | Isaac | CARLA |
| --- | --- | --- |
| camera intrinsics | `CameraInfo` was all zeros, so K came from the USD | `CameraInfo` is valid and used directly |
| image encoding | `rgb8` | `bgra8` |
| distortion | `equidistant`, zero coefficients | `plumb_bob`, zero coefficients, skipped outright |
| IMU | 60 Hz | 60 Hz in the clean release kit |
| cameras | 60 Hz, 1920x1080 | four 1280x1280 pinhole sources at 30 Hz, assembled into front/rear fisheyes |
| ground truth | none | exact ego transform, and the reason for the move |

The camera extrinsics did not change. The CARLA sensor kit reproduces the
Isaac rig, and deriving the base_link-from-optical rotation from the kit poses
reproduces the Isaac matrices exactly, so `rotation_base_from_camera` and
`translation_base_from_camera` carry over unchanged.

The assembler still declares 640x360 defaults. That size is sufficient for
the occupancy mapper, but it is not the odometry deployment recommendation:
keep the assembled image at least 1280x720 and let the C++ tracker resize it to
its 640-pixel processing width. See the clean60 resolution comparison in the
handoff document before changing this contract.

The `fov: 93.695221` in the sensor kit is not arbitrary: it is the Isaac lens,
`fx = fy = 900` at 1920 px, and a test derives the extrinsics from the kit
poses and fails if the two configurations ever drift apart.

OpenVINS and its adapter were not carried over.

## Nodes

- `odometry_node` — tracks features in the non-overlapping front and
  rear images, projects ground features through the calibrated camera poses,
  fixes inter-frame yaw to the IMU, and fuses both cameras by inlier count.
  Publishes `/localization/kinematic_state`, `/current_pose`, `map -> base_link`
  and `/vision/occupancy_grid`.
- `odometry_evaluator` — anchors the CARLA ground truth to the estimate at the
  first matched sample and reports absolute drift and segment-wise relative
  error. Anchoring only, no best-fit alignment, so what it prints is accumulated
  drift rather than a fitting residual.
- `keyboard_teleop` — drives the ego from a terminal, for watching the stack
  live rather than scoring it.
- `carla_ground_truth` — publishes the exact ego transform for scoring, read
  from the CARLA API rather than from a sensor.
- `grid_evaluator` — accumulates the LiDAR reference grid along the path and
  reports precision, recall and IoU of the vision occupancy grid against it.
  Needs `launch_lidar_reference:=true`.
- `ego_pilot` — drives the ego through a fixed manoeuvre over the actuation and
  gear topics the CARLA bridge already subscribes to, so two runs are
  comparable. Off by default.

## Ground truth

`carla_ground_truth` reads the ego transform straight from the CARLA API and
publishes it on `/carla/ground_truth/pose`, once per simulation frame. It never
ticks the world; the bridge owns the clock.

Earlier this came from a noise-free GNSS pseudo sensor added to the kit, which
is how the bridge exposes the ego transform. That worked but put a sensor in
the kit that only existed to be ground truth, so the kit no longer matched the
Isaac rig. Reading the API directly keeps the kit clean.

The pose is the vehicle centre, and the evaluator shifts it back half a
wheelbase to the rear axle before comparing.

Reported metrics: sample count, ground truth distance, scale ratio over
segments as well as over the whole path, ATE RMSE
and maximum, final position error, drift as a percentage of distance travelled,
heading RMSE, and relative pose error per segment.

With `output_directory` set, the evaluator writes `estimate.tum`, `truth.tum`
and `summary.txt` on shutdown. The TUM files feed
[evo](https://github.com/MichaelGrupp/evo) directly:

```bash
evo_traj tum estimate.tum --ref truth.tum -p --plot_mode xy
evo_ape tum truth.tum estimate.tum -p
```

## Run

Start CARLA:

```bash
cd /home/i/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Low
```

Build and run the original all-in-one Python-frontend path:

```bash
cd /home/i/ros2_ws
colcon build --symlink-install --packages-select monoscale_odometry
source /opt/ros/humble/setup.bash
source /home/i/autoware_ws/install/setup.bash
source /home/i/ros2_ws/install/setup.bash
ros2 launch monoscale_odometry odometry.launch.py
```

The deployment-oriented odometry path uses the separate C++ tracker. It is not
yet the default of the launch above; build both packages and follow the exact
launch and benchmark commands in the odometry handoff document when comparing
accuracy or estimating the Orin budget.

A scored run that drives itself and writes trajectories:

```bash
ros2 launch monoscale_odometry odometry.launch.py \
  launch_pilot:=true output_directory:=/tmp/carla_vision_eval
```

Attach only the vision stack to a bridge someone else already started:

```bash
ros2 launch monoscale_odometry vision_estimation.launch.py
```

Add the LiDAR reference grid, to see what the cameras missed:

```bash
ros2 launch monoscale_odometry odometry.launch.py \
  launch_lidar_reference:=true
```

Enable the display in RViz; it is off by default.

## Driving it yourself

`ego_pilot` repeats a fixed manoeuvre, which is what comparisons need. To drive
by hand and watch RViz, run the teleop in a terminal of its own. It cannot go
in the launch file, because a launched node has no terminal to read keys from.

```bash
ros2 run monoscale_odometry keyboard_teleop
```

`f` for drive, `w` to add throttle, `a` and `d` to steer, `c` to centre the
wheel, space to stop, `r` for reverse. Settings hold until changed: a terminal
reports keys pressed, not keys held, so anything that decayed on its own would
need the key hammered to move at all.

CARLA's own `PythonAPI/examples/manual_control.py` cannot be used here. It
always spawns its own vehicle, which is not the ego the sensors are attached
to. `use_traffic_manager:=True` adds NPC traffic but does not drive the ego
either.

## What the move to CARLA already found

Nothing here was visible before the estimate could be compared against truth.

**The bridge stamped the ego pose from a different clock than the sensors.**
`autoware_carla_interface` stamped camera and IMU messages with the CARLA
measurement time but `/clock`, the ego pose and the vehicle status with a clock
counting from bridge start, leaving them 400 to 700 s apart. Fixed at the
source: `carla_ros.py` now takes `GameTime.get_carla_time()`. The evaluator
keeps a workaround for older bridges, off when the offset is zero.

**The camera frame arrived before the IMU sample covering it,** so every pair
was tracked against a stale IMU and then thrown away. The tracker now waits for
the IMU to reach the frame time and retries from the IMU callback.

**Tracking blocks the node for tens of milliseconds** and the default queue
depth dropped the IMU samples that arrived meanwhile, halving the effective
rate. The IMU queue is now 200 deep.

**Rejected pairs discarded the IMU rotation with the vision translation,**
leaking heading that never came back. Rejected pairs now still apply the yaw.

**The camera height was the mounting height, not the height above the road.**
CARLA holds the actor origin 0.0372 m off the ground, and a ground plane solve
scales linearly with that. It read 3 % short until corrected.

**The publisher throttle beat against the sensor tick,** dropping every other
frame and taking the cameras to 8 Hz. Setting the throttle above the rate the
sensor produces, and the tick to a multiple of the simulation step, restored
17 to 18 Hz at full 1920x1080. This was first misdiagnosed as the renderer
being starved by 1080p, which measurement later disproved.

**rclpy cannot ingest two 1920x1080 frames per cycle.** The estimator fell from
17 to 9 pairs per second on the full resolution stream and the odometry
degraded with it. Prescaling inside the node did not help, because the cost
sits above it, in turning an 8.3 MB message into Python objects.

Two ways around it, both supported and both measured over the same run:

| `image_source` | sensor kit | pairs | drift | ATE RMSE | grid IoU |
| --- | --- | --- | --- | --- | --- |
| `tap` (default) | 6 sensors, adds a 960x540 pair at the same poses | 665 | 1.14 % | 0.36 m | 0.22 |
| `resize` | 4 sensors, exactly the Isaac rig, `image_proc` downscales | 557 | 1.11 % | 0.30 m | 0.20 |

Accuracy is indistinguishable; the difference is throughput. The resize node
emitted 13.8 Hz from a 15 to 16 Hz source, and that 15 % shows up as fewer
pairs and slightly thinner map coverage. Use `resize` with
`sensor_mapping_file:=sensor_mapping.yaml` when the kit has to be exactly the
Isaac rig, and `tap` when the run is about measurement.

On a 34 m scripted manoeuvre, rejected pairs went from 100 of 778 to 0 of 969,
heading RMSE from 5.4 deg to 0.03 deg, and drift from 9.5 % to between 0.5 and
2.6 %, depending on the run. Segment scale sits within 1 % of unity.

## Occupancy map

The map is scored against an Autoware occupancy grid built from a CARLA LiDAR,
which is what `launch_lidar_reference:=true` is for. Scoring a single reference
snapshot is misleading, because obstacles the vision correctly remembers from
earlier count as false positives; the evaluator accumulates the reference along
the path instead and compares once, over cells within 15 m of where the vehicle
actually drove.

Three changes came out of that measurement.

- **Triangulate against a keyframe, not the previous frame.** Consecutive
  frames are centimetres apart at parking speed, which is no baseline at all.
  The current view is now triangulated against a keyframe held until the
  vehicle has moved `mapping_baseline_m`, which makes a one degree parallax
  gate meaningful rather than fatal.
- **Obstacle rays no longer carve free space.** Depth from a short baseline is
  not good enough to declare everything in front of an obstacle empty, and the
  carving was erasing real obstacles. Free space comes from ground inliers,
  which are reliable.
- **One good triangulation marks a cell.** Demanding a second hit on the same
  10 cm cell cost most of the recall.

Measured, occupied cells within 15 m of the path:

| | precision | recall | IoU |
| --- | --- | --- | --- |
| before | 0.62 | 0.10 | 0.10 |
| after | 0.25 to 0.28 | 0.30 | 0.16 to 0.17 |
| after the kit merge | 0.41 | 0.32 | 0.22 |

The last row is the same mapper on the canonical kit, with the bridge clock
fixed and the ego transform read from the API. A better pose smears the map
less, so measurement quality and map quality are not independent.

That is a deliberate trade. `occupied_probability_threshold` is the knob:
0.65 takes one triangulation, 0.85 demands two and returns to the precise but
nearly blind behaviour. For a system that warns a driver, missing an obstacle
is worse than mentioning one that is not there, so it ships at 0.65.

The map is still sparse. Recall is capped by how many trackable features sit on
obstacles at all, and a dense map needs a metric video-depth or multi-view
stereo backend, not more tuning of this one.

## Two things to know before trusting a number

**Runs are not comparable unless the spawn point is fixed.** It now defaults to
a straight stretch of Town10HD_Opt for that reason. Pass `spawn_point:=None` to
spawn randomly, or pick another point with:

```python
import carla
points = carla.Client('localhost', 2000).get_world().get_map().get_spawn_points()
print(points[0])
```

**One run is not a measurement.** With the spawn point fixed, drift over the
same manoeuvre still varied between 0.5 % and 2.6 % across runs, because the
pilot steers open loop and the simulator does not repeat exactly. Compare
configurations over several runs, not one.

## Known limits

- Planar vision-inertial odometry has globally unobservable position and yaw
  drift. Absolute correction needs loop closure or a prior map.
- The estimator assumes a flat ground plane. Town04 has banked and sloped
  sections; pick a flat area or a flatter town when the numbers matter.
- The occupancy map is sparse and local. Textureless surfaces and low-parallax
  motion stay unknown, which is what the LiDAR reference grid is there to show.
- `ego_pilot` is open loop in steering and only closes the loop on speed, so
  the path repeats only as well as the simulator does.
