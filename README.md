# monoscale

**Metric visual-inertial odometry whose scale comes from the ground plane**, and
an occupancy grid built from the same images. No stereo, no lidar, no learned
model, no prior map. One camera is enough, N is allowed, and they are free not
to overlap -- the two on this vehicle face opposite ways and share no field of
view at all.

It has been measured in two places, and they are far enough apart that one
number would misdescribe both. Same code and same definitions throughout --
`monoscale_evaluation/benchmark.py`, tuned figure first, held-out second:

| | CARLA, 29-150 m | Ford Multi-AV, 5.2 km |
| --- | ---: | ---: |
| **ATE / distance** | **0.0237 %** / **0.1781 %** | 0.510 % / **0.297 %** |
| final error / distance | 0.0324 % / 0.2659 % | 0.558 % / 0.326 % |
| `hop %`, what one observation is worth | 0.14 / 0.83 | 4.14 / 3.72 |
| `walk`, accumulation over random walk | 1.10 / 0.76 | 4.82 / **1.67** |
| road visible to the front camera | 49.6 deg | 11.8 deg |

In millimetres, because a percentage of distance is hard to feel: 26 mm of ATE
over a 111 m simulated drive, and 15.6 m over a 5.2 km real one. Neither number
is the other's failure -- the bottom row is most of why they differ, and the
rest of that argument is in
[what it assumes](#what-it-assumes-and-where-it-has-not-been-tested).

What transferred from the simulator to the vehicle is `walk`, which is
accumulation and is the hard half of odometry: 1.67 held out against 1.10 and
0.76. What did not is `hop %`, the worth of a single ground observation, and
that follows the angle in the bottom row rather than anything in the code.

![Estimate against ground truth](media/trajectories.png)

The CARLA slalom and the held-out park manoeuvre. At the scale of the drive the
two lines are one; the right-hand panel zooms until the gap is visible and puts
a scale bar beside it, because a figure where nothing can be seen proves
nothing. Regenerate it with `monoscale_evaluation/plot_trajectories.py
<slalom_dir> <park_dir> <out.png>` from any pair of `--tum` outputs.

The held-out figures are the ones to read. In CARLA two parking drives are kept
out of every sweep and every judgement, and they earn their place: in one week
they refused four changes that had won on the tuning set, including the removal
of the last fitted multiplier in the stack. On Ford, Log6 plays the same part and
no parameter was ever chosen against it. Numbers, the set they come from and how
to reproduce them are in
[`src/monoscale_evaluation/README.md`](src/monoscale_evaluation/README.md); the
Ford harness is in the untracked `tools/`, so those two figures are recorded
here rather than reproducible from a clone.

## What it is for

The rig is two fisheye cameras, front and rear, pitched 30 degrees down. That is
the surround-view configuration a production car already carries for parking,
and the point of taking scale from the ground plane is that **nothing has to be
added to the vehicle to make those cameras metric** -- no stereo pair, no lidar,
no HD map, no wheel encoder on the bus.

Where that matters is where GNSS does not reach and the vehicle is moving
slowly: indoor parking structures, loading yards, the last thirty metres of an
automated park. The two drives held out of every decision here are parking
manoeuvres for that reason. The accuracy of this method scales as the camera
height over the distance travelled per frame, so low and slow is the regime it
is best in, which is the opposite of what a stereo baseline prefers.

## Where the scale comes from, and what follows from it

Monocular vision has no scale. The usual sources are a second camera with a
known baseline, a lidar, or an IMU excited enough to observe it. This takes it
from the one distance a road vehicle already knows to a millimetre: **the height
of the camera above the road**. A plane-induced homography between two frames
recovers `t/h`, and `h` is the mounting height, so the answer is metric without
anything else in the vehicle measuring length.

Two properties follow, and neither is a design goal -- they are what is left
over once the scale does not come from a baseline.

**The cameras do not have to overlap, and here they cannot.** The two optical
axes point in opposite directions and each carries a 139.47 deg horizontal
field, so the front spans bearings −69.74 to +69.74 and the rear 110.27 to
249.74. That leaves **40.53 deg of gap on each side**, computable from the
configuration and not a claim: there is no shared feature anywhere, and no
baseline to triangulate on. What the second camera buys is not stereo -- it is a
nuisance that the two read with opposite sign, so averaging them cancels it.

**One camera is a rig, not a degenerate case.** With the disagreement between
two of them gone, `single_camera_variance` carries that weight instead. It
solves every drive in both sets:

| | bench mean / worst | held-out mean / worst |
| --- | ---: | ---: |
| front + rear | 0.0237 % / 0.0391 % | 0.1781 % / 0.2659 % |
| front only | 0.0621 % / 0.1265 % | 0.5689 % / 1.0588 % |
| rear only | 0.1474 % / 0.4529 % | 0.5694 % / 0.7701 % |

So one camera costs a factor of 2.6 to 6.2 and does not fall over, and the
front mount -- the lower one, which sees the ground it is about to drive over --
is the better of the two to keep. A unit test holds the floor:
`EstimatorDrive.OneCameraIsARigNotADegenerateCase`. What must not happen is that
it silently stops solving.

## What it assumes, and where it has not been tested

**The road under the cameras is locally planar.** The whole scale chain rests on
it: the homography is plane-induced, and a surface that is not a plane is model
error the four-parameter fit has to absorb. On these roads it holds to under a
millimetre over a hundred metres. On a crest, a rutted surface or a steep
camber it is the first thing that would break, and nothing here has measured
that.

**The mounting height has to be right to a millimetre.** The scale is `t/h`, so
a relative error in `h` is the same relative error in every distance: correcting
one millimetre of it moved the benchmark by half. A real vehicle's ride height
moves with load, tyre pressure and suspension travel, and none of that exists in
these recordings.

**The numbers above are measured in simulation.** CARLA, one town, one weather,
one sun angle, `-quality-level=Low`. Two facts follow that a reader should have
before the numbers. The rendered road's texture is what the photometric
alignment reads, and changing the quality level alone moves the length bias by a
factor of sixty. And **the simulated IMU carries zero noise on every axis** --
accelerometer, gyro and gyro bias all set to 0.0 -- so the attitude filter, the
heading and the inertial propagation have never met a noisy one.

The heading itself is self-contained, and that is worth saying because it is the
easy thing to fake in a simulator: `imu_yaw_from_gyro` is on, so the yaw is
integrated from the gyro and its bias is corrected by the road fit's own yaw
observation. The orientation CARLA reports -- which is truth to 0.000000 deg on
these bags -- is not read. What has not been tested is that integration against
a gyro with real angle random walk.

**The real-vehicle column is one car and two drives.** Ford Multi-AV Seasonal,
2017-10-26, V2, Log5 and Log6, 5.2 km each. Both needed their calibration
repaired before they would score at all -- Log5's recorded IMU orientation is
88309 all-zero quaternions, recoverable only from a second topic, and its mount
pitch was out by 0.154 degrees -- and no third drive with a rear camera exists
to check the pair against. KITTI cannot be that third witness: it has no rear
camera, and putting Ford in the same condition accounts for most of the gap
between the two benchmarks.

They also are not one regime. Log5 runs at 13.7 m/s median against Log6's 7.4,
its hops are 1.99 m against 1.18, and it is the worse of the two on `walk` by a
factor of 2.9 -- in the direction this method's own rule predicts, since the
accuracy goes as camera height over distance travelled per frame. One number
cannot describe it. A specification would have to read "this much at this speed
with this much road in view".

The geometry claim above was taken apart before it was made, because it is the
easy thing to assert. Cutting the per-hop noise by a sixth -- carrying the
inertial attitude between anchor fixes takes the fused hop noise from 10.78 % to
8.91 % and the accumulating common mode from 18.6 % to 14.0 % -- does not move
the segment error at all. Neither does rescaling the whole trajectory by its best
constant, worth 0.30 pp on one drive and nothing on the other. Neither does
letting the anchor map answer 2.2x as many hops. What is left is a *local* scale
error, 6.2 % and 6.6 % rms over 100 m windows while its mean is +1.4 % and
+0.2 %: the mean is right and each stretch is wrong. Grade, speed, curvature, hop
count, hop length and map share each take under 6 % of its variance. The mean
range of the band takes 6 and 10 %, and the pitch gain 9 and 18 %, with the same
sign on both drives -- a scale that depends on which range answered is a plane
whose tilt and height were never separated, and over a 6-30 m band those two
bases correlate 0.987.

The quantity that governs this is the angular extent of road the camera sees,

    extent = atan(h / d_near) - atan(h / d_far)

and it is the near edge that carries it: on this vehicle 30 m to 40 m is worth
0.76 deg and 6 m to 4 m is worth 6.85 deg. Pushing the near edge in makes both
drives worse and the held-out one sharply (2.33 % at 6.0 m, 3.24 % at 5.5 m,
4.03 % at 5.0 m), because that is where the bodywork is. Twenty degrees is
where the noise is still near one per cent, ten is the knee, and a camera that
cannot reach twenty is short of geometry rather than of tuning.

So the claim that column supports is the modest one: on a vehicle the method was
not designed around, with a real IMU on real asphalt at up to 24 m/s, it ran and
it landed where its mounting said it would. What it does not support is a number
for a product. Two facts stand in the way and neither is measurement noise: the
calibration is fragile, in that perturbing the mount pitch by 0.004 degrees --
a fortieth of the correction this vehicle actually needed -- swings ATE by tens
of per cent, while the same vehicle's pitch moved 0.154 degrees over three
months; and about 0.6 % of residual scale is a rig constant worth 0.96 cm of
camera height that the lidars, the accelerometer and the truth trajectory each
fail to determine.

The single measurement worth most next is not a parameter. It is one drive from
a rear-equipped vehicle whose camera can see road inside six metres -- a bumper
or mirror mount rather than a roof one. That tests the claim in the row above
better than any further tuning of this rig can.

None of that is a reason to discount the method. It is the list of things that
would have to be measured again on a vehicle, and it is written down here rather
than left for a reader to find.

## How it is built

Every constant in `vision_fisheye.param.yaml` carries the measurement that chose
it, and the measurements that rejected the alternatives. That is why the config
is 90 % comment: a value without its evidence is a value nobody can change
safely later. Ideas that were tried and lost are recorded with their numbers so
they are not tried again, and several of them were mine.

Two rules do most of the work. **Measure the quantity, do not fit the score** --
the scale correction that reads 0.1 % is derived from a millimetre of frame
geometry, not swept against ATE. And **held-out decides** -- a change that wins
on the bench and loses on the two park drives is an artefact of the bench.

Everything that runs on the vehicle is C++. The stack was developed against a
Python estimator, which was removed from the tree once the trajectories matched;
what was carried across and what was not is in
[`src/monoscale_core/README.md`](src/monoscale_core/README.md).

## Packages

| package | what it does |
| --- | --- |
| `monoscale_core` | the estimation itself. Knows nothing of ROS, so it is tested without a graph and can later share a process with a controller. |
| `monoscale_tracker` | the C++ KLT front end. Turns images into feature tracks and publishes those alone. |
| `monoscale_odometry` | the node around it, and `monoscale_replay`, which scores a bag directly. |
| `monoscale_occupancy_grid_map` | plane-sweeps the raw fisheye into an occupancy grid. CUDA for real time. |
| `monoscale_evaluation` | scoring against truth. Not deployed. |
| `monoscale_carla` | CARLA only: camera assembly, the truth tap, the drive scripts. |

## Topics

```
images ─┬─→ monoscale_tracker  →  /vision/tracks/<camera>
        │                          └→ monoscale_odometry
        │                                 ├→ /localization/kinematic_state
        │                                 └→ /perception/ground_points
        └────────────────────────→ monoscale_occupancy_grid_map
                  + camera_info               └→ /perception/occupancy_grid_map
                  + /localization/kinematic_state
```

The two branches split at the image. Odometry sees only feature tracks; the grid
looks at the raw frames again.

The grid is a **plane sweep**: world-horizontal planes are stacked from −0.15 m
in 0.05 m steps, each pixel is scored by ZNCC for how well two views agree at
each height, and SGM aggregates that into the surface height for that pixel. The
height is therefore a decision, not a by-product of triangulating features.
Attitude comes straight out of the subscribed odometry and enters the per-pixel
warp.

An earlier version accumulated the odometry's `/perception/ground_points` into
the grid. That topic is still published and the grid no longer uses it: an
obstacle that yields no corners never existed on that path, and an audit of the
depth chain found five defects, the largest of which was the warp holding pitch
and roll at zero -- 94 % of the frame-common error.

The grid is a fixed 60 x 60 m at 0.1 m, anchored to the first pose. Scored
against truth on `approach_hd60_occ_b`: G1 (calling the vehicle free) 0, G2
(calling blocked ground free) 3, G3 (cover) 0.831, G4 (false positives) 29, path
ghosts 0. A Python reference under the same conditions gives G2 2 / G3 0.829 /
G4 39, so cover and false positives are ahead here.

![Scored occupancy grid](media/occupancy.png)

Four numbers are not a map, so here is the map they are scored over. White is
ground the sweep carved free, dark grey is what it called occupied, blue is
ground that truly is blocked and pale grey is never observed; the parked cars
are magenta and the parts of them the sweep found are green. The three failure
predicates have their own colours, and what the scores above say is that **there
is no red at all** -- not one cell of a vehicle was called free -- with three
yellow cells and a thin orange rim. Regenerate it with
`monoscale_evaluation/render_map.py`; the truth archives it scores against are
too large to ship, and `sim/` is what makes them.

## Building and running

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install
source install/setup.bash

ros2 launch monoscale_odometry odometry.launch.py
```

Odometry needs no CUDA. `monoscale_tracker` has a GPU path for the optical flow
(`use_cuda`), built only when an OpenCV with cudaoptflow is found, and off by
default.

**The occupancy grid does need it.** The CPU sweep takes 28 minutes over 674
keyframes and cannot ship; the CUDA path takes **34 ms** per keyframe, both
cameras counted, with the card 95 % busy. The kernel is
`src/monoscale_occupancy_grid_map/src/sweep_kernels.cu` and CMake compiles it
when `nvcc` is present. Without `nvcc` the build falls back to the CPU path,
so a successful build is not a deployable one -- check the first run for
`cuda backend: available=1`.

Until 2026-09-18 that fallback did not exist: `cuda_match` was declared
unconditionally and defined only in the CUDA translation unit, so a machine
without `nvcc` failed at the link rather than falling back. It went unseen
because this machine's CMake finds `/usr/local/cuda` whether or not `nvcc` is
on PATH; a build of a tree made only of the tracked files is what surfaced it.

## Number of cameras

`camera_names` sets it. The default is the two on the vehicle.

The estimator and the tracker use different keys: the estimator reads
`camera_names`, the tracker `cameras`. The same split applies to image topics --
the estimator composes `<name>_image_topic`, the tracker reads an `image_topics`
array. In deployment `deployment.param.yaml` fills the tracker's side.

```yaml
camera_names: ['front', 'rear']       # estimator
cameras: ['front', 'rear']            # tracker
front.k: [...]                        # 3x3, row major
front.rotation_base_from_camera: [...]   # 3x3
front.translation_base_from_camera: [x, y, z]
front_image_topic: /sensing/camera/front/image_raw
```

Add a name and the same parameters are read under it; frames are gathered by
whichever combination of stamps sits closest together. There is no upper bound
and no pairing: frames are aligned by stamp, not by being a pair. One camera is
a supported configuration and costs a factor of 2.6 to 6.2 -- the table above.

## Tests

```bash
colcon test
colcon test-result --all
```

148 of them: the geometry, the anchor map, the filters, the inertial path, the
attitude, the estimator end to end over a synthetic drive, and two that hold the
vehicle's frame tree against the simulator's own calibration.

## Scoring against a recorded drive

```bash
ros2 run monoscale_odometry monoscale_replay <bag> \
  --params src/monoscale_odometry/config/vision_fisheye.param.yaml \
  --set track_topic_prefix:=/vision/tracks
```

It reads the bag itself and drives the library in recording order. It reads no
clock and no random source, so one bag gives one trajectory on a desktop and on
an Orin alike -- which is why regressions are visible at all.

## Reproducing the recordings

[`sim/`](sim/README.md) carries the conditions the bags were recorded under: the
sensor kit the odometry's extrinsics are derived from, the sensor mapping, and
the drive scripts. It is there because a millimetre of mounting height moves the
result by 0.1 %, so the calibration and the config that depends on it have to
move in one commit.
