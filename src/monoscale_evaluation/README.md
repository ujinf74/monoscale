# monoscale_evaluation

Scoring replayed drives, and checking measurements against truth.

## The benchmark

What the stack scores now, and how to make those numbers again.

What the metrics mean is not repeated here. The docstring of
`monoscale_evaluation/benchmark.py` carries why RPE is the headline, what `hop%`
and `walk` separate, and why repeated recordings are counted at their worst.
This is about **which drives**, **how**, and **what it is now**.

## Two sets

**Nine tuning drives.** What is looked at when choosing parameters.

| tag | bag | condition | road |
| --- | --- | --- | --- |
| `str_v2` | `straight120_v2_r2` | straight, 1.9 m/s | A |
| `str_v3` | `straight120_v3_r2` | same condition, re-recorded | A |
| `str_v4` | `straight120_v4_r2` | same condition, re-recorded | A |
| `str_1.5` | `straight110_s15_r2` | straight, 1.4 m/s | A |
| `str_4.0` | `straight110_s4_r2` | straight, 3.7 m/s | A |
| `str_8.0` | `straight_s8_t01_r3` | straight, 7.5 m/s | B |
| `curve_s05` | `curve_s05_t01c_r3` | gentle slalom (0.0013 rad/frame) | B |
| `curve_s10` | `curve_s10_t01d_r3` | medium slalom (0.0025) | B |
| `curve_s20` | `curve_s20_t01c_r3` | hard slalom (0.0049) | B |

`str_v2`/`v3`/`v4` are **one condition recorded three times**. They enter the
headline once, at their worst, and are reported separately as the repeat spread.

Road A is the lane at spawn x~-48.8, road B is at y~-199. Read from truth, A
moves 2.1-3.8 mm in z over 100 m with local slopes spread over 0.038-0.099 deg,
while B is flat at 0.0-1.3 mm and 0.004 deg. The gradient is 0.0000 deg on both,
so neither is a hill. That difference separates the far-range error of the
ground projection, so a result that splits by road is not a coincidence.

**Two held-out drives.** What is **not** looked at when choosing parameters.

| tag | bag | condition |
| --- | --- | --- |
| `park_clean` | `park_clean_r2` | 30 m parking manoeuvre, reverse and gear changes |
| `park_obst` | `park_obst_a_r2` | 13 m parking manoeuvre, obstacle |

Parking is a 13-30 m low-speed manoeuvre, so the absolute figures are ten times
the bench's. **Comparing the absolute numbers to the bench is meaningless; what
this set is for is an A/B inside itself.** The short `park_obst` carries most of
the discriminating power.

The set has earned its place. On 2026-09-10 alone it refuted two changes that
had won on the bench -- cross-source anchor rebinding (bench worst -17%,
held-out mean +13% and final error +25%) and `anchor_max_range_m: 2.8` (bench
worst -19%, held-out final error 2.2x). In the week of 2026-09-11 it refused two
more: widening `anchor_max_range_m` and `ground_max_distance_m` together, which
is -11% of bench ATE and +7.5% held-out, and a flow-derived mask on feature
births, which is -24% of the tracker's flow stage and +11% of the held-out worst
ATE. **Winning on the bench is not a reason to deploy.**

## Making the numbers again

Two stages: pull tracks out of the bag with the tracker, then replay them
through the estimator and score.

```bash
M=<repo>
BAGS=<where the recordings are>
BASE="--params-file $M/src/monoscale_odometry/config/vision_fisheye.param.yaml \
  -p cameras:=['front','rear'] \
  -p image_topics:=['/sensing/camera/front/fisheye/image_raw','/sensing/camera/rear/fisheye/image_raw']"

$M/build/monoscale_tracker/feature_tracker --offline $BAGS/<bag> <tracks_dir> \
  --ros-args $BASE

$M/build/monoscale_odometry/monoscale_replay <tracks_dir> \
  --params $M/src/monoscale_odometry/config/vision_fisheye.param.yaml \
  --set track_topic_prefix:=/vision/tracks --tum <out_dir>

python3 $M/src/monoscale_evaluation/monoscale_evaluation/benchmark.py \
  str_v2=<out_v2> str_v3=<out_v3> ... curve_s20=<out_cs20>
```

**Start CARLA with `-quality-level=Low`.** This matters when recording a new
bag, and a bag recorded otherwise cannot be compared with this set. Measured
2026-09-10: the same straight at 1.4 m/s recorded against a server started with
`-quality-level=Epic -RenderOffScreen` gives a photometric length of **-6.826%**
against truth, and `-quality-level=Low` gives **+0.111%**, which is the range of
the 09-02 `str_1.5` at +0.076%. A factor of sixty in one launch argument.

The photometric alignment works on the road's texture and the quality level
changes that texture, its shadows and its post-processing. The trap produced a
conclusion and then erased it: recorded at Epic, a fast drive (3.4 m/s) read
-2.0% and a slow one (0.8 m/s) -7.7%, which stood up as "the fit has a working
range of 45-70 mm of step" -- re-recorded at Low they read +0.150% and +0.129%,
inside the bench's own range. There was no window. `record_run.sh` pins the sun
angle "for comparability" and leaves the rendering quality to whoever starts the
server.

```bash
cd <CARLA> && ./CarlaUE4.sh -quality-level=Low
```

**Extract with the parameter file alone.** That is what `odometry.launch.py`
does, and a difference here means the bench is scoring a configuration that is
not deployed. Until 2026-09-08 the extraction script added `road_step_esm:=true`
and the yaml did not carry it -- ATE over distance measured 0.0445% and read
0.0229% once the key was written down. **A factor of two was one line.** There
is one check that catches the class: extract with the parameter file only, and
score that.

Track extractions stay valid across estimator-side changes. They have to be
pulled again only when a tracker parameter moves.

## Now (2026-09-12, `road_step_esm_dof: 3`)

|  | bench 9 mean | bench 9 worst | held-out mean | held-out worst |
| --- | ---: | ---: | ---: | ---: |
| **ATE / distance** | **0.0237 %** | **0.0391 %** | **0.1781 %** | **0.2659 %** |
| **final error / distance** | **0.0326 %** | **0.0627 %** | **0.1461 %** | **0.1937 %** |
| hop% | 0.14 | 0.17 | 0.83 | 1.29 |
| walk | 1.10 | 2.57 | 0.76 | 1.17 |
| repeat spread | 1.06x | | | |

The changes of 2026-09-11 and 12, in order:

| | bench ATE | bench final | held-out ATE | held-out final |
| --- | ---: | ---: | ---: | ---: |
| before (`ef10fe7`) | 0.0226 % | 0.0287 % | 0.1880 % | 0.1777 % |
| `road_step_arc` deployed | 0.0224 | 0.0281 | 0.1875 | 0.1750 |
| `photometric_scale` removed | 0.0523 | 0.0919 | 0.1903 | 0.1891 |
| mount heights corrected | 0.0234 | 0.0324 | 0.1833 | 0.1582 |
| **`road_step_esm_dof: 3`** | **0.0237** | **0.0326** | **0.1781** | **0.1461** |

Taking out the one fitted multiplier, correcting a millimetre of frame geometry
and dropping one degree of freedom leaves **the bench where it was, inside its
repeat spread, and the held-out set better by 5.3% of ATE and 17.8% of the final
error.** The set that was never tuned on is the one that improved.

**Careful**: when a tracker parameter moves (mount heights, `road_step_esm_dof`
and the like), **the extraction and the parameter file have to move together.**
The estimator reads the heights from that same file, so changing only the
extraction leaves the two paths on different geometries and the scoring is
quietly wrong.

## This set is not on one road surface

The nine bench drives are split 5:4 across **two stretches** of Town01, and the
photometric length bias follows the stretch. Measured against truth with no
estimator in the loop, after correcting the truth pose's report point:

| surface | drives | what the band shows | bias |
| --- | --- | --- | ---: |
| A | str_v2 v3 v4 1.5 4.0 | dark asphalt with white bay and lane markings | -0.03 .. +0.10 % |
| B | str_8.0 curve_s05 s10 s20 | plain light concrete, no markings at all | +0.17 .. +0.20 % |
| -- | park_clean park_obst (held-out) | a third asphalt; park_obst has a parked car in the front band | path length -0.44 / +0.25 % |

Same speed, same map, uniform along each drive. **It is the road, not the speed
and not the turning.** A height error cannot know what is painted on the road,
so this is not a shape a constant scale can correct.

Fill this axis on purpose when extending the set. Right now the surface is a
side effect of which drives were recorded, and that tied one parameter to this
set's mix.

## What not to ask this set

**Do not ask ATE about a length bias.** The harness's noise floor is +-2% and a
length bias is measured in tenths of a per cent, and above all **the scale that
minimises ATE is not the unbiased scale** -- every other error gets to lean on
it. `photometric_scale` was exactly that, a constant fitted to the mean of a
bias that varies by road surface, and it was removed on 2026-09-11: it cost 134%
of bench ATE and 1.5% of held-out, which is what a fit to the set looks like.

Ask truth directly, with the estimator taken out:
`monoscale_evaluation/photometric_bias.py`. Across three recordings of one
condition it **repeats to 0.012%**, 150 times sharper than ATE.

**Precise and correctly aimed are different things.** For weeks this tool
reported a length bias that appeared only when turning; three drives agreed with
each other, and thirteen mechanisms were excluded one at a time. The cause was
that CARLA reports the truth pose at the actor origin, 1.399 m ahead of the rear
axle. Along a straight that offset cancels in a displacement; through a turn the
forward point traces the longer path, exceeding the reference point's by
`(L psi)^2 / 2s`. Corrected 2026-09-11 (`TRUTH_REPORT_OFFSET_M`): the straight
figures did not move and 96-98% of the turn term disappeared. **Run a straight
drive as a control** with anything new this tool says.

**The repeat spread of 1.05x is the floor.** Three recordings of one condition
differ by that much, so a smaller difference is not a measurement. An isolated
optimum with worse neighbours on both sides is the signature of a fit, and then
the held-out set decides.

## Updating

**When the deployed configuration (`vision_fisheye.param.yaml`) changes, fix the
tables here in the same commit.** The commit hash beside a table says which tree
its numbers came from.

## The other tools

| file | what it measures |
| --- | --- |
| `benchmark.py` | scores replayed trajectories against truth. Headline and diagnostic metrics. |
| `photometric_bias.py` | the road fit's step against truth directly. No estimator in between, so it sees a length bias at 0.012%. |
| `jacobian.py` | the error Jacobian of the ground projection: which extrinsic or intrinsic enters range, and in which direction. |
| `nullspace.py` | the rank and nullspace of that Jacobian: what is observable and what cannot be told apart. |
| `bag_gate.py` | whether a fresh recording is usable: collisions, drift out of the lane, and the PhysX substep artefact. |
