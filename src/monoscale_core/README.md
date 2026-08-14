# monoscale_core

The estimation itself, with no ROS in it: ground projection, the anchor map,
the filters and the state machine that drives them. `monoscale_odometry`
and `monoscale_occupancy_grid_map` are thin wrappers over this, which is
what lets the same code be tested without a graph and later be called directly
by the tracker in one process.

## What it is a port of, and from when

This is a rewrite of the Python estimator, not a new design. Every number in
the parameter files still means what it meant, and the comments that record
what each one cost when it was wrong were carried across rather than
summarised.

The Python it was ported from is still being developed elsewhere, and it is no
longer in the tree here -- once the C++ was held to it and passed, carrying two
implementations of the same thing was the liability rather than the safety net.
It is still in the history, and the port names the exact state it was taken
from. These are the git blob hashes:

| source | blob |
| --- | --- |
| `monoscale_odometry/geometry.py` | `603d8cd3662d9e35a600cac68c79426f94923d43` |
| `monoscale_odometry/tracks.py` | `539d123f1f1abb87ebda3e920279eea5f1951918` |
| `monoscale_odometry/fusion.py` | `f2802c49fa10357138caa412c125f05ada9f5f1b` |
| `monoscale_odometry/inertial.py` | `a43132150155672315e6e1759e13d9a821296348` |
| `monoscale_odometry/attitude.py` | `b6255d3ab028b5405853df9ee1530e168420211e` |
| `monoscale_odometry/runtime.py` | `9fb5e74fab7637b4e2cf4bf8c4409e2acd791236` |
| `monoscale_odometry/odometry_node.py` | `ab4a6908a02e0e563f964d675f76e1333533cbb6` |
| `monoscale_occupancy_grid_map/occupancy.py` | `08acd0a05112bd5a845a4df4b1cf6706505029ee` |

`git cat-file -p <blob>` recovers any of them, so a later sync can diff against
what was actually read rather than against whatever the file says today.

## What was left behind, and why

**The optical flow front end.** `image_frontend.py` and the tracking inside the
Python node do the work `monoscale_tracker` already does in C++. Two
implementations of the one stage whose cost actually matters is a liability,
not a fallback. What that path offered -- running without a second node -- is
better served by calling this library from the tracker's own process, which is
possible precisely because there is no ROS in here.

**Anchor reassociation.** The ORB descriptor path in `GroundAnchorMap` is off
by default and was measured not to work on this surface: of the ground points
that came back within half a metre of a stored anchor, 85 % had a second
candidate that looked just as much like them. Worth revisiting against features
that are actually distinctive, which is a different detector rather than a
different threshold.

**Keyframe triangulation for obstacles.** It needs the images this path does
not receive. The Python reached for it anyway and dereferenced a frame with no
picture in it, which is a crash on the deployed path as soon as mapping is
enabled. Here obstacles come from `obstacle_slip_baseline_m`, and where that is
zero the mapper is told there are none rather than being handed a null.

## Held to the Python it replaces

`monoscale_replay` reads a bag directly and drives this library in recorded
order, which is what the Python's `offline_replay.py` did for it. On
`approach_hd60_tracks_w1280`, with the same parameter file, before the Python
was removed:

| | Python | C++ |
| --- | ---: | ---: |
| ATE RMSE | 0.110537 m | 0.1111 m |
| final position error | 0.142806 m | 0.1451 m |
| scale ratio | 1.00886 | 1.009 |
| drift | 0.5257 % | 0.53 % |
| pairs / solves / estimates | 810 / 285 / 284 | 810 / 285 / 284 |

Nothing in here reads a clock or a random seed, so the same bag gives the same
trajectory on the desktop and on the Orin -- identical to four decimal places,
which is what makes a regression visible.

## Cost

Measured on the same recording, per solve:

| stage | desktop Python | desktop C++ | Orin Python | Orin C++ |
| --- | ---: | ---: | ---: | ---: |
| ground projection | 0.47 ms | 0.30 | 2.23 | 1.25 |
| registration | 0.99 | 0.09 | 5.87 | 0.50 |
| anchor update | 0.57 | 0.08 | 2.93 | 0.31 |
| correspondence | 0.22 | 0.14 | 0.88 | 0.37 |
| **total** | **2.25 ms** | **0.61** | **11.91** | **2.43** |

The ground projection barely moves because the Python was already calling C++
for it through `monoscale_fast`. What changed by an order of magnitude is the
bookkeeping around it.
