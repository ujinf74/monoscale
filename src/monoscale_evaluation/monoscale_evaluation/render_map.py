#!/usr/bin/env python3
"""Draw a scored occupancy map: what the sweep said, against what was there.

    python3 render_map.py <layer.npz> <out.png> --reference <truth.npz>
                          --surface <truth_surface.npz> [--label TEXT]

`sweep_offline` writes the ternary grid and nothing else, and the four scores
the benchmark quotes -- a cell that is the vehicle called free, blocked ground
called free, cover, free ground called occupied -- mean nothing to a reader who
cannot see the map they are scored over. This paints each of those predicates
its own colour, so the numbers can be checked against the picture.

`layer.npz` holds `values` (the grid `sweep_offline` writes) and `track` (the
path, in the grid's frame). The reference and surface archives are the truth for
that drive; like the recordings they are too large to ship, and the drive
scripts under `sim/` are what makes them.

The predicates are score_goal.py's, copied rather than imported so that a change
there shows up as a disagreement rather than silently moving the picture. No
dilation anywhere -- a dilated overlay hides exactly the boundary cells the score
is about. Full frame, never cropped.
"""

import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('layer')
ap.add_argument('out')
ap.add_argument('--reference', required=True)
ap.add_argument('--surface', required=True)
ap.add_argument('--label', default='')
ap.add_argument('--seen-range', type=float, default=8.0)
ap.add_argument('--min-seen', type=int, default=3)
args = ap.parse_args()

T = np.load(args.reference)
truth, tag = T['values'], T['label']
res = float(T['resolution'])
ox, oy = float(T['origin_x']), float(T['origin_y'])
slab = np.load(args.surface)['slab']
car = (truth == 100) & (tag == 14)
blocked = (truth == 100) | slab
free_truth = (truth == 0) & (T['seen_range'] <= args.seen_range) \
    & (T['seen_count'] >= args.min_seen)

F = np.load(args.layer)
values = F['values']
said_free = values == 0
said_occupied = values == 100
track = F['track']

h, w = truth.shape
seed = np.ones((h, w), np.uint8)
gx = np.clip(((track[:, 0] - ox) / res).astype(np.int64), 0, w - 1)
gy = np.clip(((track[:, 1] - oy) / res).astype(np.int64), 0, h - 1)
seed[gy, gx] = 0
reach = cv2.distanceTransform(seed, cv2.DIST_L2, 5) * res
near = reach < 10.0

canvas = np.full((h, w, 3), 245, np.uint8)
canvas[near] = 225
canvas[said_free] = 255
canvas[said_occupied] = (70, 70, 70)
canvas[blocked] = (200, 150, 90)
canvas[car] = (200, 60, 200)
canvas[car & said_occupied] = (140, 230, 140)
canvas[free_truth & said_occupied & near] = (0, 140, 255)
canvas[blocked & said_free & near] = (60, 220, 240)
canvas[car & said_free] = (0, 0, 230)
canvas[gy, gx] = (30, 30, 30)

canvas = cv2.flip(canvas, 0)
canvas = cv2.resize(canvas, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
if args.label:
    cv2.putText(canvas, args.label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2, cv2.LINE_AA)
legend = ['red=G1 car called free', 'yellow=G2 blocked called free',
          'orange=G4 free called occupied', 'magenta=true car',
          'green=car correctly occupied', 'blue=true blocked']
for n, text in enumerate(legend):
    cv2.putText(canvas, text, (12, 56 + 22 * n), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imwrite(args.out, canvas)
# The same predicates the colours use, counted. Printed beside the picture so
# a change that moves one moves the other, and neither can drift from the other.
g1 = int((car & said_free).sum())
g2 = int((blocked & said_free & near).sum())
g4 = int((free_truth & said_occupied & near).sum())
reachable = blocked & near
g3 = float((reachable & said_occupied).sum()) / max(int(reachable.sum()), 1)
print(f'{args.out}  free={said_free.sum():,} occupied={said_occupied.sum():,}  '
      f'G1={g1} G2={g2} G3={g3:.3f} G4={g4}')
