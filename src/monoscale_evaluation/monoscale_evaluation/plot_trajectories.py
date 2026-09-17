"""The figure on the front page, from the trajectories the benchmark produces.

Everything else in this repository carries the measurement that made it, and a
picture with no script would be the one thing a reader has to take on trust. So
this is here, it reads the same `.tum` pairs `monoscale_replay --tum` writes,
and it recomputes the numbers it prints rather than being handed them.

    python3 plot_trajectories.py <slalom_dir> <park_dir> <out.png>

Three panels, and the third is the point. At the scale of a 111 m drive the
estimate and the truth are one line, which says nothing a reader can size; the
third panel zooms until the gap is visible and puts a scale bar next to it, so
the error is shown rather than asserted. The window is chosen from the local
error itself -- six times it -- so the panel neither flatters the result by
zooming out nor exaggerates it by zooming in.
"""
import sys

import numpy as np

TRUTH = '#9aa7b8'
ESTIMATE = '#c2410c'


def load(path):
    """x, y out of a TUM-format trajectory."""
    out = []
    for line in open(path):
        if line.startswith('#'):
            continue
        field = line.split()
        if len(field) >= 3:
            out.append((float(field[1]), float(field[2])))
    return np.array(out)


def pair(directory):
    """Estimate and truth from one output directory, both from their first pose.

    No alignment beyond that: the first pose is the common origin and everything
    after it is what the estimator produced.
    """
    estimate = load(f'{directory}/estimate.tum')
    truth = load(f'{directory}/truth.tum')
    count = min(len(estimate), len(truth))
    estimate, truth = estimate[:count], truth[:count]
    return estimate - estimate[0], truth - truth[0]


def errors(estimate, truth):
    return np.hypot(estimate[:, 0] - truth[:, 0], estimate[:, 1] - truth[:, 1])


def ate(estimate, truth):
    return float(np.sqrt((errors(estimate, truth) ** 2).mean()))


def travelled(truth):
    return float(np.hypot(np.diff(truth[:, 0]), np.diff(truth[:, 1])).sum())


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 1
    slalom_dir, park_dir, out = argv[1], argv[2], argv[3]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13.2, 3.9))
    grid = figure.add_gridspec(1, 3, width_ratios=[2.35, 1.0, 1.0], wspace=0.30)
    wide, square, close = (figure.add_subplot(grid[i]) for i in range(3))

    estimate, truth = pair(slalom_dir)
    wide.plot(truth[:, 0], truth[:, 1], color=TRUTH, lw=5.0,
              solid_capstyle='round', label='ground truth')
    wide.plot(estimate[:, 0], estimate[:, 1], color=ESTIMATE, lw=1.5,
              label='monoscale')
    distance = travelled(truth)
    wide.set_title(
        f'slalom, {distance:.0f} m\n'
        f'ATE {ate(estimate, truth) * 1000:.0f} mm  ·  '
        f'{ate(estimate, truth) / distance * 100:.4f} % of distance',
        fontsize=10, pad=9)
    wide.legend(fontsize=9, frameon=False, loc='upper left')
    span = truth[:, 1].max() - truth[:, 1].min()
    wide.set_ylim(truth[:, 1].min() - 0.35 * span, truth[:, 1].max() + 0.40 * span)

    held, held_truth = pair(park_dir)
    square.plot(held_truth[:, 0], held_truth[:, 1], color=TRUTH, lw=5.0,
                solid_capstyle='round')
    square.plot(held[:, 0], held[:, 1], color=ESTIMATE, lw=1.5)
    square.set_aspect('equal')
    held_distance = travelled(held_truth)
    square.set_title(
        f'parking manoeuvre, {held_distance:.0f} m, held out\n'
        f'ATE {ate(held, held_truth) * 1000:.0f} mm  ·  '
        f'{ate(held, held_truth) / held_distance * 100:.4f} %',
        fontsize=10, pad=9)

    # Where the local error is closest to the drive's own ATE, so the zoom shows
    # a typical gap rather than the best or the worst of it.
    local = errors(estimate, truth)
    at = int(np.argmin(np.abs(local - ate(estimate, truth))))
    half = 3.0 * local[at]
    cx, cy = truth[at, 0], truth[at, 1]
    close.plot(truth[:, 0], truth[:, 1], color='#64748b', lw=1.6)
    close.plot(estimate[:, 0], estimate[:, 1], color=ESTIMATE, lw=1.6)
    close.set_xlim(cx - half, cx + half)
    close.set_ylim(cy - half, cy + half)
    close.set_aspect('equal')
    close.set_title(
        f'the same slalom, {2 * half * 100:.0f} cm across\n'
        f'the gap here is {local[at] * 1000:.0f} mm',
        fontsize=10, pad=9)
    bar = 10.0 ** np.floor(np.log10(half))
    close.plot([cx - half * 0.85, cx - half * 0.85 + bar],
               [cy - half * 0.86, cy - half * 0.86],
               color='0.3', lw=2.0, solid_capstyle='butt')
    close.text(cx - half * 0.85 + bar / 2, cy - half * 0.76,
               f'{bar * 100:.0f} cm', ha='center', fontsize=8.5, color='0.3')
    close.set_xticks([])
    close.set_yticks([])

    for axis in (wide, square):
        axis.grid(alpha=0.18, lw=0.5)
        axis.tick_params(labelsize=8)
        axis.set_xlabel('x [m]', fontsize=9)
        axis.set_ylabel('y [m]', fontsize=9)
    for axis in (wide, square, close):
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        close.spines[side].set_visible(False)

    figure.suptitle(
        'Estimate against CARLA ground truth — no alignment beyond the first '
        'pose, no loop closure',
        fontsize=10.5, color='0.35', y=1.02)
    figure.savefig(out, dpi=160, bbox_inches='tight')
    print(f'{out}: slalom ATE {ate(estimate, truth) * 1000:.1f} mm over '
          f'{distance:.1f} m, held-out ATE {ate(held, held_truth) * 1000:.1f} mm '
          f'over {held_distance:.1f} m')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
