"""Scores a set of replayed drives the way this stack should be judged.

Absolute trajectory error was the metric here for a long time and it hides
three things. It mixes path lengths, so a metre on a 12.8 m park manoeuvre
counts the same as a metre on a 135 m straight. It lets one early mistake carry
through the whole drive. And it cannot say whether an estimator got better
because it reads its observations more accurately or because it stopped
compounding what it already knew -- which is the distinction that mattered
most, and it was invisible for weeks.

So the headline is segment-relative translation error: KITTI's metric, sliding
over every start point, scaled to these path lengths rather than kilometres.
It is a per cent of distance travelled, so every drive is comparable.

Two diagnostics sit beside it and are the reason this file exists:

  hop%   the per-hop error as a fraction of the hop. What the observations and
         the solve together are worth. Measured against truth, a ground feature
         gives about 1% of the displacement being measured at any baseline, so
         1.0 here is the floor and not an arbitrary target.

  walk   what the trajectory actually accumulated, over what independent hops
         predict it should. A random walk of n hops has RMS error sd*sqrt(n/2),
         so 1.0 means the estimator adds no memory of its own. Above 1.0 the
         hop errors are correlated, and no amount of per-hop accuracy will fix
         that -- it is a different defect and it needs a different search.

The three recordings of the 2 m/s straight are three recordings of ONE
condition. They enter the headline once, as their median, and separately as the
repeat spread, rather than weighting that condition three elevenths of the
score. The spread is its own signal: across this stack's history, moving a
parameter swings the mean and inflates the spread, while fixing a defect leaves
the mean alone and brings the spread down.
"""

import argparse
import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Long enough to see drift, short enough that a 29 m park still supports one.
SEGMENT_LENGTHS: Tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
# Three recordings of one condition, not three conditions.
REPEATS: Tuple[str, ...] = ('str_v2', 'str_v3', 'str_v4')


def read_pair(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """The estimate and the truth from one replay's TUM output."""
    estimate = np.loadtxt(f'{directory}/estimate.tum')
    truth = np.loadtxt(f'{directory}/truth.tum')
    count = min(len(estimate), len(truth))
    return estimate[:count, 1:3], truth[:count, 1:3]


def segment_error(
    estimate: np.ndarray, truth: np.ndarray, length: float
) -> Optional[float]:
    """Mean relative translation error over every segment of `length`, in %."""
    walked = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    )
    # Two segment-lengths of path at least. Below that the start points all
    # overlap and an average over segments is one measurement in a crowd's
    # clothing -- a 12.8 m drive reported 5.07% at 10 m for exactly this.
    if walked[-1] < 2.0 * length:
        return None
    ends = np.searchsorted(walked, walked + length)
    errors: List[float] = []
    for start, end in enumerate(ends):
        if end >= len(walked):
            break
        travelled = truth[end] - truth[start]
        distance = float(np.linalg.norm(travelled))
        if distance < 1e-6:
            continue
        drifted = estimate[end] - estimate[start]
        errors.append(float(np.linalg.norm(drifted - travelled)) / distance)
    if not errors:
        return None
    return 100.0 * float(np.mean(errors))


def hop_summary(estimate: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Per-hop accuracy, and how much of it the estimator carries forward."""
    stepped = np.diff(estimate, axis=0)
    walked = np.diff(truth, axis=0)
    distance = np.linalg.norm(walked, axis=1)
    moving = distance > 1e-9
    stepped, walked, distance = stepped[moving], walked[moving], distance[moving]
    forward = walked / distance[:, None]
    error = stepped - walked
    along = np.einsum('ij,ij->i', error, forward)
    across = error[:, 0] * -forward[:, 1] + error[:, 1] * forward[:, 0]
    hops = len(along)
    expected = math.hypot(float(along.std()), float(across.std())) * math.sqrt(hops / 2.0)
    absolute = float(np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1))))
    return {
        'hop': 100.0 * float(np.median(np.abs(along) / distance)),
        'walk': absolute / max(expected, 1e-9),
        'ate': absolute,
        'distance': float(distance.sum()),
    }


def score(drives: Sequence[Tuple[str, str]]) -> Dict[str, Dict]:
    scored: Dict[str, Dict] = {}
    for name, directory in drives:
        estimate, truth = read_pair(directory)
        entry = hop_summary(estimate, truth)
        entry['segments'] = {
            length: segment_error(estimate, truth, length) for length in SEGMENT_LENGTHS
        }
        scored[name] = entry
    return scored


def _headline(scored: Dict[str, Dict], pick) -> Optional[Tuple[float, float]]:
    """One entry per condition, the repeats contributing their median."""
    values = [pick(v) for k, v in scored.items() if k not in REPEATS]
    values = [v for v in values if v is not None]
    repeated = [pick(scored[k]) for k in REPEATS if k in scored]
    repeated = [v for v in repeated if v is not None]
    if repeated:
        values.append(statistics.median(repeated))
    if not values:
        return None
    return float(np.mean(values)), max(values)


def report(scored: Dict[str, Dict]) -> None:
    header = ''.join(f'{"RPE" + str(int(n)):>8}' for n in SEGMENT_LENGTHS)
    print(f'{"drive":11}{"거리":>7}{header}{"hop%":>8}{"walk":>7}{"ATE":>9}')
    for name, entry in scored.items():
        cells = ''.join(
            f'{entry["segments"][n]:8.3f}' if entry['segments'][n] is not None
            else f'{"-":>8}'
            for n in SEGMENT_LENGTHS
        )
        print(
            f'{name:11}{entry["distance"]:7.0f}{cells}'
            f'{entry["hop"]:8.2f}{entry["walk"]:7.2f}{entry["ate"]:9.4f}'
        )
    print()
    for length in SEGMENT_LENGTHS:
        found = _headline(scored, lambda e, n=length: e['segments'][n])
        if found is None:
            continue
        mean, worst = found
        print(f'  RPE {int(length):2d}m   평균 {mean:6.3f}%   최악 {worst:6.3f}%')
    for key, label, width in (('hop', 'hop%', 2), ('walk', 'walk', 2), ('ate', 'ATE', 4)):
        mean, worst = _headline(scored, lambda e, k=key: e[k])
        print(f'  {label:8} 평균 {mean:6.{width}f}   최악 {worst:6.{width}f}')
    spread = [
        scored[k]['segments'][SEGMENT_LENGTHS[0]] for k in REPEATS
        if k in scored and scored[k]['segments'][SEGMENT_LENGTHS[0]] is not None
    ]
    if len(spread) >= 2:
        print(
            f'  반복폭    {max(spread) / min(spread):6.2f}x  '
            f'(RPE {int(SEGMENT_LENGTHS[0])}m, 같은 드라이브 {len(spread)}회 녹화)'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'drives', nargs='+', metavar='NAME=DIR',
        help='a replay --tum directory per drive, holding estimate.tum and truth.tum',
    )
    parsed = parser.parse_args()
    drives = []
    for spec in parsed.drives:
        name, _, directory = spec.partition('=')
        if not directory:
            parser.error(f'expected NAME=DIR, got {spec!r}')
        drives.append((name, directory))
    report(score(drives))


if __name__ == '__main__':
    main()
