"""The photometric length against truth, with no estimator in between.

ATE answers "how wrong is the trajectory", and its noise floor on the nine
drives is about two per cent. That is too blunt to see a length bias of a
tenth of a per cent, and it is contaminated: the scale that minimises ATE is
not the scale that makes the measurement unbiased, because everything else in
the estimator gets to lean on it.

This asks the narrower question. The tracker's road fit reports a step in
metres every frame; CARLA reports where the vehicle actually was. Comparing
them needs nothing else, and measured by recording one condition three times
the answer repeats to **0.012 per cent** -- sharp enough to see the terms the
trajectory can only feel.

    MONOSCALE_ESM_SIGMA=/tmp/run.csv \
      build/monoscale_tracker/feature_tracker --offline <bag> <out> --ros-args \
      --params-file src/monoscale_odometry/config/vision_fisheye.param.yaml \
      -p cameras:="['front','rear']" -p image_topics:="[...]"

    python3 photometric_bias.py <bag> /tmp/run.csv

Reports each camera and their combination three ways. Median, mean and
accumulated agree closely on a clean drive; where they part company the drive
carries a tail, and the accumulated figure is the one the trajectory sees.

The weight is the one the estimator uses to null the pitch lever between the
two mounts, which is near even. It is a default here rather than a claim:
`Estimator` derives it per frame from the extrinsics and the measured leak.
"""
import bisect
import csv
import math
import os
import statistics
import sys

DEFAULT_FRONT_WEIGHT = 0.514


def truth_track(bag):
    """Ground-truth positions out of a bag, as (stamps, xs, ys)."""
    import sqlite3
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.serialization import deserialize_message

    database = next(f for f in os.listdir(bag) if f.endswith(".db3"))
    connection = sqlite3.connect(os.path.join(bag, database))
    topic = connection.execute(
        "select id from topics where name='/carla/ground_truth/pose'").fetchone()[0]
    stamps, xs, ys = [], [], []
    for (blob,) in connection.execute(
            "select data from messages where topic_id=? order by timestamp", (topic,)):
        message = deserialize_message(bytes(blob), PoseWithCovarianceStamped)
        stamps.append(message.header.stamp.sec + 1e-9 * message.header.stamp.nanosec)
        xs.append(message.pose.pose.position.x)
        ys.append(message.pose.pose.position.y)
    return stamps, xs, ys


def bias(dump, bag, front_weight=DEFAULT_FRONT_WEIGHT):
    stamps, xs, ys = truth_track(bag)

    def at(t):
        i = bisect.bisect_left(stamps, t)
        if i <= 0 or i >= len(stamps):
            return None
        span = stamps[i] - stamps[i - 1]
        a = (t - stamps[i - 1]) / span if span > 0 else 0.0
        return (xs[i - 1] + a * (xs[i] - xs[i - 1]), ys[i - 1] + a * (ys[i] - ys[i - 1]))

    steps = {}
    for row in csv.DictReader(open(dump)):
        steps.setdefault(float(row["stamp"]), {})[row["camera"]] = float(row["step"])
    # Only frames where both cameras answered: the combination is the quantity
    # the estimator consumes, and a one-sided frame is not it.
    times = sorted(t for t, seen in steps.items() if len(seen) == 2)

    out = {}
    for camera in ("front", "rear", "weighted"):
        relative, measured, actual = [], 0.0, 0.0
        for i in range(1, len(times)):
            before, after = times[i - 1], times[i]
            # One frame at 30 Hz, and nothing that spans a gap.
            if not 0.02 < after - before < 0.06:
                continue
            was, now = at(before), at(after)
            if was is None or now is None:
                continue
            travelled = math.hypot(now[0] - was[0], now[1] - was[1])
            if travelled < 1e-3:
                continue
            if camera == "weighted":
                step = (front_weight * steps[after]["front"] +
                        (1.0 - front_weight) * steps[after]["rear"])
            else:
                step = steps[after][camera]
            if not math.isfinite(step):
                continue
            relative.append(step / travelled - 1.0)
            measured += step
            actual += travelled
        if not relative:
            continue
        out[camera] = {
            "median": statistics.median(relative),
            "mean": statistics.mean(relative),
            "accumulated": measured / actual - 1.0,
            "hops": len(relative),
        }
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    bag = argv[1]
    for dump in argv[2:]:
        found = bias(dump, bag)
        print(f"{os.path.basename(dump):24s}", end="")
        for camera in ("front", "rear", "weighted"):
            if camera not in found:
                continue
            it = found[camera]
            print(f"  {camera}: 중앙 {100 * it['median']:+6.3f}%"
                  f" 평균 {100 * it['mean']:+7.3f}%"
                  f" 누적 {100 * it['accumulated']:+6.3f}%", end="")
        print(f"  n={found.get('front', {}).get('hops', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
