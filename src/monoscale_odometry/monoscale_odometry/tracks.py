"""Persistent ground features and the map they are matched against.

The two-frame estimator asked one question per frame: where did these points go
since last time? When tracking degraded the wrong answer could win, and it did,
badly, above about 2 m/s.

This keeps each feature's position in the world, averaged over every frame it
has been seen in, and matches the current view against that instead. A feature
that mistracks now disagrees with its own history rather than voting with it,
which is the same reason scan-to-map beats scan-to-scan in LiDAR odometry.

The two priors that make this stack work are untouched: the ground plane still
supplies metric scale, and yaw still comes from the IMU.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import wrap_pi

try:
    import monoscale_fast as _fast
except ImportError:  # pragma: no cover - the pure Python path is the fallback
    _fast = None


class GroundAnchorMap:
    """World positions of ground features, refined over the frames they survive.

    Storage is a dense array with a free list rather than a dict of numpy
    points. The per-anchor arithmetic used to allocate a temporary for every
    feature every frame, which cost more than the registration it fed.
    """

    def __init__(
        self,
        max_anchors: int = 4000,
        max_age_frames: int = 40,
        update_gain: float = 0.35,
        max_observations: int = 20,
        initial_variance: float = 0.04,
        max_variance: float = 0.09,
        trial_observations: int = 4,
        select_by_consistency: bool = True,
    ):
        self.max_anchors = int(max_anchors)
        self.max_age_frames = int(max_age_frames)
        self.update_gain = float(update_gain)
        self.max_observations = int(max_observations)
        # How far each anchor's sightings scatter around it. A point on a
        # parked car, on a slope, or simply mistracked keeps landing somewhere
        # else, and counting sightings cannot tell that apart from a good one
        # seen often.
        self.initial_variance = float(initial_variance)
        self.max_variance = float(max_variance)
        self.trial_observations = int(trial_observations)
        # Off, an anchor is trusted for having been seen often, which was the
        # original rule. On, it also has to keep landing in the same place.
        self.select_by_consistency = bool(select_by_consistency)
        # A binary descriptor per anchor, so a landmark that leaves the frame
        # and comes back can be recognised instead of being met as a stranger.
        # Tracking alone cannot do this: the moment optical flow loses a
        # feature its identity is gone, and the map it had built with it
        # becomes unreachable.
        self.descriptor_bytes = 32
        capacity = self.max_anchors + 1
        self._position = np.zeros((capacity, 2), dtype=np.float64)
        self._observation = np.zeros(capacity, dtype=np.int64)
        self._variance = np.zeros(capacity, dtype=np.float64)
        self._seen = np.zeros(capacity, dtype=np.int64)
        # How much this anchor's position is worth knowing, summed over every
        # sighting that went into it.
        self._information = np.zeros(capacity, dtype=np.float64)
        self._identifier = np.full(capacity, -1, dtype=np.int64)
        self._descriptor = np.zeros((capacity, self.descriptor_bytes), dtype=np.uint8)
        self._described = np.zeros(capacity, dtype=bool)
        self.reassociations = 0
        self._free = list(range(capacity - 1, -1, -1))
        self._slot: Dict[int, int] = {}
        # Slot of each identity, indexed by the identity, -1 where unknown.
        self._by_id = np.full(4096, -1, dtype=np.int64)
        self.frame = 0
        self.discarded = 0

    def __len__(self) -> int:
        return len(self._slot)

    @property
    def positions(self) -> Dict[int, np.ndarray]:
        return {key: self._position[slot] for key, slot in self._slot.items()}

    @property
    def observations(self) -> Dict[int, int]:
        return {key: int(self._observation[slot]) for key, slot in self._slot.items()}

    @property
    def variances(self) -> Dict[int, float]:
        return {key: float(self._variance[slot]) for key, slot in self._slot.items()}

    def _slots(self, ids: np.ndarray) -> np.ndarray:
        """Slot of each id, or -1 where the id is not anchored yet.

        A table indexed by the identity itself. Sorting the live identities
        and searching that was the first attempt and is still too slow: the
        map holds eight thousand anchors, the sort is redone whenever it
        changes, and it changes every frame, so the two cameras between them
        paid 1.7 ms a pair for a lookup that touches seven hundred entries.
        Identities are handed out from zero and upwards by the tracker, so
        they index an array directly and nothing has to be kept in order.
        """
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=np.int64)
        table = self._by_id
        inside = ids < len(table)
        slots = np.where(inside, table[np.where(inside, ids, 0)], -1)
        return slots

    def _grow_table(self, highest: int) -> None:
        if highest < len(self._by_id):
            return
        # Geometric, and never below the identity that asked for it, so the
        # whole drive costs a handful of reallocations.
        size = max(2 * len(self._by_id), highest + 1)
        grown = np.full(size, -1, dtype=np.int64)
        grown[:len(self._by_id)] = self._by_id
        self._by_id = grown

    def anchored(self, ids: np.ndarray) -> np.ndarray:
        """Mask of which ids already have a world position."""
        return self._slots(ids) >= 0

    def world_points(self, ids: np.ndarray) -> np.ndarray:
        return self._position[self._slots(ids)]

    def weights(self, ids: np.ndarray) -> np.ndarray:
        """Seen often and landing in the same place each time is what counts."""
        return self._weights_at(self._slots(ids))

    def _weights_at(self, slots: np.ndarray) -> np.ndarray:
        counts = np.minimum(self._observation[slots], self.max_observations)
        if not self.select_by_consistency:
            return counts.astype(np.float64)
        return counts / np.maximum(self._variance[slots], 1e-4)

    def anchor_view(self, ids: np.ndarray):
        """World positions and weights together, from one lookup.

        The solve wants both for the same features, and asking twice meant
        searching the index twice for an answer that could not have changed
        in between.
        """
        slots = self._slots(ids)
        return self._position[slots], self._weights_at(slots)

    def describe(self, ids: np.ndarray, descriptors: np.ndarray) -> None:
        """Attach a descriptor to each anchor that does not have one yet."""
        if descriptors is None:
            return
        for identifier, descriptor in zip(ids, descriptors):
            slot = self._slot.get(int(identifier), -1)
            if slot >= 0 and not self._described[slot]:
                self._descriptor[slot] = descriptor
                self._described[slot] = True

    def undescribed(self, ids: np.ndarray) -> np.ndarray:
        """Mask of anchors that exist but have no descriptor yet."""
        slots = self._slots(ids)
        return (slots >= 0) & ~self._described[slots]

    def reassociate(
        self,
        world_points: np.ndarray,
        descriptors: np.ndarray,
        active: set,
        radius: float,
        max_hamming: int,
        min_observations: int,
    ):
        """Match fresh detections to anchors they have been before.

        A candidate has to be both in the right place and look right: position
        alone mistakes one paving joint for the next, and appearance alone
        mistakes them for every other joint in the car park.
        """
        if descriptors is None or len(world_points) == 0:
            return {}
        live = np.flatnonzero(
            (self._identifier >= 0)
            & self._described
            & (self._observation >= min_observations)
        )
        live = np.array(
            [s for s in live if int(self._identifier[s]) not in active], dtype=np.int64
        )
        if len(live) == 0:
            return {}
        positions = self._position[live]
        bits = np.unpackbits(self._descriptor[live], axis=1)
        found = {}
        taken = set()
        for index, (point, descriptor) in enumerate(zip(world_points, descriptors)):
            offset = positions - point
            near = np.flatnonzero(np.einsum('ij,ij->i', offset, offset) <= radius * radius)
            near = np.array([s for s in near if int(live[s]) not in taken], dtype=np.int64)
            if len(near) == 0:
                continue
            distance = np.count_nonzero(
                bits[near] != np.unpackbits(descriptor[None, :], axis=1), axis=1
            )
            best = int(np.argmin(distance))
            if distance[best] > max_hamming:
                continue
            # An ambiguous match is worse than none: ground texture repeats.
            if len(distance) > 1 and np.partition(distance, 1)[1] < 1.5 * distance[best] + 8:
                continue
            slot = int(live[near[best]])
            taken.add(slot)
            found[index] = int(self._identifier[slot])
        self.reassociations += len(found)
        return found

    def update(
        self,
        ids: np.ndarray,
        world_points: np.ndarray,
        allow_new: bool = True,
        information: Optional[np.ndarray] = None,
    ) -> None:
        """Fold in this frame's sightings.

        `allow_new` separates observing the map from defining it. Every frame
        refines the anchors it can see, but a new anchor is born at whatever
        pose the estimate held at that instant, so seeding from every frame
        fills the map with points inheriting short-baseline error. Admitting
        them only after real travel keeps the map built from viewpoints far
        enough apart to be worth registering against, at any camera rate.

        `information` is how much each sighting is worth, and it is not
        optional in spirit. A point on the ground eight metres out is pinned
        down far less precisely than the same point at one metre -- the same
        angular error is worth eight times as many centimetres -- and
        averaging the two as equals leaves the anchor sitting between them,
        displaced along its own bearing. Over a frame full of features that
        displacement does not cancel: it reads as a rotation, and the ground
        solve asks for 2.7 mrad of heading that is not there. Weighted by
        precision it asks for 0.24.
        """
        self.frame += 1
        ids = np.asarray(ids).reshape(-1)
        points = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
        finite = np.all(np.isfinite(points), axis=1)
        if not np.any(finite):
            self._prune()
            return
        ids = ids[finite]
        points = points[finite]

        weight = None
        if information is not None:
            weight = np.asarray(information, dtype=np.float64).reshape(-1)[finite]

        slots = self._slots(ids)
        known = slots >= 0
        if np.any(known):
            existing = slots[known]
            counts = self._observation[existing]
            if weight is None:
                gain = np.maximum(self.update_gain, 1.0 / (counts + 1.0))[:, None]
            else:
                # A running mean weighted by precision: each sighting moves the
                # anchor by its share of everything the anchor now knows.
                fresh = np.maximum(weight[known], 1e-9)
                total = self._information[existing] + fresh
                gain = (fresh / total)[:, None]
                self._information[existing] = total
            offset = points[known] - self._position[existing]
            residual = np.einsum('ij,ij->i', offset, offset)
            self._variance[existing] += gain[:, 0] * (
                residual - self._variance[existing]
            )
            self._position[existing] += gain * offset
            self._observation[existing] = counts + 1
            self._seen[existing] = self.frame

        fresh = np.flatnonzero(~known) if allow_new else np.empty(0, dtype=np.int64)
        room = min(len(fresh), len(self._free))
        if room:
            # In one go, not one at a time. Several hundred features are born
            # on the frames that matter, and a Python level assignment per
            # field per feature was most of what the map update cost.
            chosen = fresh[:room]
            taken = np.array(self._free[-room:][::-1], dtype=np.int64)
            del self._free[-room:]
            keys = ids[chosen].astype(np.int64)
            self._position[taken] = points[chosen]
            self._observation[taken] = 1
            self._variance[taken] = self.initial_variance
            self._seen[taken] = self.frame
            self._identifier[taken] = keys
            self._information[taken] = (
                1.0 if weight is None else np.maximum(weight[chosen], 1e-9))
            self._grow_table(int(keys.max()))
            self._by_id[keys] = taken
            self._slot.update(zip(keys.tolist(), taken.tolist()))
        self._prune()

    def _forget_slots(self, slots: np.ndarray) -> None:
        if len(slots) == 0:
            return
        slots = np.asarray(slots, dtype=np.int64)
        gone = self._identifier[slots]
        inside = (gone >= 0) & (gone < len(self._by_id))
        self._by_id[gone[inside]] = -1
        for key in gone.tolist():
            self._slot.pop(key, None)
        self._identifier[slots] = -1
        self._observation[slots] = 0
        self._described[slots] = False
        self._free.extend(int(s) for s in slots)

    def _prune(self) -> None:
        live = self._identifier >= 0
        stale = live & (self.frame - self._seen > self.max_age_frames)
        # Given a few sightings to settle, an anchor that still scatters is not
        # a landmark and should not be registered against.
        scattered = (
            live
            & (self._observation >= self.trial_observations)
            & (self._variance > self.max_variance)
            if self.select_by_consistency
            else np.zeros_like(live)
        )
        self.discarded += int(np.count_nonzero(scattered & ~stale))
        drop = np.flatnonzero(stale | scattered)
        if len(drop):
            self._forget_slots(drop)

        surplus = len(self._slot) - self.max_anchors
        if surplus <= 0:
            return
        # Drop the least trustworthy first; they carry the least history.
        live_slots = np.flatnonzero(self._identifier >= 0)
        score = self._observation[live_slots].astype(np.float64)
        if self.select_by_consistency:
            score = score / np.maximum(self._variance[live_slots], 1e-4)
        self._forget_slots(live_slots[np.argsort(score)[:surplus]])


def align_to_anchors(
    body_points: np.ndarray,
    world_points: np.ndarray,
    weights: np.ndarray,
    yaw: float,
    threshold: float,
    min_inliers: int,
    refine_yaw: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float]]:
    """Translation that places the current ground view onto its world anchors.

    With yaw taken as known each correspondence votes for one translation, so
    the solve is a robust average rather than a search. Returns the
    translation, the inlier mask, the residual spread, the heading it settled
    on and one standard deviation of that heading, or None when too few
    agree.

    `refine_yaw` asks it to solve for heading as well as translation. Off, the
    heading handed in stands and only the translation is fitted, which is what
    every measurement before this assumed. On, this becomes an iterated update
    in the sense FAST-LIO uses: the inliers chosen under one pose re-solve the
    pose, and the new pose reselects the inliers. Yaw and translation are not
    separable -- a heading error a hundredth of a radian wide drags the ground
    a centimetre sideways at every metre of range -- so the heading comes out
    of the same fit, as the 2D Procrustes rotation between the inlier sets.

    What comes back is the fit's own answer and how much that answer is worth,
    not a blend. Blending here was tried twice and belongs to the caller both
    times. A hard bound on the correction helped at 8 m/s and ruined 2.5 m/s,
    because a push of up to the bound on every solve is a random walk and a
    slower vehicle takes more solves per metre. Weighting the share by
    precision, the ordinary scalar gain, was no better on any drive: the
    gyro's error is a bias rather than noise, so a variance that is inflated
    and then paid off simply re-earns it on the next interval. Cancelling a
    bias means estimating it, and estimating it means a state, which is the
    caller's business.
    """
    body_points = np.ascontiguousarray(body_points, dtype=np.float64).reshape(-1, 2)
    world_points = np.ascontiguousarray(world_points, dtype=np.float64).reshape(-1, 2)
    if len(body_points) < max(2, min_inliers):
        return None
    if _fast is not None:
        given = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
        if len(given) != len(body_points):
            given = np.ones(len(body_points))
        return _fast.align_to_anchors(
            body_points, world_points, given, float(yaw), float(threshold),
            int(min_inliers), bool(refine_yaw))

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(body_points):
        weights = np.ones(len(body_points))

    given_yaw = yaw
    applied = float('inf')
    centre = None
    inliers = None
    for iteration in range(4):
        c = math.cos(yaw)
        s = math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        votes = world_points - (rotation @ body_points.T).T
        if centre is None:
            centre = np.median(votes, axis=0)
        for _ in range(3):
            residual = np.linalg.norm(votes - centre, axis=1)
            inliers = residual <= threshold
            if np.count_nonzero(inliers) < min_inliers:
                return None
            centre = np.average(votes[inliers], axis=0, weights=weights[inliers])
        if not refine_yaw or iteration == 3:
            break
        # The rotation that best carries the inlier body points onto their
        # anchors, about each set's own weighted centroid. Closed form in 2D:
        # no search, and the translation falls out of it unchanged.
        w = weights[inliers]
        b = body_points[inliers]
        t = world_points[inliers]
        b_centre = np.average(b, axis=0, weights=w)
        t_centre = np.average(t, axis=0, weights=w)
        db = b - b_centre
        dt = t - t_centre
        cross = float(np.sum(w * (db[:, 0] * dt[:, 1] - db[:, 1] * dt[:, 0])))
        dot = float(np.sum(w * (db[:, 0] * dt[:, 0] + db[:, 1] * dt[:, 1])))
        if abs(cross) < 1e-12 and abs(dot) < 1e-12:
            break
        proposed = wrap_pi(math.atan2(cross, dot))
        # How well this fit pins a rotation down, which is the caller's to
        # weigh: a residual of `fit` metres seen across a lever arm of
        # `lever` metres is an angle, and averaging it over the inliers
        # shrinks it by their root.
        total = float(np.sum(w))
        lever = math.sqrt(
            float(np.sum(w * (db[:, 0] ** 2 + db[:, 1] ** 2))) / max(total, 1e-9))
        residual = np.linalg.norm(votes[inliers] - centre, axis=1)
        fit = float(math.sqrt(float(np.mean(residual ** 2))))
        count = max(float(np.count_nonzero(inliers)), 1.0)
        if lever < 1e-6:
            break
        applied = fit / (lever * math.sqrt(count))
        updated = proposed
        if abs(wrap_pi(updated - yaw)) < 1e-6:
            yaw = updated
            break
        yaw = updated

    residual = np.linalg.norm(
        world_points - (np.array(
            [[math.cos(yaw), -math.sin(yaw)],
             [math.sin(yaw), math.cos(yaw)]]) @ body_points.T).T - centre,
        axis=1)
    inliers = residual <= threshold
    count = int(np.count_nonzero(inliers))
    if count < min_inliers:
        return None
    spread = float(math.sqrt(float(np.mean(residual[inliers] ** 2))))
    return centre, inliers, spread, yaw, applied


def fuse_by_precision(estimates):
    """Combine per-camera translations by how precise each one was.

    `estimates` are (x, y, count, spread) tuples. Weight is count over spread
    squared, the inverse variance of that camera's mean, so a camera that sees
    less ground but agrees with itself tightly is not drowned out by one that
    sees more and scatters. Weighting by count alone let the rear camera, which
    sits higher and sees more road, outvote the front twelve to one.
    """
    estimates = [e for e in estimates if e is not None]
    if not estimates:
        return None
    weights = []
    for _, _, count, spread in estimates:
        variance = max(float(spread), 1e-3) ** 2
        weights.append(max(float(count), 1.0) / variance)
    total = float(sum(weights))
    if total <= 0.0:
        return None
    x = sum(w * e[0] for w, e in zip(weights, estimates)) / total
    y = sum(w * e[1] for w, e in zip(weights, estimates)) / total
    return x, y, sum(int(e[2]) for e in estimates)
