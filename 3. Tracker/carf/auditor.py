"""Decoder-agnostic counterfactual fragility auditor.

The auditor calls TrackTrack's official full-stage ``iterative_assignment``.
It never substitutes a second decoder and never returns counterfactual matches
to the live tracker.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from trackers.utils import iterative_assignment


@dataclass(frozen=True)
class AuditResult:
    baseline_edge: tuple
    rollback_survived: dict
    valid_counterfactual_count: int
    survived_count: int
    survival: object
    fragility: object
    auditable: bool

    def as_log_fields(self):
        return {
            'baseline_edge': list(self.baseline_edge),
            'rollback_survived': {
                str(k): value for k, value in self.rollback_survived.items()
            },
            'valid_counterfactual_count': self.valid_counterfactual_count,
            'survived_count': self.survived_count,
            'S_ij': self.survival,
            'F_ij': self.fragility,
            'auditable': self.auditable,
        }


def _array_copy(value):
    return None if value is None else np.asarray(value).copy()


def _live_state_snapshot(tracks, detection_groups):
    """Capture association-visible state to prove the auditor is read-only."""
    track_state = []
    for track in tracks:
        history = tuple(
            (key, tuple(_array_copy(item) if isinstance(item, np.ndarray) else item
                        for item in value))
            for key, value in sorted(track.history.items())
        )
        track_state.append((
            _array_copy(track.feat), _array_copy(track.mean),
            _array_copy(track.covariance), _array_copy(track.velocity),
            _array_copy(track.box), track.score, track.state,
            track.end_frame_id, history,
        ))
    det_state = tuple(
        tuple((_array_copy(det.feat), _array_copy(det.box), det.score)
              for det in group)
        for group in detection_groups
    )
    return track_state, det_state


def _assert_same_item(before, after):
    if isinstance(before, np.ndarray):
        if not np.array_equal(before, after, equal_nan=True):
            raise RuntimeError('CARF auditor mutated live tracker/detection state')
    elif isinstance(before, (tuple, list)):
        if len(before) != len(after):
            raise RuntimeError('CARF auditor mutated live tracker/detection state')
        for left, right in zip(before, after):
            _assert_same_item(left, right)
    elif before != after:
        raise RuntimeError('CARF auditor mutated live tracker/detection state')


class CARFAuditor:
    """Audit accepted baseline edges by full-stage cloned reruns."""

    def __init__(self, rollback_writes=(1, 3), association_fn=None):
        rollback_writes = tuple(int(k) for k in rollback_writes)
        if not rollback_writes or any(k <= 0 for k in rollback_writes):
            raise ValueError('rollback_writes must contain positive integers')
        if len(set(rollback_writes)) != len(rollback_writes):
            raise ValueError('rollback_writes must be unique')
        self.rollback_writes = rollback_writes
        self.association_fn = association_fn or iterative_assignment

    @staticmethod
    def _clone_stage(tracks, detection_groups, target_track_idx,
                     rollback_feature):
        cloned_tracks, cloned_groups = deepcopy((tracks, detection_groups))
        cloned_tracks[target_track_idx].feat = np.asarray(
            rollback_feature).copy()
        return cloned_tracks, cloned_groups

    def audit_match(self, baseline_tracker_state, frame_detections,
                    accepted_match, association_stage, rollback_states,
                    association_kwargs):
        """Return survival of one already-accepted baseline edge.

        ``frame_detections`` is the exact ``(high, low, deleted_high)`` tuple
        supplied to the official association stage. ``association_stage`` is
        logged by callers and intentionally does not select another decoder.
        """
        del association_stage  # The official function is identical per stage.
        tracks = baseline_tracker_state
        detection_groups = tuple(frame_detections)
        if len(detection_groups) != 3:
            raise ValueError('frame_detections must be (high, low, deleted_high)')

        target_track_idx, target_det_idx = map(int, accepted_match)
        before = _live_state_snapshot(tracks, detection_groups)
        outcomes = {}

        try:
            for writes in self.rollback_writes:
                feature = rollback_states.get(writes)
                if feature is None:
                    outcomes[writes] = None
                    continue

                cloned_tracks, cloned_groups = self._clone_stage(
                    tracks, detection_groups, target_track_idx, feature)
                matches, _, _ = self.association_fn(
                    cloned_tracks,
                    cloned_groups[0], cloned_groups[1], cloned_groups[2],
                    association_kwargs['match_thr'],
                    association_kwargs['penalty_p'],
                    association_kwargs['penalty_q'],
                    association_kwargs['reduce_step'],
                    association_kwargs['frame_id'],
                )
                outcomes[writes] = any(
                    int(t) == target_track_idx and int(d) == target_det_idx
                    for t, d in matches
                )
        finally:
            after = _live_state_snapshot(tracks, detection_groups)
            _assert_same_item(before, after)

        valid = [value for value in outcomes.values() if value is not None]
        valid_count = len(valid)
        survived_count = sum(bool(value) for value in valid)
        survival = survived_count / valid_count if valid_count else None
        fragility = 1.0 - survival if survival is not None else None
        auditable = valid_count == len(self.rollback_writes)

        return AuditResult(
            baseline_edge=(target_track_idx, target_det_idx),
            rollback_survived=outcomes,
            valid_counterfactual_count=valid_count,
            survived_count=survived_count,
            survival=survival,
            fragility=fragility,
            auditable=auditable,
        )
