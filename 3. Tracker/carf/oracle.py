"""Offline GT oracle write protection.

OFFLINE ORACLE - NOT A DEPLOYABLE METHOD.

Canonical identities are prepared from a complete baseline result sequence.
The live tracker still accepts and outputs its own baseline association; this
module only decides whether that accepted observation may write appearance.
"""

from collections import Counter, defaultdict
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackeval.datasets._base_dataset import _BaseDataset


ORACLE_SCHEMA_VERSION = 'carf.oracle.v1'
ORACLE_LABEL = 'OFFLINE ORACLE - NOT A DEPLOYABLE METHOD'


def _as_2d(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.atleast_2d(rows)


def load_mot_rows(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return np.empty((0, 10), dtype=np.float64)
    return _as_2d(np.loadtxt(path, delimiter=',', dtype=np.float64))


def rows_by_frame(rows, is_gt=False):
    grouped = defaultdict(list)
    for row in _as_2d(rows):
        if is_gt and len(row) >= 8:
            if int(row[6]) == 0 or int(row[7]) != 1:
                continue
        grouped[int(row[0])].append(row)
    return {frame: _as_2d(values) for frame, values in grouped.items()}


def _xyxy_to_xywh(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    if boxes.size:
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
    return boxes


def _iou_xywh(gt_boxes, tracker_boxes):
    return _BaseDataset._calculate_box_ious(
        np.asarray(gt_boxes, dtype=np.float64),
        np.asarray(tracker_boxes, dtype=np.float64),
        box_format='xywh')


def clear_sequence_matches(gt_frames, tracker_frames, threshold=0.5):
    """Replicate TrackEval CLEAR frame matching and emit labels, not metrics."""
    previous_tracker_for_gt = {}
    previous_timestep_tracker_for_gt = {}
    matches = []
    idsw_frames = []
    all_frames = sorted(set(gt_frames) | set(tracker_frames))

    for frame_id in all_frames:
        gt = gt_frames.get(frame_id, np.empty((0, 10)))
        tracker = tracker_frames.get(frame_id, np.empty((0, 10)))
        if len(gt) == 0 or len(tracker) == 0:
            previous_timestep_tracker_for_gt = {}
            continue

        gt_ids = gt[:, 1].astype(int)
        tracker_ids = tracker[:, 1].astype(int)
        similarity = _iou_xywh(gt[:, 2:6], tracker[:, 2:6])
        continuity = np.zeros_like(similarity, dtype=bool)
        for row, gt_id in enumerate(gt_ids):
            previous = previous_timestep_tracker_for_gt.get(int(gt_id))
            if previous is not None:
                continuity[row] = tracker_ids == previous
        score_matrix = 1000.0 * continuity + similarity
        score_matrix[similarity < threshold - np.finfo(float).eps] = 0.0

        match_rows, match_cols = linear_sum_assignment(-score_matrix)
        valid = score_matrix[match_rows, match_cols] > np.finfo(float).eps
        match_rows, match_cols = match_rows[valid], match_cols[valid]
        current_timestep = {}
        frame_has_switch = False
        for row, col in zip(match_rows, match_cols):
            gt_id = int(gt_ids[row])
            tracker_id = int(tracker_ids[col])
            previous = previous_tracker_for_gt.get(gt_id)
            is_switch = previous is not None and previous != tracker_id
            frame_has_switch = frame_has_switch or is_switch
            matches.append({
                'frame_id': int(frame_id),
                'gt_id': gt_id,
                'track_id': tracker_id,
                'iou': float(similarity[row, col]),
                'idsw_event': bool(is_switch),
            })
            previous_tracker_for_gt[gt_id] = tracker_id
            current_timestep[gt_id] = tracker_id
        if frame_has_switch:
            idsw_frames.append(int(frame_id))
        previous_timestep_tracker_for_gt = current_timestep
    return matches, idsw_frames


def build_oracle_artifact(gt_root, tracker_results, sequences, output_path,
                          threshold=0.5):
    artifact = {
        'schema_version': ORACLE_SCHEMA_VERSION,
        'label': ORACLE_LABEL,
        'matching': 'TrackEval CLEAR-compatible IoU/Hungarian matching',
        'iou_threshold': float(threshold),
        'sequences': {},
    }
    for sequence in sequences:
        gt_path = resolve_gt_path(gt_root, sequence)
        tracker_path = os.path.join(tracker_results, sequence + '.txt')
        gt_frames = rows_by_frame(load_mot_rows(gt_path), is_gt=True)
        tracker_frames = rows_by_frame(load_mot_rows(tracker_path))
        matches, idsw_frames = clear_sequence_matches(
            gt_frames, tracker_frames, threshold=threshold)
        counts = defaultdict(Counter)
        frame_matches = defaultdict(dict)
        for match in matches:
            counts[match['track_id']][match['gt_id']] += 1
            frame_matches[str(match['frame_id'])][str(match['track_id'])] = match['gt_id']
        canonical = {}
        for track_id, gt_counts in counts.items():
            canonical[str(track_id)] = min(
                (gt_id for gt_id, count in gt_counts.items()
                 if count == max(gt_counts.values())))
        artifact['sequences'][sequence] = {
            'canonical_gt_id': canonical,
            'frame_matches': dict(frame_matches),
            'idsw_frames': idsw_frames,
        }

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    return artifact


def resolve_gt_path(gt_root, sequence):
    candidates = [
        os.path.join(gt_root, sequence, 'gt', 'gt.txt'),
        os.path.join(gt_root, 'BEE24-val', sequence, 'gt', 'gt.txt'),
        os.path.join(gt_root, 'train', sequence, 'gt', 'gt.txt'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError('GT not found for %s under %s' % (sequence, gt_root))


class OfflineGTOracle:
    """Read-only runtime oracle backed by a frozen baseline artifact."""

    label = ORACLE_LABEL

    def __init__(self, gt_root, artifact_path, sequence, threshold=0.5):
        with open(artifact_path, 'r', encoding='utf-8') as handle:
            artifact = json.load(handle)
        if artifact.get('schema_version') != ORACLE_SCHEMA_VERSION:
            raise ValueError('unsupported oracle artifact schema')
        self.sequence = sequence
        self.threshold = float(threshold)
        sequence_data = artifact.get('sequences', {}).get(sequence, {})
        self.canonical = {
            int(key): int(value)
            for key, value in sequence_data.get('canonical_gt_id', {}).items()
        }
        self.idsw_frames = set(sequence_data.get('idsw_frames', []))
        gt_rows = load_mot_rows(resolve_gt_path(gt_root, sequence))
        self.gt_frames = rows_by_frame(gt_rows, is_gt=True)

    def match_observations(self, frame_id, detections):
        """Map stage-local detections to GT using CLEAR's IoU primitive."""
        gt = self.gt_frames.get(int(frame_id), np.empty((0, 10)))
        mapping = {}
        if len(gt) == 0 or len(detections) == 0:
            return mapping
        det_xyxy = np.asarray([det.x1y1x2y2 for det in detections],
                              dtype=np.float64)
        similarity = _iou_xywh(gt[:, 2:6], _xyxy_to_xywh(det_xyxy))
        scores = similarity.copy()
        scores[similarity < self.threshold - np.finfo(float).eps] = 0.0
        rows, cols = linear_sum_assignment(-scores)
        valid = scores[rows, cols] > np.finfo(float).eps
        for row, col in zip(rows[valid], cols[valid]):
            mapping[int(col)] = int(gt[row, 1])
        return mapping

    def judge(self, frame_id, track_id, detection_index, detections):
        observation_gt = self.match_observations(frame_id, detections).get(
            int(detection_index))
        canonical_gt = self.canonical.get(int(track_id))
        if observation_gt is None or canonical_gt is None:
            correct = None
        else:
            correct = observation_gt == canonical_gt
        return {
            'gt_id': observation_gt,
            'canonical_gt_id': canonical_gt,
            'oracle_correct_write': correct,
            'oracle_unknown': correct is None,
            'idsw_event': int(frame_id) in self.idsw_frames,
            'oracle_label': ORACLE_LABEL,
        }
