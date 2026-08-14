import unittest
from types import SimpleNamespace
import os
import pickle
import tempfile

import numpy as np

from carf.auditor import CARFAuditor
from carf.policies import authority_for_policy
from carf.inputs import load_topic_cache
from trackers.track import Track, TrackCounter
from trackers.tracker import Tracker
from trackers.utils import iterative_assignment


def args(**overrides):
    values = dict(
        data_path='BEE24', min_len=1, max_time_lost=30,
        penalty_p=0.20, penalty_q=0.40, reduce_step=0.05,
        tai_thr=0.55, det_thr=0.60, init_thr=0.60, match_thr=0.80,
        carf_enabled=False, carf_policy='baseline',
        carf_rollback_writes=[1, 3], carf_soft_power=1.0,
        cmc_identity=True, cmc_dir='unused',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def detection(feature, score=0.9, box=(0.0, 0.0, 10.0, 10.0)):
    return np.asarray([*box, score, 1.0, *feature], dtype=np.float64)


def initiated_track(feature=(1.0, 0.0), box=(0.0, 0.0, 10.0, 10.0)):
    track = Track(args(), detection(feature, box=box))
    track.initiate(1, TrackCounter())
    return track


class MemoryLogger:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


class TestSSU(unittest.TestCase):
    def test_authority_one_exact_baseline(self):
        track = Track(args(), detection((0.6, 0.8)))
        old = track.feat.copy()
        new = np.asarray([[0.8, 0.6]], dtype=np.float64)
        score = 0.73
        beta = track.alpha + (1 - track.alpha) * (1 - score)
        expected = beta * old + (1 - beta) * new
        expected /= np.linalg.norm(expected)

        track.update_features(new.copy(), score, authority=1.0)
        self.assertTrue(np.array_equal(track.feat, expected))

    def test_authority_zero_keeps_feature(self):
        track = Track(args(), detection((0.6, 0.8)))
        before = track.feat.copy()
        track.update_features(np.asarray([[1.0, 0.0]]), 0.7, authority=0.0)
        self.assertTrue(np.array_equal(track.feat, before))
        self.assertEqual(track.num_feature_writes, 0)

    def test_soft_authority_formula(self):
        track = Track(args(), detection((0.6, 0.8)))
        old = track.feat.copy()
        new = np.asarray([[0.8, 0.6]], dtype=np.float64)
        score, authority = 0.73, 0.35
        beta = track.alpha + (1 - track.alpha) * (1 - score)
        effective = authority * (1.0 - beta)
        expected = (1.0 - effective) * old + effective * new
        expected /= np.linalg.norm(expected)

        track.update_features(new.copy(), score, authority=authority)
        np.testing.assert_allclose(track.feat, expected, rtol=0, atol=1e-15)

    def test_rollback_by_write_not_by_frame(self):
        track = Track(args(carf_rollback_writes=[1, 2]),
                      detection((1.0, 0.0)))
        initial = track.feat.copy()
        track.update_features(np.asarray([[0.0, 1.0]]), 0.9, authority=1.0)
        after_first_write = track.feat.copy()
        # No frame ID is supplied to appearance writes.
        track.update_features(np.asarray([[-1.0, 0.0]]), 0.9, authority=1.0)
        np.testing.assert_array_equal(track.get_rollback_feature(1),
                                      after_first_write)
        np.testing.assert_array_equal(track.get_rollback_feature(2), initial)


class TestAuditor(unittest.TestCase):
    association_kwargs = dict(match_thr=0.8, penalty_p=0.2, penalty_q=0.4,
                              reduce_step=0.05, frame_id=2)

    def test_auditor_does_not_mutate_main_tracker(self):
        track = initiated_track()
        dets = [Track(args(), detection((1.0, 0.0)))]
        before = (track.feat.copy(), track.mean.copy(), track.box.copy(),
                  dets[0].feat.copy())
        CARFAuditor([1]).audit_match(
            [track], (dets, [], []), (0, 0), 'tracked_lost',
            {1: np.asarray([[0.0, 1.0]])}, self.association_kwargs)
        after = (track.feat, track.mean, track.box, dets[0].feat)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def test_counterfactual_changes_only_target_track_feature(self):
        tracks = [initiated_track((1.0, 0.0), (0, 0, 10, 10)),
                  initiated_track((0.0, 1.0), (20, 0, 30, 10))]
        dets = [Track(args(), detection((1.0, 0.0)))]
        originals = [track.feat.copy() for track in tracks]
        rollback = np.asarray([[-1.0, 0.0]])

        def inspecting_association(cloned_tracks, high, low, deleted, *unused):
            np.testing.assert_array_equal(cloned_tracks[0].feat, rollback)
            np.testing.assert_array_equal(cloned_tracks[1].feat, originals[1])
            self.assertIsNot(cloned_tracks[0], tracks[0])
            self.assertIsNot(high[0], dets[0])
            return [[0, 0]], [1], []

        result = CARFAuditor([1], association_fn=inspecting_association).audit_match(
            tracks, (dets, [], []), (0, 0), 'tracked_lost',
            {1: rollback}, self.association_kwargs)
        self.assertEqual(result.survival, 1.0)
        for original, track in zip(originals, tracks):
            np.testing.assert_array_equal(original, track.feat)

    def test_insufficient_history_falls_back_to_baseline(self):
        track = initiated_track()
        dets = [Track(args(), detection((1.0, 0.0)))]
        result = CARFAuditor([1, 3]).audit_match(
            [track], (dets, [], []), (0, 0), 'tracked_lost',
            {1: np.asarray([[1.0, 0.0]]), 3: None},
            self.association_kwargs)
        self.assertFalse(result.auditable)
        self.assertEqual(authority_for_policy('hard', result), 1.0)
        self.assertEqual(authority_for_policy('soft', result), 1.0)

    def test_synthetic_rollback_changes_association_s_less_than_one(self):
        track = initiated_track((1.0, 0.0))
        dets = [Track(args(), detection((1.0, 0.0))),
                Track(args(), detection((0.0, 1.0)))]
        baseline, _, _ = iterative_assignment(
            [track], dets, [], [], **self.association_kwargs)
        self.assertIn([0, 0], baseline)
        result = CARFAuditor([1]).audit_match(
            [track], (dets, [], []), (0, 0), 'tracked_lost',
            {1: np.asarray([[0.0, 1.0]])}, self.association_kwargs)
        self.assertTrue(result.auditable)
        self.assertLess(result.survival, 1.0)

    def test_synthetic_rollback_keeps_association_s_equal_one(self):
        track = initiated_track((1.0, 0.0))
        dets = [Track(args(), detection((1.0, 0.0))),
                Track(args(), detection((0.0, 1.0)))]
        result = CARFAuditor([1]).audit_match(
            [track], (dets, [], []), (0, 0), 'tracked_lost',
            {1: np.asarray([[0.9, 0.1]])}, self.association_kwargs)
        self.assertEqual(result.survival, 1.0)
        self.assertEqual(result.fragility, 0.0)


class TestEquivalence(unittest.TestCase):
    def test_audit_only_tracking_output_equivalence(self):
        baseline = Tracker(args(carf_enabled=False), 'synthetic')
        audit_logger = MemoryLogger()
        audit_only = Tracker(
            args(carf_enabled=True, carf_policy='audit_only',
                 carf_rollback_writes=[1]),
            'synthetic', audit_logger=audit_logger)
        baseline_outputs, audit_outputs = [], []
        for frame_id in range(1, 7):
            rows = np.atleast_2d(detection(
                (1.0, 0.0), score=0.9,
                box=(frame_id, 0.0, frame_id + 10.0, 10.0)))
            baseline_result = baseline.update(rows.copy(), rows.copy())
            audit_result = audit_only.update(rows.copy(), rows.copy())
            baseline_outputs.append([
                (track.track_id, track.x1y1wh.copy()) for track in baseline_result])
            audit_outputs.append([
                (track.track_id, track.x1y1wh.copy()) for track in audit_result])

        self.assertEqual(len(baseline_outputs), len(audit_outputs))
        for baseline_frame, audit_frame in zip(baseline_outputs, audit_outputs):
            self.assertEqual([item[0] for item in baseline_frame],
                             [item[0] for item in audit_frame])
            for baseline_item, audit_item in zip(baseline_frame, audit_frame):
                np.testing.assert_array_equal(baseline_item[1], audit_item[1])
        self.assertGreater(len(audit_logger.records), 0)

    def test_policy_baseline_uses_disabled_route(self):
        tracker = Tracker(args(carf_enabled=True, carf_policy='baseline'),
                          'synthetic')
        self.assertFalse(tracker.carf_enabled)
        self.assertIsNone(tracker.carf_auditor)


class TestBEE24Adapter(unittest.TestCase):
    def test_topic_cache_adapter_preserves_row_order_and_scale(self):
        with tempfile.TemporaryDirectory() as root:
            sequence = 'BEE2401'
            seq_dir = os.path.join(root, 'train', sequence)
            emb_dir = os.path.join(root, 'embeddings')
            os.makedirs(seq_dir)
            os.makedirs(emb_dir)
            with open(os.path.join(seq_dir, 'seqinfo.ini'), 'w') as handle:
                handle.write('[Sequence]\nimHeight=100\nimWidth=200\nseqLength=2\n')
            det_path = os.path.join(root, 'detections.pkl')
            raw = np.asarray([[0, 0, 20, 20, .8],
                              [20, 0, 40, 20, .5]], dtype=float)
            with open(det_path, 'wb') as handle:
                pickle.dump({sequence + ':1': raw}, handle)
            with open(os.path.join(emb_dir, sequence + '_embedding.pkl'), 'wb') as handle:
                pickle.dump({sequence + ':1@one': np.asarray([[1., 0.]]),
                             sequence + ':1@second': np.asarray([[0., 1.]])}, handle)

            converted = load_topic_cache(
                det_path, emb_dir, root, input_size=(100, 200),
                area_range=(0, 1000), score_floor=.4, embedding_split=.6)
            np.testing.assert_array_equal(converted[sequence][1][:, :4],
                                          raw[:, :4])
            np.testing.assert_array_equal(converted[sequence][1][:, 6:],
                                          np.eye(2))
            self.assertIsNone(converted[sequence][2])


if __name__ == '__main__':
    unittest.main()
