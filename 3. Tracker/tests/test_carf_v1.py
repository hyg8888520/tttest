import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import importlib.util
import os
import pickle
import sys
import tempfile

import numpy as np

TRACKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if TRACKER_ROOT not in sys.path:
    sys.path.insert(0, TRACKER_ROOT)

from carf.auditor import CARFAuditor
from carf.policies import (authority_for_policy,
                           counterfactual_auditing_active)
from carf.inputs import load_topic_cache, sanity_check_inputs
from carf.evaluation import (metrics_delta_vs_baseline,
                             require_trackeval_compatible_gt,
                             trackeval_gt_conflicts)
from carf.oracle import OnlineGTAnchorOracle, match_detections_to_gt
from trackeval.datasets import MotChallenge2DBox
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
        self.assertFalse(tracker.auditing_active)
        self.assertIsNone(tracker.carf_auditor)

    def test_auditing_policies_execute_carf_auditor(self):
        for policy in ('audit_only', 'hard', 'soft'):
            with self.subTest(policy=policy):
                tracker = Tracker(
                    args(carf_enabled=True, carf_policy=policy,
                         carf_rollback_writes=[1]), 'synthetic')
                self.assertTrue(tracker.auditing_active)
                original = tracker.carf_auditor.audit_match
                tracker.carf_auditor.audit_match = Mock(wraps=original)
                rows = np.atleast_2d(detection((1.0, 0.0)))
                tracker.update(rows.copy(), rows.copy())
                tracker.update(rows.copy(), rows.copy())
                self.assertGreater(
                    tracker.carf_auditor.audit_match.call_count, 0)


class TestBEE24Adapter(unittest.TestCase):
    def test_sanity_reports_out_of_bounds_without_mutating_detections(self):
        with tempfile.TemporaryDirectory() as root:
            sequence = 'BEE2414'
            seq_dir = os.path.join(root, 'train', sequence)
            os.makedirs(seq_dir)
            with open(os.path.join(seq_dir, 'seqinfo.ini'), 'w') as handle:
                handle.write(
                    '[Sequence]\nimHeight=600\nimWidth=950\nseqLength=1\n')
            rows = np.asarray([
                [-3.0, 570.0, 35.0, 607.0, .9, 1.0, 1.0, 0.0]
            ])
            before = rows.copy()
            detections = {sequence: {1: rows}}
            sanity = sanity_check_inputs(
                detections, detections, root, [sequence], det_thr=.6)
            values = sanity['sequences'][sequence]
            self.assertEqual(sanity['status'], 'PASS')
            self.assertEqual(values['out_of_bounds_detections'], 1)
            self.assertEqual(values['clipped_detections'], 0)
            self.assertEqual(values['dropped_degenerate_detections'], 0)
            self.assertEqual(values['max_boundary_overflow']['left'], 3.0)
            self.assertEqual(values['max_boundary_overflow']['bottom'], 7.0)
            np.testing.assert_array_equal(rows, before)

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
            sanity = sanity_check_inputs(
                converted, converted, root, [sequence], det_thr=.6)
            self.assertEqual(sanity['status'], 'PASS')
            self.assertEqual(
                sanity['sequences'][sequence]['detections'], 2)


class TestOnlineGTAnchorOracle(unittest.TestCase):
    sequence = 'BEE2406'

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        gt_dir = os.path.join(
            self.tempdir.name, self.sequence, 'gt')
        os.makedirs(gt_dir)
        rows = np.asarray([
            [1, 101, 0, 0, 10, 10, 1, 1, 1],
            [2, 202, 0, 0, 10, 10, 1, 1, 1],
            [3, 101, 0, 0, 10, 10, 1, 1, 1],
            [6, 202, 0, 0, 10, 10, 1, 1, 1],
        ], dtype=float)
        np.savetxt(os.path.join(gt_dir, 'gt.txt'), rows, delimiter=',')
        self.oracle = OnlineGTAnchorOracle(
            self.tempdir.name, self.sequence, threshold=0.5)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def observation(feature=(1.0, 0.0), box=(0, 0, 10, 10)):
        item = Track(args(), detection(feature, box=box))
        item.frame_detection_index = 0
        return item

    def test_online_anchor_created_on_first_valid_gt_identity(self):
        track = initiated_track()
        observation = self.observation()
        self.oracle.prepare_frame(1, [observation])
        result = self.oracle.judge(1, track, observation)
        self.assertTrue(result['oracle_anchor_created'])
        self.assertEqual(result['oracle_anchor_gt_id'], 101)
        self.assertFalse(result['identity_contamination'])

    def test_frame_gt_matching_is_one_to_one(self):
        first = self.observation()
        second = self.observation()
        second.frame_detection_index = 1
        gt = self.oracle.gt_frames[1]
        mapping = match_detections_to_gt(gt, [first, second], threshold=0.5)
        self.assertEqual(len(mapping), 1)
        self.assertEqual(list(mapping.values()), [101])

    def test_online_anchor_never_changes(self):
        track = initiated_track()
        observation = self.observation()
        self.oracle.prepare_frame(1, [observation])
        self.oracle.judge(1, track, observation)
        self.oracle.prepare_frame(2, [observation])
        result = self.oracle.judge(2, track, observation)
        self.assertEqual(result['oracle_anchor_gt_id'], 101)
        self.assertTrue(result['identity_contamination'])
        self.assertEqual(self.oracle.anchors[(track.track_id, 1)], 101)

    def test_unknown_gt_does_not_suppress_or_create_anchor(self):
        track = initiated_track()
        observation = self.observation()
        self.oracle.prepare_frame(4, [observation])
        result = self.oracle.judge(4, track, observation)
        self.assertTrue(result['oracle_unknown'])
        self.assertIsNone(result['identity_contamination'])
        self.assertEqual(self.oracle.anchors, {})
        self.assertEqual(authority_for_policy(
            'oracle_gt', identity_contamination=None), 1.0)

    def test_cross_identity_detection_gives_zero_authority(self):
        track = initiated_track()
        observation = self.observation()
        self.oracle.prepare_frame(1, [observation])
        self.oracle.judge(1, track, observation)
        self.oracle.prepare_frame(2, [observation])
        result = self.oracle.judge(2, track, observation)
        self.assertEqual(authority_for_policy(
            'oracle_gt', identity_contamination=result[
                'identity_contamination']), 0.0)

    def test_track_instance_reuse_does_not_inherit_anchor(self):
        first = initiated_track()
        observation = self.observation()
        self.oracle.prepare_frame(1, [observation])
        self.oracle.judge(1, first, observation)

        reused = Track(args(), detection((1.0, 0.0)))
        reused.initiate(5, TrackCounter())
        self.assertEqual(first.track_id, reused.track_id)
        self.oracle.prepare_frame(6, [observation])
        result = self.oracle.judge(6, reused, observation)
        self.assertTrue(result['oracle_anchor_created'])
        self.assertEqual(result['oracle_anchor_gt_id'], 202)
        self.assertFalse(result['identity_contamination'])

    def test_oracle_never_changes_current_assignment(self):
        reference_oracle = OnlineGTAnchorOracle(
            self.tempdir.name, self.sequence, threshold=0.5)
        oracle_logger = MemoryLogger()
        oracle_tracker = Tracker(
            args(carf_enabled=True, carf_policy='oracle_gt',
                 carf_rollback_writes=[1]),
            self.sequence, audit_logger=oracle_logger, oracle=self.oracle)
        self.assertTrue(oracle_tracker.carf_enabled)
        self.assertFalse(oracle_tracker.auditing_active)
        self.assertIsNone(oracle_tracker.carf_auditor)
        audit_tracker = Tracker(
            args(carf_enabled=True, carf_policy='audit_only',
                 carf_rollback_writes=[1]),
            self.sequence, audit_logger=MemoryLogger(),
            oracle=reference_oracle)
        for frame_id, feature in ((1, (1.0, 0.0)),
                                  (2, (1.0, 0.0)),
                                  (3, (0.0, 1.0))):
            rows = np.atleast_2d(detection(feature))
            oracle_output = oracle_tracker.update(rows.copy(), rows.copy())
            audit_output = audit_tracker.update(rows.copy(), rows.copy())
            self.assertEqual([track.track_id for track in oracle_output],
                             [track.track_id for track in audit_output])
            for left, right in zip(oracle_output, audit_output):
                np.testing.assert_array_equal(left.x1y1wh, right.x1y1wh)
        diagnostics = oracle_tracker.write_diagnostics_summary()
        self.assertGreater(diagnostics['oracle_blocked_writes'], 0)
        self.assertEqual(oracle_logger.records, [])

    def test_oracle_gt_never_constructs_counterfactual_auditor(self):
        with patch('trackers.tracker.CARFAuditor',
                   side_effect=AssertionError('CARFAuditor must not run')):
            tracker = Tracker(
                args(carf_enabled=True, carf_policy='oracle_gt'),
                self.sequence, oracle=self.oracle)
            rows = np.atleast_2d(detection((1.0, 0.0)))
            tracker.update(rows.copy(), rows.copy())
            tracker.update(rows.copy(), rows.copy())
        self.assertFalse(counterfactual_auditing_active('oracle_gt', True))

    def test_oracle_unknown_gt_falls_back_to_baseline_write(self):
        class UnknownOracle:
            def __init__(self):
                self.prepared_frames = []

            def prepare_frame(self, frame_id, detections):
                self.prepared_frames.append(frame_id)

            def judge(self, frame_id, track, detection):
                return {'identity_contamination': None}

        oracle = UnknownOracle()
        tracker = Tracker(
            args(carf_enabled=True, carf_policy='oracle_gt'),
            'synthetic', oracle=oracle)
        first = np.atleast_2d(detection((1.0, 0.0)))
        second = np.atleast_2d(detection((0.9, 0.1)))
        tracker.update(first.copy(), first.copy())
        self.assertEqual(oracle.prepared_frames, [])
        tracker.update(second.copy(), second.copy())
        self.assertEqual(oracle.prepared_frames, [2])
        diagnostics = tracker.write_diagnostics_summary()
        self.assertEqual(diagnostics['accepted_write_opportunities'], 1)
        self.assertEqual(diagnostics['mean_authority_q'], 1.0)
        self.assertEqual(tracker.tracks[0].num_feature_writes, 1)


class TestV2Diagnostics(unittest.TestCase):
    def test_metrics_delta_uses_equal_weight_sequence_macro(self):
        baseline = {
            'HOTA': 0.5, 'AssA': 0.4, 'IDF1': 0.3, 'IDSW': 10,
            'per_sequence': {
                'A': {'HOTA': .2, 'AssA': .2, 'IDF1': .2, 'IDSW': 8},
                'B': {'HOTA': .8, 'AssA': .6, 'IDF1': .4, 'IDSW': 2},
            },
        }
        candidate = {
            'HOTA': 0.6, 'AssA': 0.5, 'IDF1': 0.4, 'IDSW': 8,
            'per_sequence': {
                'A': {'HOTA': .4, 'AssA': .4, 'IDF1': .4, 'IDSW': 4},
                'B': {'HOTA': .8, 'AssA': .6, 'IDF1': .4, 'IDSW': 2},
            },
        }
        delta = metrics_delta_vs_baseline(baseline, candidate)
        self.assertAlmostEqual(
            delta['macro_mean_per_sequence_delta']['delta_HOTA'], .1)
        self.assertEqual(
            delta['macro_mean_per_sequence_delta']['delta_IDSW'], -2.0)

    def test_contamination_risk_lift_uses_auditable_labeled_denominator(self):
        repo_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..'))
        path = os.path.join(repo_root, 'tools', 'analyze_carf_audit.py')
        spec = importlib.util.spec_from_file_location('carf_analysis_v2', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [
            {'auditable': True, 'identity_contamination': True,
             'F_ij': 1.0, 'authority_q': 1.0},
            {'auditable': True, 'identity_contamination': False,
             'F_ij': 0.0, 'authority_q': 1.0},
            {'auditable': False, 'identity_contamination': True,
             'F_ij': None, 'authority_q': 1.0},
        ]
        summary = module.population_summary(rows)
        self.assertEqual(
            summary['contamination_rate_gt_labeled_and_auditable'], 0.5)
        self.assertEqual(
            summary['risk_lift_F_eq_1_vs_auditable_labeled'], 2.0)

    def test_rollback_frame_age_after_suppressed_writes(self):
        track = initiated_track()
        track.update_features(np.asarray([[0.0, 1.0]]), 0.9,
                              authority=1.0, frame_id=2)
        track.update_features(np.asarray([[1.0, 0.0]]), 0.9,
                              authority=1.0, frame_id=3)
        track.update_features(np.asarray([[0.0, 1.0]]), 0.9,
                              authority=0.0, frame_id=4)
        track.update_features(np.asarray([[0.0, 1.0]]), 0.9,
                              authority=0.0, frame_id=5)
        snapshot = track.get_rollback_snapshot(1)
        self.assertEqual(snapshot['frame_id'], 2)
        self.assertEqual(5 - snapshot['frame_id'], 3)
        self.assertEqual(track.num_feature_writes, 2)


class TestVendoredTrackEval(unittest.TestCase):
    def test_gt_preflight_reports_all_conflicting_sequences(self):
        with tempfile.TemporaryDirectory() as root:
            for sequence, rows in {
                    'VALID': [[1, 1, 0, 0, 10, 10, 1, 1, 1]],
                    'BAD_A': [
                        [1, 7, 0, 0, 10, 10, 1, 1, 1],
                        [1, 7, 20, 0, 10, 10, 1, 1, 1],
                    ],
                    'BAD_B': [
                        [2, 8, 0, 0, 10, 10, 1, 1, 1],
                        [2, 8, 20, 0, 10, 10, 1, 1, 1],
                    ],
            }.items():
                gt_dir = os.path.join(root, sequence, 'gt')
                os.makedirs(gt_dir)
                np.savetxt(os.path.join(gt_dir, 'gt.txt'), rows, delimiter=',')

            sequences = ['VALID', 'BAD_A', 'BAD_B']
            conflicts = trackeval_gt_conflicts(root, sequences)
            self.assertEqual(set(conflicts), {'BAD_A', 'BAD_B'})
            with self.assertRaisesRegex(
                    ValueError, r'BAD_A: conflicts=1[\s\S]*BAD_B: conflicts=1'):
                require_trackeval_compatible_gt(root, sequences)

    def test_non_zip_tracker_subfolder_is_used_for_check_and_load(self):
        with tempfile.TemporaryDirectory() as root:
            sequence = 'BEE2406'
            gt_root = os.path.join(root, 'gt')
            tracker_root = os.path.join(root, 'trackers')
            gt_dir = os.path.join(gt_root, sequence, 'gt')
            result_dir = os.path.join(tracker_root, 'baseline', 'data')
            os.makedirs(gt_dir)
            os.makedirs(result_dir)
            with open(os.path.join(gt_dir, 'gt.txt'), 'w') as handle:
                handle.write('1,1,0,0,10,10,1,1,1\n')
            with open(os.path.join(result_dir, sequence + '.txt'), 'w') as handle:
                handle.write('1,7,0,0,10,10,0.9,-1,-1,-1\n')

            dataset = MotChallenge2DBox({
                'GT_FOLDER': gt_root,
                'TRACKERS_FOLDER': tracker_root,
                'OUTPUT_FOLDER': None,
                'TRACKERS_TO_EVAL': ['baseline'],
                'CLASSES_TO_EVAL': ['pedestrian'],
                'BENCHMARK': 'BEE24',
                'SPLIT_TO_EVAL': 'val',
                'INPUT_AS_ZIP': False,
                'PRINT_CONFIG': False,
                'DO_PREPROC': False,
                'TRACKER_SUB_FOLDER': 'data',
                'OUTPUT_SUB_FOLDER': '',
                'TRACKER_DISPLAY_NAMES': None,
                'SEQMAP_FOLDER': None,
                'SEQMAP_FILE': None,
                'SEQ_INFO': {sequence: 1},
                'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                'SKIP_SPLIT_FOL': True,
            })
            raw = dataset._load_raw_file('baseline', sequence, is_gt=False)
            self.assertEqual(raw['tracker_ids'][0].tolist(), [7])


if __name__ == '__main__':
    unittest.main()
