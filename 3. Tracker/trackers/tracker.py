from trackers.cmc import *
from trackers.utils import *
from trackers.track import *
from carf.auditor import CARFAuditor
from carf.policies import authority_for_policy, counterfactual_auditing_active


def _feature_cosine(before, after):
    before = np.asarray(before).reshape(-1)
    after = np.asarray(after).reshape(-1)
    denominator = np.linalg.norm(before) * np.linalg.norm(after)
    if denominator == 0:
        return None
    return float(np.dot(before, after) / denominator)


def _track_instance_key(track):
    birth_frame = min(track.history) if track.history else -1
    return int(track.track_id), int(birth_frame)


def _format_track_instance_key(key):
    return '%d@%d' % key


class Tracker(object):
    def __init__(self, args, vid_name, audit_logger=None, oracle=None):
        # Initialize
        self.args = args
        self.max_time_lost = args.max_time_lost

        # Initialize
        self.tracks = []
        self.frame_id = 0
        self.counter = TrackCounter()

        # Set global motion compensation model
        self.cmc = CMC(
            vid_name,
            cmc_dir=getattr(args, 'cmc_dir', './trackers/cmc'),
            identity=getattr(args, 'cmc_identity', False),
        )

        # CARF never supplies matches to the main tracker. It only determines
        # authority for the appearance update after official matches exist.
        self.sequence = vid_name
        self.carf_policy = getattr(args, 'carf_policy', 'baseline')
        self.carf_enabled = (bool(getattr(args, 'carf_enabled', False)) and
                             self.carf_policy != 'baseline')
        self.auditing_active = counterfactual_auditing_active(
            self.carf_policy, self.carf_enabled)
        self.carf_soft_power = float(getattr(args, 'carf_soft_power', 1.0))
        self.audit_logger = audit_logger if self.auditing_active else None
        self.oracle = oracle
        self._oracle_frame_detections = []
        self._oracle_prepared_frame = None
        self._write_stats = {
            'accepted_write_opportunities': 0,
            'gt_labeled_writes': 0,
            'identity_contaminating_writes': 0,
            'oracle_blocked_writes': 0,
            'suppressed_writes': 0,
            'authority_sum': 0.0,
        }
        self._track_write_diagnostics = {}
        self.carf_auditor = None
        if self.auditing_active:
            self.carf_auditor = CARFAuditor(
                getattr(args, 'carf_rollback_writes', [1, 3]))

    def _association_kwargs(self):
        return {
            'match_thr': self.args.match_thr,
            'penalty_p': self.args.penalty_p,
            'penalty_q': self.args.penalty_q,
            'reduce_step': self.args.reduce_step,
            'frame_id': self.frame_id,
        }

    def _update_accepted_matches(self, stage_tracks, detection_groups,
                                 matches, stage_name):
        flat_detections = (list(detection_groups[0]) +
                           list(detection_groups[1]) +
                           list(detection_groups[2]))

        if not self.carf_enabled:
            # Exact original update route when CARF is disabled.
            for t, d in matches:
                stage_tracks[t].update(self.frame_id, flat_detections[d])
            return

        if (self.oracle is not None and len(matches) > 0 and
                self._oracle_prepared_frame != self.frame_id):
            # The official association has already produced an accepted edge.
            # Prepare one frame-global GT mapping only now; it cannot affect
            # either official decoder call or the accepted assignment.
            self.oracle.prepare_frame(
                self.frame_id, self._oracle_frame_detections)
            self._oracle_prepared_frame = self.frame_id

        pending = []
        for t, d in matches:
            track = stage_tracks[t]
            detection = flat_detections[d]
            audit = None
            rollback_frame_ages = {}
            if self.auditing_active:
                rollback_snapshots = {
                    writes: track.get_rollback_snapshot(writes)
                    for writes in self.carf_auditor.rollback_writes
                }
                rollback_states = {
                    writes: (None if snapshot is None else snapshot['feature'])
                    for writes, snapshot in rollback_snapshots.items()
                }
                rollback_frame_ages = {
                    writes: (None if snapshot is None or
                             snapshot['frame_id'] is None else
                             int(self.frame_id - snapshot['frame_id']))
                    for writes, snapshot in rollback_snapshots.items()
                }
                audit = self.carf_auditor.audit_match(
                    baseline_tracker_state=stage_tracks,
                    frame_detections=detection_groups,
                    accepted_match=(t, d),
                    association_stage=stage_name,
                    rollback_states=rollback_states,
                    association_kwargs=self._association_kwargs(),
                )

            oracle_fields = {
                'gt_id': None,
                'oracle_anchor_gt_id': None,
                'identity_contamination': None,
                'oracle_unknown': True,
                'oracle_anchor_created': False,
                'oracle_label': None,
                'track_instance_key': _format_track_instance_key(
                    _track_instance_key(track)),
            }
            if self.oracle is not None:
                oracle_fields = self.oracle.judge(
                    self.frame_id, track, detection)

            authority = authority_for_policy(
                self.carf_policy,
                audit_result=audit,
                soft_power=self.carf_soft_power,
                identity_contamination=oracle_fields[
                    'identity_contamination'],
                enabled=self.carf_enabled,
            )
            pending.append((t, d, audit, oracle_fields, authority,
                            track.feat.copy(), track.num_feature_writes,
                            rollback_frame_ages))

        # Apply only after every counterfactual has observed the exact state
        # on which the official stage assignment was decoded.
        for (t, d, audit, oracle_fields, authority, feature_before,
             num_writes, rollback_frame_ages) in pending:
            track = stage_tracks[t]
            detection = flat_detections[d]
            instance_key = _track_instance_key(track)
            state = self._track_write_diagnostics.setdefault(instance_key, {
                'consecutive_q_zero': 0,
                'max_consecutive_q_zero': 0,
            })
            if authority == 0.0:
                state['consecutive_q_zero'] += 1
                state['max_consecutive_q_zero'] = max(
                    state['max_consecutive_q_zero'],
                    state['consecutive_q_zero'])
            else:
                state['consecutive_q_zero'] = 0
            track.update(self.frame_id, detection, authority=authority)

            stats = self._write_stats
            stats['accepted_write_opportunities'] += 1
            stats['authority_sum'] += float(authority)
            if authority < 1.0:
                stats['suppressed_writes'] += 1
            contamination = oracle_fields['identity_contamination']
            if contamination is not None:
                stats['gt_labeled_writes'] += 1
            if contamination is True:
                stats['identity_contaminating_writes'] += 1
            if (self.carf_policy == 'oracle_gt' and
                    contamination is True and authority == 0.0):
                stats['oracle_blocked_writes'] += 1

            if self.audit_logger is None:
                continue
            fields = audit.as_log_fields()
            for writes, survived in audit.rollback_survived.items():
                fields['rollback_%d_survived' % writes] = survived
                fields['rollback_%d_frame_age' % writes] = (
                    rollback_frame_ages[writes])
            last_update = track.appearance_last_update_frame
            fields.update({
                'sequence': self.sequence,
                'frame_id': self.frame_id,
                'track_id': int(track.track_id),
                'association_stage': stage_name,
                'detection_index': int(getattr(detection, 'frame_detection_index', d)),
                'stage_detection_index': int(d),
                'detection_score': float(detection.score),
                'num_previous_feature_writes': int(num_writes),
                'policy': self.carf_policy,
                'authority_q': float(authority),
                'feature_cos_before_after': _feature_cosine(feature_before, track.feat),
                'consecutive_q_zero': state['consecutive_q_zero'],
                'max_consecutive_q_zero': state['max_consecutive_q_zero'],
                'frames_since_last_effective_appearance_update': (
                    None if last_update is None else
                    int(self.frame_id - last_update)),
                **oracle_fields,
            })
            self.audit_logger.write(fields)

    def write_diagnostics_summary(self):
        stats = dict(self._write_stats)
        total = stats['accepted_write_opportunities']
        labeled = stats['gt_labeled_writes']
        stats['mean_authority_q'] = (
            stats.pop('authority_sum') / total if total else None)
        stats['suppressed_write_ratio'] = (
            stats['suppressed_writes'] / total if total else None)
        stats['identity_contamination_rate'] = (
            stats['identity_contaminating_writes'] / labeled
            if labeled else None)
        stats['oracle_blocked_rate'] = (
            stats['oracle_blocked_writes'] / labeled if labeled else None)
        stats['maximum_consecutive_q_zero_per_track'] = max(
            (value['max_consecutive_q_zero']
             for value in self._track_write_diagnostics.values()),
            default=0)
        stats['per_track_max_consecutive_q_zero'] = {
            _format_track_instance_key(key): value['max_consecutive_q_zero']
            for key, value in sorted(self._track_write_diagnostics.items())
        }
        return stats

    def init_tracks(self, dets):
        # Get alive tracks, iou_similarity, and scores
        tracks = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.New]
        iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
        scores = np.array([d.score for d in dets])

        # Run track aware NMS
        allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.init_thr)

        for idx, flag in enumerate(allow_indices):
            if flag:
                dets[idx].initiate(self.frame_id, self.counter)
                self.tracks.append(dets[idx])

    def update(self, dets, dets_95):
        # ==============================================================================================================
        # Update frame id
        self.frame_id += 1

        # Get deleted detections &  Encode
        dets_del = find_deleted_detections(dets, dets_95)
        dets = [Track(self.args, d) for d in dets]
        dets_del = [Track(self.args, d) for d in dets_del]
        for detection_index, detection in enumerate(dets):
            detection.frame_detection_index = detection_index
        for detection_index, detection in enumerate(dets_del, start=len(dets)):
            detection.frame_detection_index = detection_index
        # Keep the complete observation set for lazy Oracle preparation after
        # the official decoder has produced at least one accepted edge.
        self._oracle_frame_detections = dets + dets_del

        # Divide detections
        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]

        # Split tracks
        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(tracked_lost, warp_matrix)
        apply_cmc(new, warp_matrix)

        # Predict the current location with KF
        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]

        # ==============================================================================================================
        # Association between (tracked and lost tracks) & (high confidence detections)
        dets = dets_high + dets_low + dets_del_high
        matches, u_tracks, u_dets = iterative_assignment(tracked_lost, dets_high, dets_low, dets_del_high,
                                                         self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id)

        # Update matched tracks
        self._update_accepted_matches(
            tracked_lost, (dets_high, dets_low, dets_del_high), matches,
            'tracked_lost')

        # Mark "lost" to unmatched tracks
        for t in u_tracks:
            tracked_lost[t].mark_lost()

        # ==============================================================================================================
        # Get remained high confidence detections
        dets_high_left = [dets[i] for i in u_dets if i < len(dets_high)]

        # Association between (new tracks) & (left high confidence detections)
        matches, u_tracks, u_dets = iterative_assignment(new, dets_high_left, [], [], self.args.match_thr,
                                                         self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id)

        # Update matched tracks
        self._update_accepted_matches(
            new, (dets_high_left, [], []), matches, 'new')

        # Mark "remove" to unmatched tracks
        for t in u_tracks:
            new[t].mark_removed()

        # ==============================================================================================================
        # Mark "remove" lost tracks which are too old and add to finished
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        # Init new tracks
        self.init_tracks([dets_high_left[udx] for udx in u_dets])

        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
        # Update frame id
        self.frame_id += 1

        # Only maintain already tracked and new tracks, Drop all the new tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(self.tracks, warp_matrix)

        # Predict the current location with KF
        [t.predict() for t in self.tracks]

        # Change every track as lost tracks
        for t in self.tracks:
            t.mark_lost()
            
        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return []


