# CARF V1 baseline audit

## Provenance

- Upstream repository: `git@github.com:kamkyu94/TrackTrack.git`
- Audited upstream commit: `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`
- Local BEE24 reference: `TOPICTrack/` (read-only input-format reference)
- CARF scope: authority of an appearance write after an accepted match only

The audit was performed before CARF integration. TrackTrack's active decoder is
not a single Hungarian objective. The active association path is the iterative
mutual-nearest decoder described below. The LAP/Hungarian helper exists but its
call is commented out in the official path and CARF does not activate it.

## Appearance state and write site

| Question | Official location and interface |
| --- | --- |
| Mutable appearance state | `3. Tracker/trackers/track.py`, `Track.__init__`: `self.feat = detection[6:][np.newaxis, :].copy()` |
| Appearance update | `Track.update_features(feat, score)` |
| Baseline formula | `beta = alpha + (1-alpha)*(1-score)`; blend old/new; L2 normalize |
| Accepted-write caller | `Track.update(frame_id, detection)`, called only from the two accepted-match loops in `Tracker.update` |
| Deployed feature history | `Track.history[frame_id][4]`, saved after the feature update |

The CARF patch adds `authority=1.0` to the two functions. The `authority == 1.0`
branch contains the original arithmetic verbatim. `authority == 0.0` returns
without modifying `feat`. The remaining Kalman, box, score, velocity, lifecycle,
and history operations stay in `Track.update` and are unconditional.

## Official association call chain

```text
run.py: track / run_configured
  -> Tracker.update(dets, dets_95)
     -> find_deleted_detections
     -> Track(...) for each detection
     -> high / low / deleted-high split
     -> tracked+lost / new split
     -> CMC apply
     -> Kalman predict
     -> iterative_assignment(tracked_lost, high, low, deleted_high, ...)
        -> iou_distance
        -> cos_distance (reads Track.feat and detection.feat)
        -> conf_distance
        -> angle_distance
        -> penalty_p / penalty_q
        -> IoU <= 0.10 gating and [0,1] clipping
        -> repeated associate(cost, decreasing match threshold)
           -> row/column mutual nearest accepted under threshold
        -> official matches / unmatched tracks / unmatched detections
     -> accepted Track.update calls
     -> lost marking
     -> iterative_assignment(new, remaining_high, [], [], same interface)
     -> accepted Track.update calls
     -> removed marking and new-track birth
```

`trackers/utils.py:linear_assignment` wraps LAP but is inactive in
`iterative_assignment`. CARF counterfactuals call the same
`iterative_assignment` function with the same stage groups and all original
threshold, gating, motion, score, and decoder behavior.

## Multiple stages

There are exactly two online association stages in `Tracker.update`:

1. `tracked_lost`: Tracked and Lost tracks against high, low, and recovered
   deleted-high detections.
2. `new`: New tracks against unmatched high-confidence detections from stage 1.

Both stages may create accepted appearance writes. Unmatched Tracked/Lost tracks
are marked Lost; unmatched New tracks are Removed. Detection birth happens only
after both stages. CARF does not modify any of these decisions.

## History and feature formats

`Track.history` is a dictionary keyed by observation frame ID. Each value is:

```text
[box, score, mean, covariance, feat]
```

- `box`: copied `xyxy` observation
- `score`: scalar confidence
- `mean`: copied 8-D Kalman mean
- `covariance`: copied Kalman covariance
- `feat`: copied `1 x D` post-update appearance state

History is only appended on birth and accepted update, but its key is a frame,
and birth is not an accepted-match write. V1 therefore adds a bounded,
association-invisible deque of pre-write appearance snapshots. It is indexed by
committed appearance writes, not frames, and is never read by baseline
association. A suppressed `q=0` update is not counted as a write. The normal
history still stores the actual post-SSU feature.

Detection arrays have the official TrackTrack layout:

```text
[x1, y1, x2, y2, score, reserved/class slot, reid_feature...]
```

`Track.__init__` reads box columns `0:4`, score column `4`, and feature columns
`6:`. FastReID's extractor L2-normalizes embeddings before saving them.

## TrackEval

The vendored evaluator is `3. Tracker/trackeval`. The legacy entry point is
`utils/etc.py:evaluate`, which constructs `MotChallenge2DBox` plus the official
`HOTA`, `CLEAR`, and `Identity` metrics. CARF adds only a configurable wrapper
around these same classes. Formal results are reported as HOTA, DetA, AssA,
IDF1, IDSW, MOTA, and Frag; no metric is reimplemented.

The deprecated offline modal-GT diagnostic mirrors frame matching from
`trackeval/metrics/clear.py`. The runtime online-anchor Oracle instead performs
one frame-global detection-to-GT match solely to label already accepted edges.
These labels are analysis metadata, not a second IDSW metric. Formal IDSW still
comes only from TrackEval.

## BEE24/TOPIC interface audit

The local official TOPIC implementation stores:

- detections in `cache/det_<checkpoint-stem>.pkl`, keyed by `sequence:frame`;
- ReID features in `cache/embeddings/<sequence>_embedding.pkl`, keyed by
  `sequence:frame@one` and `sequence:frame@second`;
- the official five-sequence test list in
  `TOPICTrack/results/gt/seqmaps/BEE24-val.txt`.

The adapter reproduces TOPIC's published area filter `(432, 10710)`, score
floor `0.4`, high/second split `0.6`, and input resize scale before emitting the
TrackTrack array layout. It reads caches only. TOPIC publishes one NMS stream,
so that same stream is supplied as TrackTrack's companion detection stream;
this preserves observations and yields no synthetic deleted detections.

TrackTrack ships no BEE24 GMC file. The BEE24 YAML explicitly freezes identity
CMC as the BEE24 baseline adaptation. This setting is identical across all five
CARF policies and is not selected by CARF.
