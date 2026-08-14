# CARF + SSU V1: first real experiment

## Question and limits

This experiment asks whether a fragile accepted association can pollute a
track's mutable appearance state and amplify later identity errors. CARF
fragility measures sensitivity of TrackTrack's official association result to
recent appearance state. It is not correctness, an error probability, or a
calibrated continuous score. With rollback writes `[1, 3]`, fully auditable
rows normally have `F` in `{0, 0.5, 1}`.

CARF never changes the current-frame assignment. Counterfactuals deep-clone the
official stage, replace only the target track feature, and rerun the same
`iterative_assignment` function. Detection, ReID, motion/Kalman state, gating,
track lifecycle, output IDs, boxes, and velocities remain baseline behavior.

## SSU policies

The original TrackTrack feature write is preserved verbatim when `q=1`. `q=0`
leaves the appearance feature byte-for-byte unchanged. For `0<q<1`, SSU scales
only the original appearance innovation.

- `baseline`: CARF disabled; original update route.
- `audit_only`: compute/log CARF; `q=1` exact baseline update.
- `oracle_gt`: online GT-anchor Oracle controls only the accepted write.
- `soft`: primary intervention, `q=S` (`soft_power=1`).
- `hard`: aggressive ablation, `q=1` only when fully auditable `S=1`.

Insufficient rollback history always gives `auditable=false` and `q=1`.

## Online GT identity-anchor Oracle

`oracle_gt` is explicitly:

```text
OFFLINE ORACLE - NOT A DEPLOYABLE METHOD: online GT identity anchor
```

At each frame, detections are matched once to GT with deterministic one-to-one
Hungarian matching on IoU at the fixed, untuned threshold `0.5`. This matching
labels observations only and is unrelated to TrackTrack's association decoder.
Both official TrackTrack association stages share this frame-global mapping.

After TrackTrack accepts an edge:

1. The first valid GT identity for `(track_id, birth_frame)` becomes its
   immutable oracle anchor.
2. A later known observation with the same identity gets `q=1`.
3. A later known observation with another identity is labeled
   `identity_contamination=true` and gets `q=0` under `oracle_gt`.
4. Unknown/unmatched observations get `q=1`, never create/change an anchor,
   and are excluded from contamination statistics.

The Oracle cannot reject detections, select another edge, or change the current
assignment. The first valid observation can itself be wrong, so
`identity_contamination` means disagreement with the track instance's initial
known identity. It is not complete causal harmful-write ground truth. Same-ID
low-quality or occluded writes can still be harmful.

The old `tools/prepare_carf_oracle.py` modal-trajectory artifact remains only as
deprecated offline diagnostic code. Runtime `oracle_gt` does not read it and
does not require `BEE24_ORACLE_ARTIFACT`.

## BEE24 V2 splits

Use `configs/carf/manifests/bee24_carf_v2.yaml`. Sequence IDs exactly match TOPIC
cache keys, for example `BEE2406`, not `BEE24-06`.

- Gate 0: `BEE2406`. `BEE2414` is excluded because its raw training GT has
  conflicting boxes sharing one `(frame, identity)` pair and therefore is not
  valid TrackEval input. CARF does not rewrite or guess GT identities.
- `development_core`: 06, 10, 26, 29.
- `development_long` and `stress_dense_long` are empty in V2 because their
  former sequences do not satisfy TrackEval's per-frame ID uniqueness rule.
- `excluded_non_trackeval`: 14, 15, 33, 35. Their conflicting `(frame, ID)`
  counts are respectively 1, 1, 959, and 36. The raw GT is never rewritten.
- Official test: 12, 13, 16, 18, 20, frozen and never used for tuning.

The old hash-based v1 manifest is retained but marked deprecated.
`development_core` is not described as a 20% validation split.

## Environment

Nothing is downloaded. TOPIC detector/ReID caches must already exist.

```bash
python -m pip install -r requirements-carf.txt
export BEE24_ROOT=/path/to/BEE24
export BEE24_DETECTIONS=/path/to/TOPICTrack/cache/det_bee24.pkl
export BEE24_REID_FEATURES=/path/to/TOPICTrack/cache/embeddings
export BEE24_DET_CKPT=/path/to/topictrack_bee_detector.pth.tar
export BEE24_REID_CKPT=/path/to/BEE24_AGW.pth
export BEE24_CMC_ROOT=/optional/unused/cmc/path
export OUTPUT_ROOT=/path/to/output/carf_v2
```

Checkpoint paths are provenance fields when cached observations are used.
Identity CMC is frozen for BEE24 and does not read `BEE24_CMC_ROOT`.

## Phase 0: data sanity

This validates sequence/frame keys, detection and embedding row alignment,
finite/positive boxes, feature dimensions, and high/low detection counts. It
also reports out-of-image detector boxes, their ratio, coordinate extrema, and
maximum boundary overflow. It never regenerates a TOPIC cache.

Out-of-image detector boxes are not clipped or dropped. TOPIC's official cache
contains raw YOLOX geometry, and its tracker divides coordinates by the resize
scale without clipping association boxes; only the local ReID crop copy is
clipped. Changing association boxes here would alter IoU, Kalman updates, and
tracking output, violating baseline preservation. Non-finite values,
non-positive boxes, cache-key errors, and detection/embedding misalignment
remain hard failures.

```bash
python "3. Tracker/run.py" --sanity-only \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=development_core
```

Expected output: `$OUTPUT_ROOT/data_sanity.json`. Structural/format mismatches
fail loudly; boundary overflow is recorded for diagnosis without mutation.

## Phase 1: real-data Gate 0

Use a dedicated output root and only the two Gate 0 sequences.

```bash
export OUTPUT_ROOT=/path/to/output/carf_v2_gate0

python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=gate0 carf.enabled=false carf.policy=baseline

python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=gate0 carf.enabled=true carf.policy=audit_only

python tools/compare_tracking_outputs.py \
  --baseline "$OUTPUT_ROOT/baseline/data" \
  --candidate "$OUTPUT_ROOT/audit_only/data" \
  --output "$OUTPUT_ROOT/equivalence.json"
```

Every `(frame, track_id, bbox)` must be exactly equal. Stop on any difference.

## Phase 2: online GT Oracle

Run baseline and Oracle on core plus the long-horizon sequence. No baseline
identity artifact preparation is needed.

```bash
export OUTPUT_ROOT=/path/to/output/carf_v2_oracle

python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=oracle_development carf.enabled=false carf.policy=baseline

python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=oracle_development carf.enabled=true carf.policy=oracle_gt
```

Inspect combined and per-sequence `metrics.json`,
`metrics_delta_vs_baseline.json`, and `write_diagnostics.json`. Interpret event
count and metrics jointly:

- Very rare contamination means cross-ID appearance contamination is not a
  major observed baseline failure mode.
- Common contamination without useful Oracle benefit means this selective
  write intervention is ineffective.
- Improved AssA/IDF1 and/or lower IDSW without meaningful HOTA degradation
  supports proceeding.

Tiny metric changes alone are not sufficient to conclude that SSU is useless.

## Phase 3: audit-only signal analysis

```bash
export OUTPUT_ROOT=/path/to/output/carf_v2_core

python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=development_core carf.enabled=false carf.policy=baseline

python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=development_core carf.enabled=true carf.policy=audit_only

python tools/analyze_carf_audit.py \
  "$OUTPUT_ROOT/audit_only/carf_audit.jsonl" \
  --output "$OUTPUT_ROOT/audit_only/carf_analysis.json" \
  --sequence-csv "$OUTPUT_ROOT/audit_only/per_sequence.csv"
```

The primary table is `F | N | contamination_count | contamination_rate` for
`F=0, 0.5, 1`. Risk lift for `F=1` uses only the GT-labeled, fully auditable
population as its denominator. AUROC is secondary. This is mechanism evidence,
not a complete success/failure definition for CARF.

## Phase 4: CARF-guided SSU

Only after Phase 2/3 provide useful evidence, run exactly:

```bash
# Primary intervention: q=S
python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=development_core carf.enabled=true \
  carf.policy=soft carf.soft_power=1.0

# Aggressive ablation
python "3. Tracker/run.py" \
  --config configs/carf/bee24_tracktrack_audit.yaml \
  dataset.split=development_core carf.enabled=true carf.policy=hard
```

Do not grid-search yet. Report combined TrackEval HOTA, DetA, AssA, IDF1,
IDSW, MOTA, Frag, per-sequence deltas, and the equal-weight macro mean of
per-sequence scalar deltas. If useful, repeat on `development_long`; only later
consider `stress_dense_long` and finally the frozen official test.

## Diagnostics and artifacts

CARF JSONL schema `carf.audit.v2` records `S`, `F`, auditable status,
`identity_contamination`, instance key, authority, consecutive zero-authority
writes, frames since the last effective appearance update, and actual frame age
of rollback-1/rollback-3. Rollback depth remains measured in committed
appearance writes, never frames.

Each run produces policy-specific MOT files, FPS, TrackEval combined and
per-sequence metrics, write diagnostics, and when baseline metrics are present,
per-sequence/macro deltas. No full embedding is logged. No result in this
framework asserts that CARF is effective; the real-data phases are intended to
falsify or support the hypothesis.
