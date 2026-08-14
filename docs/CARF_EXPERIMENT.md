# CARF + SSU V1 experiment

## Hypothesis and limits

The falsifiable hypothesis is that a fragile accepted association can pollute a
track's mutable appearance state and amplify later identity errors. CARF V1
measures **counterfactual fragility**, not correctness and not an error
probability. It does not change the current frame's assignment.

V1 contains no learned component, alternate decoder, assignment entropy,
no-appearance counterfactual, anchor, quarantine, re-anchor, TTL, router,
Transformer, graph model, conformal layer, MoE, MHT, or new loss.

## Algorithm

For each official accepted edge `(track_i, detection_j)` and each configured
write rollback `r`:

1. Deep-clone all track and detection objects required by that association
   stage.
2. Replace only cloned `track_i.feat` with its rollback-by-write snapshot.
3. Re-run TrackTrack's official `iterative_assignment` with unchanged inputs,
   motion/Kalman state, thresholds, penalties, gating, and decoder.
4. Record whether `(i,j)` survives.

For valid rollbacks, `S_ij` is the survived fraction and `F_ij = 1-S_ij`. A row
is `auditable=true` only when every requested rollback exists. Partial rollback
outcomes may be logged for coverage analysis, but insufficient full history
forces `q=1`.

All counterfactuals for a stage run before any accepted write from that stage,
so they observe exactly the state used for the official assignment. Runtime
state fingerprints are checked around every audit; a mutation raises an error.

## SSU policies

The original innovation is:

```text
beta_base = alpha + (1-alpha)*(1-score)
innovation = 1-beta_base
```

SSU applies only to `feat`:

```text
effective = q * innovation
feat = normalize((1-effective)*old_feat + effective*new_feat)
```

- `baseline`: no CARF rerun; `q=1` through the original code path.
- `audit_only`: compute/log CARF; `q=1` through the original code path.
- `oracle_gt`: block only known GT-inconsistent writes; unknown means `q=1`.
- `hard`: `q=1` iff fully auditable `S=1`, else `q=0`.
- `soft`: `q=S**soft_power` when fully auditable.

`q=0` leaves the feature byte-for-byte unchanged. `q=1` executes the original
baseline arithmetic verbatim. Kalman mean/covariance, box, velocity, score,
lifecycle, end frame, assignment, and current output ID never depend on `q`.

## Offline GT oracle

`oracle_gt` is always labeled:

```text
OFFLINE ORACLE - NOT A DEPLOYABLE METHOD
```

First run the baseline over the complete analysis sequence. Oracle preparation
uses TrackEval CLEAR-compatible frame matching and assigns each predicted track
the modal GT identity across its valid matches. During the oracle run, the
current matched detection is matched to frame GT with the same IoU primitive,
Hungarian primitive, and `0.5` threshold. A known mismatch gets `q=0`; a match
gets `q=1`; unknown observation or canonical identity gets baseline `q=1`.
The current accepted assignment is never replaced.

## Fixed BEE24 split

`configs/carf/manifests/bee24_carf_v1.yaml` partitions complete sequences, never
frames. It records all 25 project-train, 6 development, and 5 official-test
sequence names. Development was selected deterministically by SHA256 from the
31 official train sequences. The adapter fails if a named sequence is absent.

Do not tune rollback writes, `soft_power`, any threshold, or policy on
`official_test`. Freeze choices using `project_train` and `development` first.
The runner refuses enabled official-test evaluation unless
`experiment.frozen=true` is explicit.

## Gates

1. **Gate 0 — baseline equivalence.** If baseline and audit-only differ at any
   `(frame, track_id, bbox)`, fix the bug and stop all experiments.
2. **Gate 1 — oracle upper bound.** Continue only if oracle write protection
   improves AssA or IDF1, reduces IDSW, and does not degrade HOTA. Otherwise:
   `appearance-state pollution is not a useful intervention target`.
3. **Gate 2 — fragility signal.** Continue only if wrong writes show visible
   fragility enrichment/separation (the gate script checks wrong median > clean
   median, AUROC > 0.5, and `P(wrong|F>0) > P(wrong|F=0)`). Do not train a
   predictor when this fails.
4. **Gate 3 — real intervention.** Only after Gates 1 and 2 compare hard and
   soft and report delta HOTA, AssA, IDF1, and IDSW against baseline.

Speed is reported but is not a V1 gate.

## Environment and YAML

Install the small runtime additions, then set paths. Nothing is downloaded:

```bash
python -m pip install -r requirements-carf.txt
export BEE24_ROOT=/path/to/BEE24
export BEE24_DETECTIONS=/path/to/TOPICTrack/cache/det_bee24.pkl
export BEE24_REID_FEATURES=/path/to/TOPICTrack/cache/embeddings
export BEE24_DET_CKPT=/path/to/topictrack_bee_detector.pth.tar
export BEE24_REID_CKPT=/path/to/bee24_AGW.pth
export BEE24_CMC_ROOT=/optional/path/to/cmc
export BEE24_ORACLE_ARTIFACT=/path/to/output/baseline/oracle_artifact.json
export OUTPUT_ROOT=/path/to/output/carf_v1
```

The two checkpoint paths are provenance fields when cached detection/features
are used; they are not loaded by the adapter. `BEE24_CMC_ROOT` is likewise not
read while `tracker.cmc.identity=true`.

## Runs

Run from the repository root. Baseline and audit-only must use the same frozen
YAML and cache files.

```bash
# pristine framework baseline (no CARF reruns)
python "3. Tracker/run.py" --config configs/carf/bee24_tracktrack_audit.yaml carf.enabled=false carf.policy=baseline

# prepare baseline modal identities from development GT
python tools/prepare_carf_oracle.py --gt-root "$BEE24_ROOT/train" --tracker-results "$OUTPUT_ROOT/baseline/data" --manifest configs/carf/manifests/bee24_carf_v1.yaml --split development --output "$BEE24_ORACLE_ARTIFACT"

# audit only (CARF logs plus oracle labels, exact baseline appearance writes)
python "3. Tracker/run.py" --config configs/carf/bee24_tracktrack_audit.yaml carf.enabled=true carf.policy=audit_only

# exact output Gate 0
python tools/compare_tracking_outputs.py --baseline "$OUTPUT_ROOT/baseline/data" --candidate "$OUTPUT_ROOT/audit_only/data" --output "$OUTPUT_ROOT/equivalence.json"

# GT oracle
python "3. Tracker/run.py" --config configs/carf/bee24_tracktrack_audit.yaml carf.enabled=true carf.policy=oracle_gt

# hard SSU (only after Gates 1 and 2)
python "3. Tracker/run.py" --config configs/carf/bee24_tracktrack_audit.yaml carf.enabled=true carf.policy=hard

# soft SSU (only after Gates 1 and 2)
python "3. Tracker/run.py" --config configs/carf/bee24_tracktrack_audit.yaml carf.enabled=true carf.policy=soft
```

Analyze audit rows and performance:

```bash
python tools/analyze_carf_audit.py "$OUTPUT_ROOT/audit_only/carf_audit.jsonl" --output "$OUTPUT_ROOT/audit_only/carf_analysis.json" --sequence-csv "$OUTPUT_ROOT/audit_only/per_sequence.csv"
python tools/compare_carf_performance.py --baseline "$OUTPUT_ROOT/baseline/performance.json" --audit "$OUTPUT_ROOT/audit_only/performance.json" --output "$OUTPUT_ROOT/performance_comparison.json"
python tools/check_carf_gates.py --equivalence "$OUTPUT_ROOT/equivalence.json" --baseline-metrics "$OUTPUT_ROOT/baseline/metrics.json" --oracle-metrics "$OUTPUT_ROOT/oracle_gt/metrics.json" --audit-analysis "$OUTPUT_ROOT/audit_only/carf_analysis.json" --output "$OUTPUT_ROOT/gates.json"
```

## Expected artifacts

Each policy directory contains `data/<sequence>.txt`, `performance.json`, and,
when GT evaluation is enabled, `metrics.json`. CARF-enabled non-baseline modes
also contain `carf_audit.jsonl` with schema `carf.audit.v1`. Analysis produces
`carf_analysis.json` and a per-sequence CSV. Full embeddings are never logged.

The code establishes a falsifiable experiment. It does not assert that CARF is
effective; only the remote data runs can determine the gates.

After all choices are frozen, change `dataset.gt_root` to the official-test GT
root used by the local TOPIC TrackEval layout and run with
`dataset.split=official_test experiment.frozen=true`. Repeat the five policy
commands without changing tracker/CARF parameters. If official-test GT is not
available on the execution server, set `evaluation.enabled=false` and evaluate
the emitted MOT files through the official service instead.
