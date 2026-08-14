#!/usr/bin/env python3
"""Summarize CARF V2 evidence without inventing automatic science gates."""

import argparse
import json
import sys


def load(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--equivalence', required=True)
    parser.add_argument('--baseline-metrics')
    parser.add_argument('--oracle-metrics')
    parser.add_argument('--oracle-diagnostics')
    parser.add_argument('--audit-analysis')
    parser.add_argument('--output')
    args = parser.parse_args()
    equivalence = load(args.equivalence)
    report = {
        'gate_0_baseline_equivalence': {
            'pass': bool(equivalence['equivalent']),
            'decision': 'GO' if equivalence['equivalent'] else 'NO-GO: fix equivalence bug',
        }
    }
    if not equivalence['equivalent']:
        report['overall'] = 'STOP_AT_GATE_0'
    elif args.baseline_metrics and args.oracle_metrics:
        baseline, oracle = load(args.baseline_metrics), load(args.oracle_metrics)
        deltas = {
            name: oracle[name] - baseline[name]
            for name in ('HOTA', 'AssA', 'IDF1', 'IDSW')
        }
        diagnostics = (load(args.oracle_diagnostics)['combined']
                       if args.oracle_diagnostics else None)
        report['gate_1_oracle_upper_bound'] = {
            'metric_deltas': deltas,
            'write_diagnostics': diagnostics,
            'decision': (
                'INTERPRET EVENT COUNT AND METRICS JOINTLY; tiny deltas alone '
                'do not establish that SSU is useless'),
        }
        if args.audit_analysis:
            analysis = load(args.audit_analysis)
            combined = analysis['combined']
            report['gate_2_fragility_signal'] = {
                'fragility_table': combined['fragility_table'],
                'risk_lift_F_eq_1': combined[
                    'risk_lift_F_eq_1_vs_auditable_labeled'],
                'secondary_auroc': combined['secondary_auroc'],
                'decision': (
                    'MECHANISM EVIDENCE ONLY; identity_contamination is not '
                    'complete harmful-update ground truth'),
            }
            report['gate_3_real_intervention'] = {
                'candidate_policies': ['soft_q_equals_S', 'hard_ablation'],
                'report': ['delta_HOTA', 'delta_AssA', 'delta_IDF1', 'delta_IDSW'],
            }
            report['overall'] = 'READY_FOR_HUMAN_GATE_REVIEW'
        else:
            report['overall'] = 'WAITING_FOR_GATE_2_ARTIFACT'
    else:
        report['overall'] = 'WAITING_FOR_GATE_1_ARTIFACTS'
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(payload + '\n')
    return 2 if report['overall'] == 'STOP_AT_GATE_0' else 0


if __name__ == '__main__':
    sys.exit(main())
