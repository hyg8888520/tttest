#!/usr/bin/env python3
"""Evaluate the documented CARF V1 stop/go gates from frozen artifacts."""

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
        identity_improved = (oracle['AssA'] > baseline['AssA'] or
                             oracle['IDF1'] > baseline['IDF1'])
        switches_reduced = oracle['IDSW'] < baseline['IDSW']
        tracking_not_degraded = oracle['HOTA'] >= baseline['HOTA']
        passed = identity_improved and switches_reduced and tracking_not_degraded
        report['gate_1_oracle_upper_bound'] = {
            'pass': passed,
            'identity_improved': identity_improved,
            'idsw_reduced': switches_reduced,
            'hota_not_degraded': tracking_not_degraded,
            'decision': ('GO' if passed else
                         'NO-GO: appearance-state pollution is not a useful intervention target'),
        }
        if not passed:
            report['overall'] = 'STOP_AT_GATE_1'
        elif args.audit_analysis:
            analysis = load(args.audit_analysis)
            wrong = analysis['fragility_oracle_wrong_write']['median']
            clean = analysis['fragility_clean_write']['median']
            auc = analysis['wrong_write_auroc']
            p0 = analysis['P_wrong_write_given_F_eq_0']
            p1 = analysis['P_wrong_write_given_F_gt_0']
            separated = (wrong is not None and clean is not None and
                         auc is not None and p0 is not None and p1 is not None and
                         wrong > clean and auc > 0.5 and p1 > p0)
            report['gate_2_fragility_signal'] = {
                'pass': separated,
                'decision': ('GO: hard/soft intervention unlocked' if separated else
                             'NO-GO: no fragility enrichment/separation; do not train a predictor'),
            }
            report['gate_3_real_intervention'] = {
                'unlocked': separated,
                'report': ['delta_HOTA', 'delta_AssA', 'delta_IDF1', 'delta_IDSW'],
            }
            report['overall'] = 'GATE_3_UNLOCKED' if separated else 'STOP_AT_GATE_2'
        else:
            report['overall'] = 'WAITING_FOR_GATE_2_ARTIFACT'
    else:
        report['overall'] = 'WAITING_FOR_GATE_1_ARTIFACTS'
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(payload + '\n')
    return 0 if report['overall'] in {'GATE_3_UNLOCKED', 'WAITING_FOR_GATE_1_ARTIFACTS',
                                      'WAITING_FOR_GATE_2_ARTIFACT'} else 2


if __name__ == '__main__':
    sys.exit(main())
