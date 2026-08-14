#!/usr/bin/env python3
"""Report baseline FPS, audit FPS, and CARF per-frame overhead."""

import argparse
import json


def load(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    baseline, audit = load(args.baseline), load(args.audit)
    baseline_spf = baseline['tracker_seconds'] / baseline['frames']
    audit_spf = audit['tracker_seconds'] / audit['frames']
    report = {
        'baseline_fps': baseline['fps'],
        'audit_fps': audit['fps'],
        'carf_overhead_fraction': audit_spf / baseline_spf - 1.0,
        'carf_overhead_percent': 100.0 * (audit_spf / baseline_spf - 1.0),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(payload + '\n')


if __name__ == '__main__':
    main()
