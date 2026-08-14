#!/usr/bin/env python3
"""Gate 0: exact per-frame (frame, track_id, bbox) comparison."""

import argparse
import json
import os
import sys


def read_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            columns = line.strip().split(',')
            if len(columns) < 6:
                raise ValueError('%s:%d has fewer than 6 columns' %
                                 (path, line_number))
            rows.append((int(float(columns[0])), int(float(columns[1])),
                         *(float(value) for value in columns[2:6])))
    return rows


def compare_directories(baseline_dir, candidate_dir):
    baseline_files = sorted(name for name in os.listdir(baseline_dir)
                            if name.endswith('.txt'))
    candidate_files = sorted(name for name in os.listdir(candidate_dir)
                             if name.endswith('.txt'))
    differences = []
    if baseline_files != candidate_files:
        differences.append({
            'kind': 'file_set', 'baseline': baseline_files,
            'candidate': candidate_files,
        })
    for name in sorted(set(baseline_files) & set(candidate_files)):
        baseline = read_rows(os.path.join(baseline_dir, name))
        candidate = read_rows(os.path.join(candidate_dir, name))
        if baseline != candidate:
            first = next((index for index, pair in enumerate(zip(baseline, candidate))
                          if pair[0] != pair[1]), min(len(baseline), len(candidate)))
            differences.append({
                'kind': 'rows', 'sequence': name[:-4],
                'baseline_count': len(baseline),
                'candidate_count': len(candidate),
                'first_difference_index': first,
                'baseline_row': baseline[first] if first < len(baseline) else None,
                'candidate_row': candidate[first] if first < len(candidate) else None,
            })
    return {
        'schema_version': 'carf.equivalence.v1',
        'equivalent': not differences,
        'compared_sequences': len(set(baseline_files) & set(candidate_files)),
        'differences': differences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = compare_directories(args.baseline, args.candidate)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(payload + '\n')
    return 0 if report['equivalent'] else 1


if __name__ == '__main__':
    sys.exit(main())
