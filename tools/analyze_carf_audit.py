#!/usr/bin/env python3
"""Analyze versioned CARF JSONL logs without fitting a predictor."""

import argparse
import csv
from collections import defaultdict
import json
import math
import os

import numpy as np


def read_jsonl(paths):
    rows = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('schema_version') != 'carf.audit.v1':
                    raise ValueError('%s:%d has unsupported schema %r' %
                                     (path, line_number,
                                      row.get('schema_version')))
                rows.append(row)
    return rows


def describe(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {'count': 0, 'mean': None, 'std': None, 'min': None,
                'q25': None, 'median': None, 'q75': None, 'max': None}
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'q25': float(np.quantile(values, .25)),
        'median': float(np.quantile(values, .5)),
        'q75': float(np.quantile(values, .75)),
        'max': float(np.max(values)),
    }


def probability(numerator_rows, denominator_rows):
    if not denominator_rows:
        return None
    return sum(bool(row['oracle_correct_write'] is False)
               for row in numerator_rows) / len(denominator_rows)


def average_ranks(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    return ranks


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def spearman(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    xr, yr = average_ranks(np.asarray(xs)), average_ranks(np.asarray(ys))
    return float(np.corrcoef(xr, yr)[0, 1])


def sequence_summary(rows):
    auditable = [row for row in rows if row.get('auditable')]
    labelled = [row for row in auditable
                if row.get('oracle_correct_write') is not None]
    wrong = [row for row in labelled if row['oracle_correct_write'] is False]
    return {
        'writes': len(rows),
        'auditable_writes': len(auditable),
        'auditable_coverage': len(auditable) / len(rows) if rows else None,
        'fragility': describe(row['F_ij'] for row in auditable),
        'oracle_labelled_writes': len(labelled),
        'oracle_wrong_writes': len(wrong),
        'wrong_write_rate': len(wrong) / len(labelled) if labelled else None,
    }


def idsw_enrichment(auditable_rows):
    global_f = [row['F_ij'] for row in auditable_rows]
    global_positive = np.mean([value > 0 for value in global_f]) if global_f else None
    global_mean = np.mean(global_f) if global_f else None
    switch_frames = defaultdict(set)
    for row in auditable_rows:
        if row.get('idsw_event'):
            switch_frames[row['sequence']].add(int(row['frame_id']))
    result = {}
    for window in (1, 3, 5):
        selected = []
        for row in auditable_rows:
            frame = int(row['frame_id'])
            if any(switch - window <= frame < switch
                   for switch in switch_frames[row['sequence']]):
                selected.append(row['F_ij'])
        selected_mean = float(np.mean(selected)) if selected else None
        selected_positive = (float(np.mean([value > 0 for value in selected]))
                             if selected else None)
        result[str(window)] = {
            'num_writes': len(selected),
            'mean_fragility': selected_mean,
            'mean_fragility_enrichment': (
                selected_mean / global_mean
                if selected_mean is not None and global_mean not in (None, 0) else None),
            'positive_fragility_rate': selected_positive,
            'positive_fragility_enrichment': (
                selected_positive / global_positive
                if selected_positive is not None and global_positive not in (None, 0) else None),
        }
    return result


def persistence_analysis(rows):
    """Relate first wrong accepted-write F to span until next known clean write."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get('auditable') and row.get('oracle_correct_write') is not None:
            grouped[(row['sequence'], int(row['track_id']))].append(row)
    events = []
    for (sequence, track_id), track_rows in grouped.items():
        track_rows.sort(key=lambda row: int(row['frame_id']))
        active = None
        for row in track_rows:
            wrong = row['oracle_correct_write'] is False
            if wrong and active is None:
                active = {'sequence': sequence, 'track_id': track_id,
                          'start': int(row['frame_id']),
                          'end': int(row['frame_id']),
                          'trigger_fragility': float(row['F_ij']),
                          'wrong_writes': 1}
            elif wrong:
                active['end'] = int(row['frame_id'])
                active['wrong_writes'] += 1
            elif active is not None:
                active['duration_frames'] = active['end'] - active['start'] + 1
                events.append(active)
                active = None
        if active is not None:
            active['duration_frames'] = active['end'] - active['start'] + 1
            events.append(active)
    return {
        'definition': 'wrong accepted-write span until next known clean accepted write',
        'num_events': len(events),
        'duration_frames': describe(event['duration_frames'] for event in events),
        'spearman_trigger_fragility_vs_duration': spearman(
            [event['trigger_fragility'] for event in events],
            [event['duration_frames'] for event in events]),
        'events': events,
    }


def rollback_contributions(rows):
    keys = sorted({key for row in rows for key in row
                   if key.startswith('rollback_') and key.endswith('_survived')},
                  key=lambda key: int(key.split('_')[1]))
    result = {}
    for key in keys:
        valid = [row[key] for row in rows if row.get(key) is not None]
        result[key] = {
            'valid': len(valid),
            'survival_rate': float(np.mean(valid)) if valid else None,
            'fragility_contribution': float(1.0 - np.mean(valid)) if valid else None,
        }
    if len(keys) >= 2:
        first, last = keys[0], keys[-1]
        paired = [row for row in rows
                  if row.get(first) is not None and row.get(last) is not None]
        result['paired'] = {
            'count': len(paired),
            'rollback_1_only_failure': sum(
                row[first] is False and row[last] is True for row in paired),
            'rollback_k_only_failure': sum(
                row[first] is True and row[last] is False for row in paired),
            'both_fail': sum(
                row[first] is False and row[last] is False for row in paired),
        }
    return result


def analyze(rows):
    auditable = [row for row in rows if row.get('auditable')]
    labelled = [row for row in auditable
                if row.get('oracle_correct_write') is not None]
    wrong = [row for row in labelled if row['oracle_correct_write'] is False]
    clean = [row for row in labelled if row['oracle_correct_write'] is True]
    f_zero = [row for row in labelled if float(row['F_ij']) == 0.0]
    f_positive = [row for row in labelled if float(row['F_ij']) > 0.0]
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row['sequence']].append(row)
    return {
        'schema_version': 'carf.analysis.v1',
        'analysis_population': 'fully auditable writes for F/wrong-write statistics',
        'writes': len(rows),
        'auditable_coverage': len(auditable) / len(rows) if rows else None,
        'fragility_overall': describe(row['F_ij'] for row in auditable),
        'fragility_clean_write': describe(row['F_ij'] for row in clean),
        'fragility_oracle_wrong_write': describe(row['F_ij'] for row in wrong),
        'P_wrong_write_given_F_eq_0': probability(f_zero, f_zero),
        'P_wrong_write_given_F_gt_0': probability(f_positive, f_positive),
        'wrong_write_auroc': auroc(
            [row['F_ij'] for row in labelled],
            [row['oracle_correct_write'] is False for row in labelled]),
        'idsw_preceding_fragility_enrichment': idsw_enrichment(auditable),
        'wrong_persistence_vs_trigger_fragility': persistence_analysis(rows),
        'rollback_independent_contribution': rollback_contributions(rows),
        'per_sequence': {
            sequence: sequence_summary(sequence_rows)
            for sequence, sequence_rows in sorted(by_sequence.items())
        },
    }


def write_sequence_csv(path, per_sequence):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'sequence', 'writes', 'auditable_writes', 'auditable_coverage',
            'mean_fragility', 'oracle_labelled_writes',
            'oracle_wrong_writes', 'wrong_write_rate'])
        writer.writeheader()
        for sequence, values in per_sequence.items():
            writer.writerow({
                'sequence': sequence,
                'writes': values['writes'],
                'auditable_writes': values['auditable_writes'],
                'auditable_coverage': values['auditable_coverage'],
                'mean_fragility': values['fragility']['mean'],
                'oracle_labelled_writes': values['oracle_labelled_writes'],
                'oracle_wrong_writes': values['oracle_wrong_writes'],
                'wrong_write_rate': values['wrong_write_rate'],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequence-csv')
    args = parser.parse_args()
    report = analyze(read_jsonl(args.logs))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    if args.sequence_csv:
        write_sequence_csv(args.sequence_csv, report['per_sequence'])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
