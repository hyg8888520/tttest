#!/usr/bin/env python3
"""Minimal CARF V2 identity-contamination signal analysis."""

import argparse
import csv
from collections import defaultdict
import json
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
                if row.get('schema_version') != 'carf.audit.v2':
                    raise ValueError('%s:%d requires carf.audit.v2, got %r' %
                                     (path, line_number,
                                      row.get('schema_version')))
                rows.append(row)
    return rows


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
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


def _auroc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def _describe(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {'count': 0, 'mean': None, 'max': None}
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'max': float(np.max(values)),
    }


def _f_table(population):
    result = {}
    for value in (0.0, 0.5, 1.0):
        selected = [row for row in population
                    if abs(float(row['F_ij']) - value) < 1e-12]
        contamination_count = sum(
            row['identity_contamination'] is True for row in selected)
        key = str(value).rstrip('0').rstrip('.') if value else '0'
        result[key] = {
            'N': len(selected),
            'contamination_count': contamination_count,
            'contamination_rate': _rate(contamination_count, len(selected)),
        }
    return result


def population_summary(rows):
    labeled = [row for row in rows
               if row.get('identity_contamination') is not None]
    auditable = [row for row in rows if row.get('auditable')]
    auditable_labeled = [row for row in auditable
                         if row.get('identity_contamination') is not None]
    contaminated_labeled = sum(
        row['identity_contamination'] is True for row in labeled)
    contaminated_auditable = sum(
        row['identity_contamination'] is True for row in auditable_labeled)
    table = _f_table(auditable_labeled)
    baseline_rate = _rate(contaminated_auditable, len(auditable_labeled))
    f1_rate = table['1']['contamination_rate']
    risk_lift = (f1_rate / baseline_rate
                 if f1_rate is not None and baseline_rate not in (None, 0)
                 else None)
    return {
        'total_accepted_edges': len(rows),
        'gt_labeled_accepted_edges': len(labeled),
        'auditable_accepted_edges': len(auditable),
        'gt_labeled_and_auditable_edges': len(auditable_labeled),
        'gt_label_coverage': _rate(len(labeled), len(rows)),
        'auditable_coverage': _rate(len(auditable), len(rows)),
        'overall_contamination_rate_gt_labeled': _rate(
            contaminated_labeled, len(labeled)),
        'contamination_rate_gt_labeled_and_auditable': baseline_rate,
        'fragility_table': table,
        'risk_lift_F_eq_1_vs_auditable_labeled': risk_lift,
        'secondary_auroc': _auroc(
            [row['F_ij'] for row in auditable_labeled],
            [row['identity_contamination'] is True
             for row in auditable_labeled]),
        'mean_authority_q': (
            float(np.mean([row['authority_q'] for row in rows]))
            if rows else None),
        'suppressed_write_ratio': _rate(
            sum(float(row['authority_q']) < 1.0 for row in rows), len(rows)),
        'maximum_consecutive_q_zero': max(
            (int(row.get('max_consecutive_q_zero', 0)) for row in rows),
            default=0),
        'frames_since_last_effective_update': _describe(
            row.get('frames_since_last_effective_appearance_update')
            for row in rows),
        'rollback_1_frame_age': _describe(
            row.get('rollback_1_frame_age') for row in rows),
        'rollback_3_frame_age': _describe(
            row.get('rollback_3_frame_age') for row in rows),
    }


def analyze(rows):
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row['sequence']].append(row)
    return {
        'schema_version': 'carf.analysis.v2',
        'label_definition': (
            'identity_contamination means the accepted observation GT identity '
            'differs from the immutable online track-instance anchor'),
        'fragility_semantics': (
            'counterfactual association sensitivity, not a calibrated '
            'probability or complete harmful-write ground truth'),
        'combined': population_summary(rows),
        'per_sequence': {
            sequence: population_summary(sequence_rows)
            for sequence, sequence_rows in sorted(by_sequence.items())
        },
    }


def write_sequence_csv(path, per_sequence):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        'sequence', 'total_accepted_edges', 'gt_labeled_accepted_edges',
        'auditable_accepted_edges', 'gt_labeled_and_auditable_edges',
        'gt_label_coverage', 'auditable_coverage',
        'overall_contamination_rate_gt_labeled',
        'contamination_rate_gt_labeled_and_auditable',
        'risk_lift_F_eq_1_vs_auditable_labeled']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sequence, values in per_sequence.items():
            writer.writerow({'sequence': sequence, **{
                name: values[name] for name in fieldnames[1:]}})


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
