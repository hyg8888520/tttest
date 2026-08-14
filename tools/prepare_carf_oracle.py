#!/usr/bin/env python3
"""Deprecated diagnostic: prepare modal baseline-track GT identities."""

import argparse
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, '3. Tracker'))

from carf.config import load_manifest
from carf.oracle import ORACLE_LABEL, build_oracle_artifact


def main():
    parser = argparse.ArgumentParser(description=ORACLE_LABEL)
    parser.add_argument('--gt-root', required=True)
    parser.add_argument('--tracker-results', required=True,
                        help='baseline MOT txt directory (the data/ directory)')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', default='development_core')
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    sequences = manifest[args.split]
    build_oracle_artifact(
        args.gt_root, args.tracker_results, sequences, args.output,
        threshold=args.iou_threshold)
    print(ORACLE_LABEL)
    print('wrote %s sequences to %s' % (len(sequences), args.output))


if __name__ == '__main__':
    main()
