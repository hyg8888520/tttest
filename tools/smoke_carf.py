#!/usr/bin/env python3
"""Data-free official-decoder CARF smoke test."""

import os
import sys
from types import SimpleNamespace

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, '3. Tracker'))

from carf.auditor import CARFAuditor
from trackers.track import Track, TrackCounter


def row(feature):
    return np.asarray([0, 0, 10, 10, .9, 1, *feature], dtype=float)


def main():
    args = SimpleNamespace(data_path='synthetic', min_len=1,
                           carf_rollback_writes=[1])
    track = Track(args, row((1, 0)))
    track.initiate(1, TrackCounter())
    detections = [Track(args, row((1, 0))), Track(args, row((0, 1)))]
    result = CARFAuditor([1]).audit_match(
        [track], (detections, [], []), (0, 0), 'tracked_lost',
        {1: np.asarray([[0.0, 1.0]])},
        dict(match_thr=.8, penalty_p=.2, penalty_q=.4,
             reduce_step=.05, frame_id=2))
    assert result.auditable and result.survival < 1.0
    print('CARF synthetic smoke: PASS (S=%.3f, F=%.3f)' %
          (result.survival, result.fragility))


if __name__ == '__main__':
    main()
