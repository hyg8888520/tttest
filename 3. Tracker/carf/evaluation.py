"""Thin configurable wrapper around the vendored official TrackEval."""

import configparser
import json
import os

import numpy as np
import trackeval


METRIC_NAMES = ('HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW', 'MOTA', 'Frag')


def _sequence_lengths(gt_root, sequences):
    result = {}
    for sequence in sequences:
        candidates = [
            os.path.join(gt_root, sequence, 'seqinfo.ini'),
            os.path.join(gt_root, 'BEE24-val', sequence, 'seqinfo.ini'),
        ]
        ini_path = next((path for path in candidates if os.path.isfile(path)), None)
        if ini_path is None:
            raise FileNotFoundError('seqinfo.ini not found for %s' % sequence)
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        result[sequence] = int(parser['Sequence']['seqLength'])
    return result


def _metric_values(sequence_result):
    return {
        'HOTA': float(np.mean(sequence_result['HOTA']['HOTA'])),
        'DetA': float(np.mean(sequence_result['HOTA']['DetA'])),
        'AssA': float(np.mean(sequence_result['HOTA']['AssA'])),
        'IDF1': float(sequence_result['Identity']['IDF1']),
        'IDSW': int(sequence_result['CLEAR']['IDSW']),
        'MOTA': float(sequence_result['CLEAR']['MOTA']),
        'Frag': int(sequence_result['CLEAR']['Frag']),
    }


def _extract_metrics(result, tracker_name, sequences):
    tracker_result = result['MotChallenge2DBox'][tracker_name]
    combined = _metric_values(tracker_result['COMBINED_SEQ']['pedestrian'])
    combined['per_sequence'] = {
        sequence: _metric_values(tracker_result[sequence]['pedestrian'])
        for sequence in sequences
    }
    return combined


def metrics_delta_vs_baseline(baseline, candidate):
    names = ('HOTA', 'AssA', 'IDF1', 'IDSW')
    per_sequence = {}
    baseline_sequences = set(baseline.get('per_sequence', {}))
    candidate_sequences = set(candidate.get('per_sequence', {}))
    if baseline_sequences != candidate_sequences:
        raise ValueError(
            'baseline/candidate metric sequence sets differ: %s vs %s' %
            (sorted(baseline_sequences), sorted(candidate_sequences)))
    common = sorted(baseline_sequences)
    for sequence in common:
        per_sequence[sequence] = {
            'delta_' + name: (
                candidate['per_sequence'][sequence][name] -
                baseline['per_sequence'][sequence][name])
            for name in names
        }
    macro = {
        'delta_' + name: (
            float(np.mean([values['delta_' + name]
                           for values in per_sequence.values()]))
            if per_sequence else None)
        for name in names
    }
    combined = {
        'delta_' + name: candidate[name] - baseline[name]
        for name in names
    }
    return {
        'schema_version': 'carf.metrics_delta.v1',
        'combined_delta': combined,
        'per_sequence': per_sequence,
        'macro_mean_per_sequence_delta': macro,
    }


def evaluate_results(gt_root, trackers_root, tracker_name, sequences,
                     output_path=None, benchmark='BEE24',
                     tracker_sub_folder='data', do_preproc=False):
    """Evaluate HOTA/DetA/AssA/IDF1/IDSW/MOTA/Frag with TrackEval."""
    eval_config = {
        'USE_PARALLEL': False,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_EMPTY_CLASSES': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
    }
    dataset_config = {
        'GT_FOLDER': gt_root,
        'TRACKERS_FOLDER': trackers_root,
        'OUTPUT_FOLDER': None,
        'TRACKERS_TO_EVAL': [tracker_name],
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': benchmark,
        'SPLIT_TO_EVAL': 'val',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': bool(do_preproc),
        'TRACKER_SUB_FOLDER': tracker_sub_folder,
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': None,
        'SEQMAP_FILE': None,
        'SEQ_INFO': _sequence_lengths(gt_root, sequences),
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'SKIP_SPLIT_FOL': True,
    }
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
               trackeval.metrics.Identity()]
    result, _ = evaluator.evaluate(dataset_list, metrics)
    values = _extract_metrics(result, tracker_name, sequences)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
    return values
