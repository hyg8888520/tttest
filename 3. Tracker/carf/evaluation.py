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


def _extract_metrics(result, tracker_name):
    combined = result['MotChallenge2DBox'][tracker_name]['COMBINED_SEQ']['pedestrian']
    return {
        'HOTA': float(np.mean(combined['HOTA']['HOTA'])),
        'DetA': float(np.mean(combined['HOTA']['DetA'])),
        'AssA': float(np.mean(combined['HOTA']['AssA'])),
        'IDF1': float(combined['Identity']['IDF1']),
        'IDSW': int(combined['CLEAR']['IDSW']),
        'MOTA': float(combined['CLEAR']['MOTA']),
        'Frag': int(combined['CLEAR']['Frag']),
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
    values = _extract_metrics(result, tracker_name)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
    return values
