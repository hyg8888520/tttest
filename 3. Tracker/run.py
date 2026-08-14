import os
import pickle
import argparse
import json
import time
from utils.etc import *
from trackers.tracker import Tracker
from carf.config import load_config, load_manifest, require_resolved
from carf.evaluation import evaluate_results, metrics_delta_vs_baseline
from carf.inputs import load_inputs, sanity_check_inputs
from carf.logging import AuditJSONLWriter
from carf.oracle import OnlineGTAnchorOracle


def make_parser():
    parser = argparse.ArgumentParser("Tracker")

    # Basic
    parser.add_argument("--pickle_dir", type=str, default="../outputs/2. det_feat/")
    parser.add_argument("--output_dir", type=str, default="../outputs/3. track/")
    parser.add_argument("--data_dir", type=str, default="../../dataset/")
    parser.add_argument("--dataset", type=str, default="MOT17")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--seed", type=float, default=10000)
    parser.add_argument("--config", type=str, default=None,
                        help="CARF experiment YAML")
    parser.add_argument("--sanity-only", action="store_true",
                        help="validate configured caches and exit")

    # For trackers
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--max_time_lost", type=float, default=30)
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)

    parser.add_argument("overrides", nargs="*",
                        help="YAML overrides such as carf.policy=audit_only")

    return parser


def _configure_args_from_yaml(args, config):
    tracker = config.get('tracker', {})
    for name in ('min_len', 'min_box_area', 'max_time_lost', 'penalty_p',
                 'penalty_q', 'reduce_step', 'tai_thr', 'det_thr',
                 'init_thr', 'match_thr'):
        if name not in tracker:
            raise ValueError('missing tracker.%s in CARF config' % name)
        setattr(args, name, tracker[name])
    carf = config.get('carf', {})
    args.carf_enabled = bool(carf.get('enabled', False))
    args.carf_policy = carf.get('policy', 'baseline')
    args.carf_rollback_writes = list(carf.get('rollback_writes', [1, 3]))
    args.carf_soft_power = float(carf.get('soft_power', 1.0))
    cmc = tracker.get('cmc', {})
    args.cmc_identity = bool(cmc.get('identity', False))
    args.cmc_dir = cmc.get('root', './trackers/cmc')
    args.data_path = config['dataset'].get('gt_root', config['dataset']['root'])
    return args


def _configured_sequences(config):
    manifest_path = require_resolved(
        config['dataset']['split_manifest'], 'dataset.split_manifest')
    manifest = load_manifest(manifest_path)
    split = config['dataset'].get('split', 'development_core')
    if split not in manifest:
        raise KeyError('split %s not found in %s' % (split, manifest_path))
    if (split == 'official_test' and
            not config.get('experiment', {}).get('frozen', False)):
        raise ValueError(
            'official_test runs require experiment.frozen=true; '
            'do not use them for tuning')
    return list(manifest[split])


def _configured_oracle(config, sequence, required=False):
    oracle_config = config.get('oracle', {})
    gt_root = config['dataset'].get('gt_root')
    if gt_root is None or '${' in str(gt_root):
        if required:
            require_resolved(gt_root, 'dataset.gt_root')
        return None
    return OnlineGTAnchorOracle(
        require_resolved(gt_root, 'dataset.gt_root'), sequence,
        threshold=oracle_config.get('iou_threshold', 0.5))


def _combined_write_diagnostics(per_sequence):
    count_names = (
        'accepted_write_opportunities', 'gt_labeled_writes',
        'identity_contaminating_writes', 'oracle_blocked_writes',
        'suppressed_writes')
    combined = {
        name: sum(values[name] for values in per_sequence.values())
        for name in count_names
    }
    total = combined['accepted_write_opportunities']
    labeled = combined['gt_labeled_writes']
    combined['mean_authority_q'] = (
        sum(values['mean_authority_q'] *
            values['accepted_write_opportunities']
            for values in per_sequence.values()
            if values['mean_authority_q'] is not None) / total
        if total else None)
    combined['suppressed_write_ratio'] = (
        combined['suppressed_writes'] / total if total else None)
    combined['identity_contamination_rate'] = (
        combined['identity_contaminating_writes'] / labeled
        if labeled else None)
    combined['oracle_blocked_rate'] = (
        combined['oracle_blocked_writes'] / labeled if labeled else None)
    combined['maximum_consecutive_q_zero_per_track'] = max(
        (values['maximum_consecutive_q_zero_per_track']
         for values in per_sequence.values()), default=0)
    return combined


def run_configured(config_path, overrides):
    config = load_config(config_path, overrides)
    _configure_args_from_yaml(args, config)
    sequences = _configured_sequences(config)

    require_resolved(config['dataset']['root'], 'dataset.root')
    require_resolved(config['inputs']['detections'], 'inputs.detections')
    require_resolved(config['inputs']['reid_features'], 'inputs.reid_features')
    output_root = os.path.abspath(require_resolved(
        config['output']['root'], 'output.root'))
    detections, detections_95 = load_inputs(config, sequences=sequences)

    if args.sanity_only:
        sanity = sanity_check_inputs(
            detections, detections_95,
            require_resolved(config['dataset']['root'], 'dataset.root'),
            sequences, det_thr=args.det_thr)
        os.makedirs(output_root, exist_ok=True)
        sanity_path = os.path.join(output_root, 'data_sanity.json')
        with open(sanity_path, 'w', encoding='utf-8') as handle:
            json.dump(sanity, handle, indent=2, sort_keys=True)
        print(json.dumps(sanity, indent=2, sort_keys=True), flush=True)
        return

    run_name = args.carf_policy if args.carf_enabled else 'baseline'
    run_root = os.path.join(output_root, run_name)
    result_folder = os.path.join(run_root, 'data')
    os.makedirs(result_folder, exist_ok=True)
    log_path = os.path.join(run_root, 'carf_audit.jsonl')
    logger = None
    if args.carf_enabled and args.carf_policy != 'baseline':
        logger = AuditJSONLWriter(
            log_path, config.get('carf', {}).get('log_schema', 'carf.audit.v2'))

    total_time, total_count = 0.0, 0
    write_diagnostics = {}
    try:
        for vid_name in sequences:
            oracle = None
            if args.carf_enabled and args.carf_policy != 'baseline':
                oracle = _configured_oracle(
                    config, vid_name,
                    required=args.carf_policy == 'oracle_gt')
            tracker = Tracker(args, vid_name, audit_logger=logger, oracle=oracle)
            results = []
            for frame_id in sorted(detections[vid_name]):
                start = time.time()
                frame_detections = detections[vid_name][frame_id]
                if frame_detections is not None:
                    track_results = tracker.update(
                        frame_detections, detections_95[vid_name][frame_id])
                else:
                    track_results = tracker.update_without_detections()
                total_time += time.time() - start
                total_count += 1

                x1y1whs, track_ids, scores = [], [], []
                for track_result in track_results:
                    if (track_result.track_id > 0 and
                            track_result.x1y1wh[2] * track_result.x1y1wh[3] > args.min_box_area):
                        x1y1whs.append(track_result.x1y1wh)
                        track_ids.append(track_result.track_id)
                        scores.append(track_result.score)
                results.append([frame_id, track_ids, x1y1whs, scores])
            write_results(os.path.join(result_folder, vid_name + '.txt'), results)
            if args.carf_enabled and args.carf_policy != 'baseline':
                write_diagnostics[vid_name] = (
                    tracker.write_diagnostics_summary())
    finally:
        if logger is not None:
            logger.close()

    fps = total_count / total_time if total_time else None
    performance = {
        'policy': run_name,
        'frames': total_count,
        'tracker_seconds': total_time,
        'fps': fps,
        'carf_enabled': bool(args.carf_enabled),
        'auditing_active': bool(args.carf_enabled and
                                args.carf_policy != 'baseline'),
    }
    with open(os.path.join(run_root, 'performance.json'), 'w', encoding='utf-8') as handle:
        json.dump(performance, handle, indent=2, sort_keys=True)
    print(json.dumps(performance, sort_keys=True), flush=True)

    if write_diagnostics:
        diagnostic_payload = {
            'schema_version': 'carf.write_diagnostics.v1',
            'policy': run_name,
            'combined': _combined_write_diagnostics(write_diagnostics),
            'per_sequence': write_diagnostics,
        }
        with open(os.path.join(run_root, 'write_diagnostics.json'), 'w',
                  encoding='utf-8') as handle:
            json.dump(diagnostic_payload, handle, indent=2, sort_keys=True)

    evaluation = config.get('evaluation', {})
    if evaluation.get('enabled', False):
        metrics = evaluate_results(
            require_resolved(config['dataset']['gt_root'], 'dataset.gt_root'),
            output_root, run_name, sequences,
            output_path=os.path.join(run_root, 'metrics.json'),
            benchmark=evaluation.get('benchmark', 'BEE24'),
            do_preproc=evaluation.get('do_preproc', False))
        print(json.dumps(metrics, sort_keys=True), flush=True)
        baseline_metrics_path = os.path.join(
            output_root, 'baseline', 'metrics.json')
        if run_name != 'baseline' and os.path.isfile(baseline_metrics_path):
            with open(baseline_metrics_path, 'r', encoding='utf-8') as handle:
                baseline_metrics = json.load(handle)
            delta = metrics_delta_vs_baseline(baseline_metrics, metrics)
            with open(os.path.join(run_root,
                                   'metrics_delta_vs_baseline.json'),
                      'w', encoding='utf-8') as handle:
                json.dump(delta, handle, indent=2, sort_keys=True)


def track(detections, detections_95, data_path, result_folder, mode):
    # For each video
    total_time, total_count = 0, 0
    for vid_name in detections.keys():
        # Set proper parameters
        set_parameters(args, vid_name, mode)

        # Set max time lost
        seq_info = open(data_path + vid_name + '/seqinfo.ini', mode='r')
        for s_i in seq_info.readlines():
            if 'frameRate' in s_i:
                args.max_time_lost = int(s_i.split('=')[-1]) * 2
            if 'imWidth' in s_i:
                args.img_w = int(s_i.split('=')[-1])
            if 'imHeight' in s_i:
                args.img_h = int(s_i.split('=')[-1])

        # Set tracker
        tracker = Tracker(args, vid_name)

        # For each frame
        results = []
        for frame_id in detections[vid_name].keys():
            # Run tracking
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                track_results = tracker.update(detections[vid_name][frame_id], detections_95[vid_name][frame_id])
            else:
                track_results = tracker.update_without_detections()
            total_time += time.time() - start
            total_count += 1

            # Filter out the results
            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                # Check aspect ratio
                if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                    continue

                # Check track id, minimum box area
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)

            # Merge
            results.append([frame_id, track_ids, x1y1whs, scores])

        # Logging & Write results
        result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
        write_results(result_filename, results)

    return total_time, total_count


def run():
    if args.config:
        run_configured(args.config, args.overrides)
        return

    # Legacy-only dependencies stay lazy so cache-driven CARF experiments do
    # not require AFLink/GBI or load checkpoints.
    import torch
    from AFLink.AppFreeLink import AFLink
    from AFLink.model import PostLinker
    from AFLink.dataset import LinkData
    from utils.gbi import gb_interpolation

    # Initialize AFLink
    model = PostLinker()
    model.load_state_dict(torch.load('./AFLink/AFLink_epoch20.pth'))
    aflink_dataset = LinkData('', '')

    # Logging & Set proper parameters
    print('Running %s %s...' % (args.dataset, args.mode))
    set_parameters(args, args.dataset, args.mode)

    # Make result folder
    trackers_to_eval = args.pickle_path.split('/')[-1].split('.pickle')[0]
    result_folder = os.path.join(args.output_dir, trackers_to_eval)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder + '_post/', exist_ok=True)

    # Read detection result
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)
    with open(args.pickle_path_95, 'rb') as f:
        detections_95 = pickle.load(f)

    # Track
    total_time, total_count = track(detections, detections_95, args.data_path, result_folder, args.mode)

    # Post-processing
    print('Running post-processing...')
    for result_file in os.listdir(result_folder):
        # Set Path
        path_in = result_folder + '/' + str(result_file)
        path_out = result_folder + '_post/' + str(result_file)

        # Link
        if 'Dance' in args.dataset:
            linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                            thrT=(0, 20), thrS=100, thrP=0.05)
            linker.link()

        # Gaussian Interpolation
        if 'MOT' in args.dataset:
            gb_interpolation(path_in, path_out, interval=30, tau=12)

    # Evaluation
    if args.mode == 'val':
        print('Evaluating...')
        evaluate(args, trackers_to_eval + '_post', args.dataset)

    # Logging
    print(total_count / total_time, flush=True)
    print('', flush=True)


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Run
    run()
