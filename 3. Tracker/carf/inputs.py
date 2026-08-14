"""Read-only adapters for TrackTrack and official TOPIC BEE24 caches."""

import configparser
import os
import pickle

import numpy as np


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def _join_detection_features(detections, features):
    combined = {}
    for sequence, frames in detections.items():
        combined[sequence] = {}
        for frame_id, rows in frames.items():
            if rows is None:
                combined[sequence][frame_id] = None
                continue
            rows = _to_numpy(rows)
            feature_rows = _to_numpy(features[sequence][frame_id])
            if len(rows) != len(feature_rows):
                raise ValueError('detection/ReID row mismatch at %s:%s' %
                                 (sequence, frame_id))
            if rows.shape[1] < 6:
                rows = np.concatenate(
                    (rows[:, :5], np.ones((len(rows), 1), dtype=rows.dtype)),
                    axis=1)
            combined[sequence][frame_id] = np.concatenate(
                (rows[:, :6], feature_rows), axis=1)
    return combined


def load_tracktrack_pickle(detections_path, reid_features_path=None):
    detections = _load_pickle(detections_path)
    if reid_features_path:
        return _join_detection_features(
            detections, _load_pickle(reid_features_path))
    return detections


def _sequence_dimensions(dataset_root, sequence):
    candidates = [
        os.path.join(dataset_root, 'train', sequence, 'seqinfo.ini'),
        os.path.join(dataset_root, 'test', sequence, 'seqinfo.ini'),
        os.path.join(dataset_root, sequence, 'seqinfo.ini'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            parser = configparser.ConfigParser()
            parser.read(path)
            section = parser['Sequence']
            return (int(section['imHeight']), int(section['imWidth']),
                    int(section['seqLength']))
    raise FileNotFoundError('seqinfo.ini not found for %s' % sequence)


def _topic_embedding_path(root, sequence):
    if os.path.isdir(root):
        return os.path.join(root, sequence + '_embedding.pkl')
    return root.format(sequence=sequence)


def load_topic_cache(detections_path, reid_features_path, dataset_root,
                     input_size=(800, 1440), area_range=(432.0, 10710.0),
                     score_floor=0.4, embedding_split=0.6):
    """Convert TOPIC's detector/ReID caches without modifying observations."""
    detector_cache = _load_pickle(detections_path)
    by_sequence = {}
    for tag, value in detector_cache.items():
        if value is None:
            continue
        sequence, frame_text = tag.rsplit(':', 1)
        by_sequence.setdefault(sequence, {})[int(frame_text)] = _to_numpy(value)

    converted = {}
    for sequence, frames in by_sequence.items():
        embedding_cache = _load_pickle(
            _topic_embedding_path(reid_features_path, sequence))
        image_h, image_w, sequence_length = _sequence_dimensions(
            dataset_root, sequence)
        scale = min(float(input_size[0]) / image_h,
                    float(input_size[1]) / image_w)
        converted[sequence] = {}
        for frame_id, raw in frames.items():
            raw = np.atleast_2d(raw)
            if raw.shape[1] < 5:
                raise ValueError('TOPIC detection cache must have >=5 columns')

            widths = raw[:, 2] - raw[:, 0]
            heights = raw[:, 3] - raw[:, 1]
            areas = widths * heights
            area_mask = ((areas > float(area_range[0])) &
                         (areas < float(area_range[1])))
            filtered = raw[area_mask]
            if filtered.shape[1] == 5:
                scores = filtered[:, 4]
            else:
                scores = filtered[:, 4] * filtered[:, 5]

            keep = scores > float(score_floor)
            filtered, scores = filtered[keep], scores[keep]
            high_mask = scores > float(embedding_split)
            second_mask = ~high_mask
            key = '%s:%d' % (sequence, frame_id)
            high_features = _to_numpy(embedding_cache.get(key + '@one'))
            second_features = _to_numpy(embedding_cache.get(key + '@second'))
            expected_high = int(high_mask.sum())
            expected_second = int(second_mask.sum())
            if expected_high and (high_features is None or len(high_features) != expected_high):
                raise ValueError('TOPIC @one ReID mismatch at %s' % key)
            if expected_second and (second_features is None or len(second_features) != expected_second):
                raise ValueError('TOPIC @second ReID mismatch at %s' % key)

            feature_dim = (high_features.shape[1] if expected_high else
                           second_features.shape[1] if expected_second else 0)
            features = np.empty((len(filtered), feature_dim), dtype=np.float32)
            if expected_high:
                features[high_mask] = high_features
            if expected_second:
                features[second_mask] = second_features
            boxes = filtered[:, :4].astype(np.float64, copy=True) / scale
            class_slot = np.ones((len(filtered), 1), dtype=np.float64)
            rows = np.concatenate((boxes, scores[:, None], class_slot,
                                   features), axis=1)
            converted[sequence][frame_id] = rows if len(rows) else None
        for frame_id in range(1, sequence_length + 1):
            converted[sequence].setdefault(frame_id, None)
        converted[sequence] = dict(sorted(converted[sequence].items()))
    return converted


def select_sequences(detections, sequence_names):
    sequence_names = list(sequence_names)
    missing = [name for name in sequence_names if name not in detections]
    if missing:
        raise KeyError('configured sequences absent from input cache: %s' %
                       ', '.join(missing))
    return {name: detections[name] for name in sequence_names}


def sanity_check_inputs(detections, detections_95, dataset_root,
                        sequences, det_thr=0.6):
    """Validate cache structure and report detector geometry non-destructively."""
    report = {
        'schema_version': 'carf.data_sanity.v2',
        'alignment_checked_during_load': True,
        'bbox_bounds_policy': 'report_only_preserve_detector_geometry',
        'sequences': {},
    }
    for sequence in sequences:
        if sequence not in detections or sequence not in detections_95:
            raise KeyError('missing sequence in detection stream: %s' % sequence)
        image_h, image_w, sequence_length = _sequence_dimensions(
            dataset_root, sequence)
        expected_frames = set(range(1, sequence_length + 1))
        frames = set(int(frame) for frame in detections[sequence])
        companion_frames = set(int(frame) for frame in detections_95[sequence])
        if frames != expected_frames:
            missing = sorted(expected_frames - frames)[:10]
            extra = sorted(frames - expected_frames)[:10]
            raise ValueError('%s frame-key mismatch; missing=%s extra=%s' %
                             (sequence, missing, extra))
        if companion_frames != expected_frames:
            raise ValueError('%s companion frame-key mismatch' % sequence)

        counts = {
            'frames': sequence_length,
            'nonempty_frames': 0,
            'detections': 0,
            'embedding_rows': 0,
            'high_detections': 0,
            'low_detections': 0,
            'companion_detections': 0,
            'out_of_bounds_detections': 0,
            'clipped_detections': 0,
            'dropped_degenerate_detections': 0,
        }
        feature_dims = set()
        coordinate_min = np.full(4, np.inf, dtype=np.float64)
        coordinate_max = np.full(4, -np.inf, dtype=np.float64)
        max_overflow = {'left': 0.0, 'top': 0.0,
                        'right': 0.0, 'bottom': 0.0}
        for frame_id in range(1, sequence_length + 1):
            rows = detections[sequence][frame_id]
            companion = detections_95[sequence][frame_id]
            if companion is not None:
                companion = np.atleast_2d(_to_numpy(companion))
                counts['companion_detections'] += len(companion)
            if rows is None:
                continue
            rows = np.atleast_2d(_to_numpy(rows))
            if rows.shape[1] < 7:
                raise ValueError(
                    '%s:%d needs box, score, class slot, and embedding' %
                    (sequence, frame_id))
            if not np.isfinite(rows).all():
                raise ValueError('%s:%d contains non-finite values' %
                                 (sequence, frame_id))
            boxes = rows[:, :4]
            if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(
                    boxes[:, 3] <= boxes[:, 1]):
                raise ValueError('%s:%d has non-positive boxes' %
                                 (sequence, frame_id))
            out_of_bounds = (
                (boxes[:, 0] < 0) | (boxes[:, 1] < 0) |
                (boxes[:, 2] > image_w) | (boxes[:, 3] > image_h))
            counts['out_of_bounds_detections'] += int(
                np.sum(out_of_bounds))
            coordinate_min = np.minimum(coordinate_min, np.min(boxes, axis=0))
            coordinate_max = np.maximum(coordinate_max, np.max(boxes, axis=0))
            max_overflow['left'] = max(
                max_overflow['left'], float(max(0.0, -np.min(boxes[:, 0]))))
            max_overflow['top'] = max(
                max_overflow['top'], float(max(0.0, -np.min(boxes[:, 1]))))
            max_overflow['right'] = max(
                max_overflow['right'],
                float(max(0.0, np.max(boxes[:, 2]) - image_w)))
            max_overflow['bottom'] = max(
                max_overflow['bottom'],
                float(max(0.0, np.max(boxes[:, 3]) - image_h)))
            counts['nonempty_frames'] += 1
            counts['detections'] += len(rows)
            counts['embedding_rows'] += len(rows)
            counts['high_detections'] += int(np.sum(rows[:, 4] > det_thr))
            counts['low_detections'] += int(np.sum(rows[:, 4] <= det_thr))
            feature_dims.add(int(rows.shape[1] - 6))
        if len(feature_dims) > 1:
            raise ValueError('%s has inconsistent embedding dimensions: %s' %
                             (sequence, sorted(feature_dims)))
        counts.update({
            'image_height': image_h,
            'image_width': image_w,
            'embedding_dim': (
                next(iter(feature_dims)) if feature_dims else None),
            'out_of_bounds_ratio': (
                counts['out_of_bounds_detections'] / counts['detections']
                if counts['detections'] else None),
            'bbox_coordinate_min_xyxy': (
                coordinate_min.tolist() if counts['detections'] else None),
            'bbox_coordinate_max_xyxy': (
                coordinate_max.tolist() if counts['detections'] else None),
            'max_boundary_overflow': max_overflow,
        })
        report['sequences'][sequence] = counts
    report['status'] = 'PASS'
    return report


def load_inputs(config, sequences=None):
    inputs = config['inputs']
    input_format = inputs.get('format', 'tracktrack_pickle')
    if input_format == 'tracktrack_pickle':
        detections = load_tracktrack_pickle(
            inputs['detections'], inputs.get('reid_features'))
        if inputs.get('detections_95'):
            detections_95 = load_tracktrack_pickle(
                inputs['detections_95'], inputs.get('reid_features_95'))
        else:
            detections_95 = detections
    elif input_format == 'topic_cache':
        adapter = inputs.get('topic_adapter', {})
        detections = load_topic_cache(
            inputs['detections'], inputs['reid_features'],
            config['dataset']['root'],
            input_size=adapter.get('input_size', [800, 1440]),
            area_range=adapter.get('area_range', [432, 10710]),
            score_floor=adapter.get('score_floor', 0.4),
            embedding_split=adapter.get('embedding_split', 0.6),
        )
        # TOPIC publishes one NMS stream. Reusing it as the companion stream
        # preserves every published observation and disables only TrackTrack's
        # deleted-detection recovery branch for this adapter.
        detections_95 = detections
    else:
        raise ValueError('unsupported inputs.format: %s' % input_format)

    if sequences is not None:
        detections = select_sequences(detections, sequences)
        detections_95 = select_sequences(detections_95, sequences)
    return detections, detections_95
