
from __future__ import annotations

from collections import defaultdict
import numpy as np


def resample_sequence(sequence, target_length, feature_size):
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) == target_length:
        return sequence

    if len(sequence) == 0:
        return np.zeros((target_length, feature_size), dtype=np.float32)

    old_x = np.linspace(0, 1, len(sequence))
    new_x = np.linspace(0, 1, target_length)

    resampled = []
    for feature_idx in range(sequence.shape[1]):
        resampled.append(np.interp(new_x, old_x, sequence[:, feature_idx]))

    return np.stack(resampled, axis=1).astype(np.float32)


def normalise_prediction_output(prediction_output):
    """
    Supports both possible model output shapes:

    Shape A from WLASL2000Engine.predict_keypoint_sequence:
    {
        "top1_gloss": "...",
        "top1_confidence": ...,
        "top_k": [
            {"gloss": "...", "probability": ...},
            ...
        ]
    }

    Shape B older helper shape:
    [
        {"gloss": "...", "probability": ...},
        ...
    ]
    """

    if isinstance(prediction_output, dict):
        if "top_k" in prediction_output and isinstance(prediction_output["top_k"], list):
            return prediction_output["top_k"]

    if isinstance(prediction_output, list):
        return prediction_output

    return []


def multi_window_predict(engine, keypoint_buffer, fast_windows=(30, 45, 60), top_k=5):
    """
    Runs prediction on multiple recent window lengths and returns a consistent structure.

    Important:
    engine.predict_keypoint_sequence() returns a dictionary containing "top_k",
    so this function extracts only that list before storing it.
    """

    keypoint_list = list(keypoint_buffer)
    predictions = []

    for window_size in fast_windows:
        if len(keypoint_list) < window_size:
            continue

        window = np.array(keypoint_list[-window_size:], dtype=np.float32)

        # Convert 30/45-frame short event into the model's expected 60-frame input.
        window = resample_sequence(
            window,
            target_length=engine.sequence_length,
            feature_size=engine.base_feature_size
        )

        prediction_output = engine.predict_keypoint_sequence(window, top_k=top_k)
        top_predictions = normalise_prediction_output(prediction_output)

        if len(top_predictions) == 0:
            continue

        predictions.append({
            "window_size": int(window_size),
            "top_k": top_predictions
        })

    return predictions


def merge_event_predictions(event_predictions, top_k=5):
    """
    Merges Top-5 predictions collected during one short sign event.

    It is intentionally defensive because different parts of the prototype may pass:
    - [{"window_size": 30, "top_k": [...]}]
    - [{"top_k": [...], "top1_gloss": "..."}]
    - [[{"gloss": "..."}]]
    """

    scores = defaultdict(float)
    counts = defaultdict(int)
    top1_counts = defaultdict(int)

    for pred_pack in event_predictions:
        # Standard shape: {"window_size": 30, "top_k": [...]}
        if isinstance(pred_pack, dict) and "top_k" in pred_pack:
            top_items = normalise_prediction_output(pred_pack)

        # Older direct list shape: [{"gloss": "...", "probability": ...}, ...]
        elif isinstance(pred_pack, list):
            top_items = pred_pack

        else:
            continue

        if not isinstance(top_items, list):
            continue

        for rank, item in enumerate(top_items[:top_k]):
            if not isinstance(item, dict):
                continue

            gloss = item.get("gloss")
            if gloss is None:
                continue

            prob = float(item.get("probability", 0.0))

            # Higher-ranked items receive a small bonus.
            rank_bonus = max(0.0, (top_k - rank) / top_k) * 0.03

            scores[gloss] += prob + rank_bonus
            counts[gloss] += 1

            if rank == 0:
                top1_counts[gloss] += 1

    for gloss in list(scores.keys()):
        scores[gloss] += 0.08 * top1_counts[gloss]
        scores[gloss] += 0.02 * counts[gloss]

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    if not sorted_items:
        return []

    max_score = max(score for _, score in sorted_items)

    merged = []
    for gloss, score in sorted_items:
        merged.append({
            "gloss": gloss,
            "probability": float(score / (max_score + 1e-6)),
            "raw_score": float(score),
            "appearances": int(counts[gloss]),
            "top1_count": int(top1_counts[gloss])
        })

    return merged
