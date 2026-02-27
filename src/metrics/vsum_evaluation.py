"""
Video Summarization Evaluation Toolkit
======================================

This module provides:
- Segment-level knapsack optimization (OR-Tools)
- Frame-level summary construction
- F1 / Precision / Recall evaluation (paper-aligned, custom)
- Spearman rho / Kendall tau-b rank correlation (scipy)

Design goals:
- Paper-consistent evaluation (SumMe / TVSum)
- Clear semantics over convenience
- Research-friendly and reproducible
"""

from scipy.stats import spearmanr, kendalltau
import math
import os
import json
import importlib
import re
from datetime import datetime
import numpy as np
import h5py

_ortools_knapsack = None
for _module_name in (
    "ortools.algorithms.pywrapknapsack_solver",
    "ortools.algorithms.python.knapsack_solver",
):
    try:
        _ortools_knapsack = importlib.import_module(_module_name)
        break
    except Exception:
        continue


# ============================================================
# Knapsack Optimization
# ============================================================

def solve_knapsack_segments(
    segment_scores,
    segment_lengths,
    capacity
):
    """
    Solve 0/1 knapsack problem for segment selection.
    """
    scores = np.asarray(segment_scores, dtype=np.float32)
    lengths = np.asarray(segment_lengths, dtype=np.int32)

    values = (scores * 1000).astype(int).tolist()
    weights = lengths.tolist()
    cap = int(capacity)

    if _ortools_knapsack is not None:
        solver_cls = getattr(_ortools_knapsack, "KnapsackSolver", None)
        if solver_cls is not None:
            try:
                if hasattr(solver_cls, "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER"):
                    solver = solver_cls(
                        solver_cls.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
                        "segment_knapsack"
                    )
                else:
                    try:
                        solver = solver_cls()
                    except TypeError:
                        solver = solver_cls("segment_knapsack")

                init_fn = getattr(solver, "Init", None) or getattr(
                    solver, "init", None)
                solve_fn = getattr(solver, "Solve", None) or getattr(
                    solver, "solve", None)
                contains_fn = (
                    getattr(solver, "BestSolutionContains", None)
                    or getattr(solver, "best_solution_contains", None)
                )

                if init_fn and solve_fn and contains_fn:
                    init_fn(values, [weights], [cap])
                    solve_fn()
                    return [
                        idx for idx in range(len(weights))
                        if contains_fn(idx)
                    ]
            except Exception:
                pass

    if cap <= 0 or len(weights) == 0:
        return []

    n = len(weights)
    dp = np.zeros((n + 1, cap + 1), dtype=np.float32)
    keep = np.zeros((n + 1, cap + 1), dtype=np.int8)

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = float(values[i - 1])
        for c in range(cap + 1):
            if w <= c and dp[i - 1, c - w] + v > dp[i - 1, c]:
                dp[i, c] = dp[i - 1, c - w] + v
                keep[i, c] = 1
            else:
                dp[i, c] = dp[i - 1, c]

    selected = []
    c = cap
    for i in range(n, 0, -1):
        if keep[i, c] == 1:
            selected.append(i - 1)
            c -= weights[i - 1]

    return selected[::-1]


# ============================================================
# Summary Construction
# ============================================================

def build_frame_summary_from_segments(
    predicted_scores,
    change_points,
    total_frames,
    frames_per_segment,
    sampled_positions,
    summary_ratio=0.15,
    selection_method="knapsack"
):
    """
    Build a frame-level binary summary from segment-level predictions.
    """
    predicted_scores = np.asarray(predicted_scores, dtype=np.float32)
    change_points = np.asarray(change_points, dtype=np.int32)
    sampled_positions = np.asarray(sampled_positions, dtype=np.int32)

    if sampled_positions[-1] != total_frames:
        sampled_positions = np.append(sampled_positions, total_frames)

    assert len(sampled_positions) == len(predicted_scores) + 1

    # Frame-level scores
    frame_scores = np.zeros(total_frames, dtype=np.float32)
    for i in range(len(predicted_scores)):
        frame_scores[
            sampled_positions[i]:sampled_positions[i + 1]
        ] = predicted_scores[i]

    # Segment-level mean scores
    segment_scores = [
        frame_scores[start:end + 1].mean()
        for start, end in change_points
    ]

    max_summary_frames = int(
        math.floor(total_frames * summary_ratio)
    )

    if selection_method == "knapsack":
        selected_segments = solve_knapsack_segments(
            segment_scores,
            frames_per_segment,
            max_summary_frames
        )
    elif selection_method == "rank":
        order = np.argsort(segment_scores)[::-1]
        selected_segments, used = [], 0
        for idx in order:
            if used + frames_per_segment[idx] <= max_summary_frames:
                selected_segments.append(idx)
                used += frames_per_segment[idx]
    else:
        raise ValueError(selection_method)

    summary = np.concatenate([
        np.ones(frames_per_segment[i], dtype=np.float32)
        if i in selected_segments
        else np.zeros(frames_per_segment[i], dtype=np.float32)
        for i in range(len(frames_per_segment))
    ])

    return summary, segment_scores, selected_segments


# ============================================================
# F1 / Precision / Recall (Paper-aligned, Custom)
# ============================================================

def _compute_f1_precision_recall(machine, human):
    """
    Paper-aligned F1 computation for one human summary.
    """
    overlap = np.sum(machine * human)
    precision = overlap / (np.sum(machine) + 1e-8)
    recall = overlap / (np.sum(human) + 1e-8)

    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def evaluate_f1_frame_summary(
    machine_summary,
    human_summaries,
    reduction="avg"
):
    """
    Evaluate machine vs human summaries using paper-aligned F1.
    """
    machine = np.asarray(machine_summary, dtype=np.float32)
    human_summaries = np.asarray(human_summaries, dtype=np.float32)

    machine[machine > 0] = 1
    human_summaries[human_summaries > 0] = 1

    n_users, n_frames = human_summaries.shape

    if machine.size < n_frames:
        machine = np.pad(machine, (0, n_frames - machine.size))
    else:
        machine = machine[:n_frames]

    f1_list, p_list, r_list = [], [], []

    for u in range(n_users):
        f1, p, r = _compute_f1_precision_recall(
            machine,
            human_summaries[u]
        )
        f1_list.append(f1)
        p_list.append(p)
        r_list.append(r)

    if reduction == "avg":
        return (
            float(np.mean(f1_list)),
            float(np.mean(p_list)),
            float(np.mean(r_list))
        )
    elif reduction == "max":
        idx = int(np.argmax(f1_list))
        return f1_list[idx], p_list[idx], r_list[idx]
    else:
        raise ValueError(reduction)


def evaluate_f1_human_consistency(
    human_summaries,
    reduction="avg"
):
    """
    Human–human consistency evaluation (paper-aligned).
    """
    human_summaries = np.asarray(human_summaries, dtype=np.float32)
    human_summaries[human_summaries > 0] = 1

    f1_list, p_list, r_list = [], [], []

    n_users = human_summaries.shape[0]
    for i in range(n_users):
        for j in range(i + 1, n_users):
            f1, p, r = _compute_f1_precision_recall(
                human_summaries[i],
                human_summaries[j]
            )
            f1_list.append(f1)
            p_list.append(p)
            r_list.append(r)

    if reduction == "avg":
        return (
            float(np.mean(f1_list)),
            float(np.mean(p_list)),
            float(np.mean(r_list))
        )
    elif reduction == "max":
        idx = int(np.argmax(f1_list))
        return f1_list[idx], p_list[idx], r_list[idx]
    else:
        raise ValueError(reduction)


# ============================================================
# Rank Correlation Evaluation
# ============================================================

def evaluate_rank_correlation(
    predicted_scores,
    human_scores,
    reduction="avg"
):
    """
    Compute Spearman rho and Kendall tau-b correlations.
    """
    predicted_scores = np.asarray(predicted_scores, dtype=np.float32).ravel()

    if predicted_scores.size == 0:
        return 0.0, 0.0

    if isinstance(human_scores, list):
        users = human_scores
    else:
        human_scores = np.asarray(human_scores)
        users = (
            [human_scores]
            if human_scores.ndim == 1
            else [human_scores[i] for i in range(human_scores.shape[0])]
        )

    rho_vals, tau_vals = [], []

    for u in users:
        u = np.asarray(u, dtype=np.float32).ravel()
        n = min(len(predicted_scores), len(u))

        rho, _ = spearmanr(predicted_scores[:n], u[:n])
        tau, _ = kendalltau(predicted_scores[:n], u[:n], variant="b")

        rho_vals.append(0.0 if np.isnan(rho) else rho)
        tau_vals.append(0.0 if np.isnan(tau) else tau)

    if reduction == "avg":
        return float(np.mean(rho_vals)), float(np.mean(tau_vals))
    elif reduction == "max":
        idx = int(np.argmax(rho_vals))
        return rho_vals[idx], tau_vals[idx]
    else:
        raise ValueError(reduction)


def evaluate_rank_correlation_batch(
    predicted_scores_list,
    human_scores_list,
    reduction="avg"
):
    """
    Compute average rank correlations across multiple videos.
    """
    rhos, taus = [], []

    for pred, human in zip(predicted_scores_list, human_scores_list):
        rho, tau = evaluate_rank_correlation(pred, human, reduction)
        rhos.append(rho)
        taus.append(tau)

    return float(np.mean(rhos)), float(np.mean(taus))


def norm_func(values, norm="none"):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    mode = (norm or "none").lower()
    if mode in ("none", "raw"):
        return arr
    if mode == "minmax":
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)
    if mode == "zscore":
        mean, std = float(np.mean(arr)), float(np.std(arr))
        if std < 1e-12:
            return np.zeros_like(arr)
        return (arr - mean) / std
    raise ValueError(f"Unsupported norm mode: {norm}")


def temporal_smoothing_func(video_data, norm="none"):
    frames_start = list(video_data.get("scene_frames", []))
    scenes_scores = list(video_data.get("scene_scores", []))
    seq_len = int(video_data.get("sequence_length", 0))

    if seq_len <= 0:
        return []
    if len(frames_start) == 0 or len(scenes_scores) == 0:
        return [0.0] * seq_len

    scenes_scores = norm_func(scenes_scores, norm)

    frames_start = sorted(int(v) for v in frames_start)
    if frames_start[0] != 0:
        frames_start = [0] + frames_start
    if frames_start[-1] != seq_len:
        frames_start = frames_start + [seq_len]

    if len(frames_start) != len(scenes_scores) + 1:
        usable = min(len(scenes_scores), len(frames_start) - 1)
        scenes_scores = scenes_scores[:usable]
        frames_start = frames_start[:usable + 1]

    if len(scenes_scores) == 0:
        return [0.0] * seq_len

    expanded_scores = []
    for i in range(1, len(frames_start)):
        seg_len = max(0, frames_start[i] - frames_start[i - 1])
        expanded_scores.extend([float(scenes_scores[i - 1])] * seg_len)

    expanded_scores = np.asarray(expanded_scores, dtype=np.float32)
    if expanded_scores.size < seq_len:
        pad_val = float(expanded_scores[-1]
                        ) if expanded_scores.size > 0 else 0.0
        expanded_scores = np.pad(
            expanded_scores, (0, seq_len - expanded_scores.size), constant_values=pad_val)
    elif expanded_scores.size > seq_len:
        expanded_scores = expanded_scores[:seq_len]

    num_scenes = len(scenes_scores)
    for i in range(1, num_scenes):
        start1 = frames_start[i - 1]
        start2 = frames_start[i]
        start3 = frames_start[i + 1] if i + 1 < len(frames_start) else seq_len

        scene1_half1 = max(0, (start2 - start1) // 2)
        scene1_half2 = max(0, (start3 - start2) // 2)

        transition_length = scene1_half1 + scene1_half2 + 1
        if transition_length <= 1:
            continue

        transition_x = np.linspace(0, np.pi, transition_length)

        x1, x2 = float(scenes_scores[i - 1]), float(scenes_scores[i])
        transition = x1 + (x2 - x1) * (1 - np.cos(transition_x)) / 2

        apply_start = start1 + scene1_half1
        apply_end = min(seq_len, apply_start + transition_length)
        if apply_start < apply_end:
            expanded_scores[apply_start:apply_end] = transition[:apply_end - apply_start]

    return expanded_scores.tolist()


def _build_temporal_smoothing_video_data(picks, frame_scene_data, pick_score_map):
    picks = np.asarray(picks, dtype=np.int32)
    if picks.size == 0:
        return {"scene_frames": [0], "scene_scores": [0.0], "sequence_length": 0}

    pick_to_idx = {int(p): idx for idx, p in enumerate(picks.tolist())}
    scene_starts = []
    scene_scores = []

    for scene_item in frame_scene_data:
        frames = scene_item.get("frames", []) if isinstance(
            scene_item, dict) else []
        scene_indices = []
        scene_values = []
        for frame_item in frames:
            if not isinstance(frame_item, dict) or frame_item.get("pick") is None:
                continue
            pick = int(frame_item["pick"])
            if pick in pick_to_idx:
                scene_indices.append(pick_to_idx[pick])
                scene_values.append(_safe_float(
                    pick_score_map.get(pick, 0.0), 0.0))

        if scene_indices:
            scene_starts.append(int(min(scene_indices)))
            scene_scores.append(float(np.mean(scene_values)))

    if not scene_starts:
        return {
            "scene_frames": [0],
            "scene_scores": [float(np.mean(list(pick_score_map.values()))) if pick_score_map else 0.0],
            "sequence_length": int(len(picks))
        }

    ordered = sorted(zip(scene_starts, scene_scores), key=lambda x: x[0])
    dedup_starts = []
    dedup_scores = []
    for st, sc in ordered:
        if not dedup_starts or st != dedup_starts[-1]:
            dedup_starts.append(st)
            dedup_scores.append(sc)
        else:
            dedup_scores[-1] = (dedup_scores[-1] + sc) / 2.0

    if dedup_starts[0] != 0:
        dedup_starts = [0] + dedup_starts
        dedup_scores = [dedup_scores[0]] + dedup_scores

    return {
        "scene_frames": dedup_starts,
        "scene_scores": dedup_scores,
        "sequence_length": int(len(picks))
    }


# ============================================================
# Exam Evaluation Pipeline (from data/scores)
# ============================================================

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_h5_dataset_as_dict(h5_path):
    dataset = {}
    with h5py.File(h5_path, "r") as h5_file:
        for video_key in h5_file.keys():
            dataset[video_key] = {}
            for sub_key in h5_file[video_key].keys():
                dataset[video_key][sub_key] = h5_file[video_key][sub_key][...]
    return dataset


def _build_video_key_mapping(dataset_name, dataset_dict, video_name_dict_path):
    dataset_keys = sorted(
        list(dataset_dict.keys()),
        key=lambda x: int(x.split("_")[1])
    )

    if dataset_name == "tvsum":
        raw_map = _load_json(video_name_dict_path)
        video_id_to_name = {video_id: name for name,
                            video_id in raw_map.items()}
        return {
            dataset_key: video_id_to_name.get(dataset_key, dataset_key)
            for dataset_key in dataset_keys
        }

    mapping = {}
    for dataset_key in dataset_keys:
        video_name = dataset_dict[dataset_key].get("video_name", None)
        if isinstance(video_name, bytes):
            video_name = video_name.decode("utf-8")
        elif isinstance(video_name, np.ndarray):
            try:
                if video_name.shape == ():
                    video_name = video_name.item()
                if isinstance(video_name, bytes):
                    video_name = video_name.decode("utf-8")
            except Exception:
                pass

        if not isinstance(video_name, str) or not video_name:
            video_name = dataset_key

        mapping[dataset_key] = video_name

    return mapping


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_video_name(name):
    if not isinstance(name, str):
        return ""
    normalized = name.lower().strip()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def _build_normalized_key_index(data_dict):
    index = {}
    for key in data_dict.keys():
        norm_key = _normalize_video_name(key)
        if norm_key and norm_key not in index:
            index[norm_key] = key
    return index


def _compose_pick_scores(
    scene_score_map,
    frame_scene_data,
    frame_video_data,
    alpha_scene_frame=1.0
):
    frame_video_frames = frame_video_data.get(
        "frames", []) if isinstance(frame_video_data, dict) else []
    fv_by_pick = {
        int(item.get("pick")): _safe_float(item.get("text_sim", 0.0))
        for item in frame_video_frames
        if isinstance(item, dict) and item.get("pick") is not None
    }

    formulas = {
        "s_mul_fs_add_fv": {},   # f_j = s_i/100 * f_s_j + f_v_j
        "s_mul_fs": {},          # f_j = s_i/100 * f_s_j
        "s_only": {},            # f_j = s_i/100
        "s_add_fs": {}           # f_j = a * (s_i/100) + f_s_j
    }

    stats = {
        "num_scene_scores": len(scene_score_map) if isinstance(scene_score_map, dict) else 0,
        "num_scenes_in_frame_scene": len(frame_scene_data) if isinstance(frame_scene_data, list) else 0,
        "num_picks_with_frame_video": len(fv_by_pick),
        "missing_scene_score_count": 0,
    }

    if not isinstance(frame_scene_data, list):
        return formulas, stats

    for scene_idx, scene_item in enumerate(frame_scene_data):
        scene_score = _safe_float(scene_score_map.get(
            str(scene_idx), 0.0), 0.0) / 100.0
        if str(scene_idx) not in scene_score_map:
            stats["missing_scene_score_count"] += 1

        frames = scene_item.get("frames", []) if isinstance(
            scene_item, dict) else []
        for frame_item in frames:
            if not isinstance(frame_item, dict) or frame_item.get("pick") is None:
                continue

            pick = int(frame_item["pick"])
            f_scene = _safe_float(frame_item.get("sim", 0.0), 0.0)
            f_video = fv_by_pick.get(pick, 0.0)
            scene_frame_term = alpha_scene_frame * (scene_score * f_scene)

            formulas["s_mul_fs_add_fv"][pick] = scene_frame_term + f_video
            formulas["s_mul_fs"][pick] = scene_frame_term
            formulas["s_only"][pick] = scene_score
            formulas["s_add_fs"][pick] = alpha_scene_frame * \
                scene_score + f_scene

    return formulas, stats


def _extract_human_scores(dataset_video_data, dataset_name):
    if "user_scores" in dataset_video_data:
        user_scores = dataset_video_data["user_scores"]
    elif "user_summary" in dataset_video_data:
        user_scores = dataset_video_data["user_summary"]
    else:
        raise ValueError(
            f"Missing user_scores/user_summary for {dataset_name}")

    if dataset_name == "tvsum":
        user_scores = np.asarray(user_scores)
        return [user_scores[i, :] for i in range(user_scores.shape[0])]

    return np.asarray(user_scores)


def _align_prediction_to_picks(pick_score_map, picks):
    aligned_scores = []
    for pick in picks:
        aligned_scores.append(_safe_float(
            pick_score_map.get(int(pick), 0.0), 0.0))
    return np.asarray(aligned_scores, dtype=np.float32)


def _evaluate_one_dataset(
    dataset_name,
    dataset_dict,
    scene_scores_data,
    frame_scene_data,
    frame_video_data,
    video_mapping,
    apply_temporal_smoothing=True,
    temporal_smoothing_norm="none",
    eval_video_keys=None,
    alpha_scene_frame=1.0
):
    f1_metric = "max" if dataset_name == "summe" else "avg"
    formula_names = ["s_mul_fs_add_fv", "s_mul_fs", "s_only", "s_add_fs"]

    scene_key_index = _build_normalized_key_index(scene_scores_data)
    frame_scene_key_index = _build_normalized_key_index(frame_scene_data)
    frame_video_key_index = _build_normalized_key_index(frame_video_data)

    per_formula_predictions = {name: [] for name in formula_names}
    per_formula_humans = {name: [] for name in formula_names}
    per_formula_f1 = {name: [] for name in formula_names}
    per_formula_details = {name: [] for name in formula_names}

    for dataset_video_key, source_video_name in video_mapping.items():
        if eval_video_keys is not None and dataset_video_key not in eval_video_keys:
            continue

        exact_name = source_video_name
        normalized_name = _normalize_video_name(source_video_name)

        scene_key = exact_name if exact_name in scene_scores_data else scene_key_index.get(
            normalized_name)
        frame_scene_key = exact_name if exact_name in frame_scene_data else frame_scene_key_index.get(
            normalized_name)
        frame_video_key = exact_name if exact_name in frame_video_data else frame_video_key_index.get(
            normalized_name)

        scene_map = scene_scores_data.get(
            scene_key) if scene_key is not None else None
        frame_scene = frame_scene_data.get(
            frame_scene_key) if frame_scene_key is not None else None
        frame_video = frame_video_data.get(
            frame_video_key) if frame_video_key is not None else None
        video_data = dataset_dict.get(dataset_video_key)

        if scene_map is None or frame_scene is None or frame_video is None or video_data is None:
            for fname in formula_names:
                per_formula_details[fname].append({
                    "video_key": dataset_video_key,
                    "video_name": source_video_name,
                    "status": "missing_inputs"
                })
            continue

        try:
            formulas, compose_stats = _compose_pick_scores(
                scene_map,
                frame_scene,
                frame_video,
                alpha_scene_frame=alpha_scene_frame
            )
            picks = np.asarray(video_data["picks"], dtype=np.int32)
            cps = np.asarray(video_data["change_points"], dtype=np.int32)
            nfps = np.asarray(
                video_data["n_frame_per_seg"], dtype=np.int32).tolist()
            n_frames = int(video_data["n_frames"])
            user_summary = np.asarray(
                video_data["user_summary"], dtype=np.float32)
            human_scores = _extract_human_scores(video_data, dataset_name)

            for fname in formula_names:
                pred_raw = _align_prediction_to_picks(formulas[fname], picks)

                if apply_temporal_smoothing:
                    smoothing_video_data = _build_temporal_smoothing_video_data(
                        picks=picks,
                        frame_scene_data=frame_scene,
                        pick_score_map=formulas[fname]
                    )
                    pred = np.asarray(
                        temporal_smoothing_func(
                            smoothing_video_data,
                            norm=temporal_smoothing_norm
                        ),
                        dtype=np.float32
                    )
                    if pred.size != pred_raw.size:
                        pred = pred_raw
                else:
                    pred = pred_raw

                if pred.size != picks.size:
                    per_formula_details[fname].append({
                        "video_key": dataset_video_key,
                        "video_name": source_video_name,
                        "status": "length_mismatch",
                        "pred_len": int(pred.size),
                        "picks_len": int(picks.size)
                    })
                    continue

                summary, _, _ = build_frame_summary_from_segments(
                    predicted_scores=pred,
                    change_points=cps,
                    total_frames=n_frames,
                    frames_per_segment=nfps,
                    sampled_positions=picks,
                    summary_ratio=0.15,
                    selection_method="knapsack"
                )

                f1, precision, recall = evaluate_f1_frame_summary(
                    machine_summary=summary,
                    human_summaries=user_summary,
                    reduction=f1_metric
                )

                per_formula_predictions[fname].append(pred)
                per_formula_humans[fname].append(human_scores)
                per_formula_f1[fname].append(f1)
                per_formula_details[fname].append({
                    "video_key": dataset_video_key,
                    "video_name": source_video_name,
                    "status": "ok",
                    "matched_scene_key": scene_key,
                    "matched_frame_scene_key": frame_scene_key,
                    "matched_frame_video_key": frame_video_key,
                    "n_picks": int(pred.size),
                    "temporal_smoothing": bool(apply_temporal_smoothing),
                    "smoothing_norm": temporal_smoothing_norm,
                    "alpha_scene_frame": float(alpha_scene_frame),
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "compose_stats": compose_stats
                })

        except Exception as exc:
            for fname in formula_names:
                per_formula_details[fname].append({
                    "video_key": dataset_video_key,
                    "video_name": source_video_name,
                    "status": "error",
                    "message": str(exc)
                })

    results = {}
    for fname in formula_names:
        if per_formula_predictions[fname]:
            rho, tau = evaluate_rank_correlation_batch(
                per_formula_predictions[fname],
                per_formula_humans[fname],
                reduction="avg"
            )
            mean_f1 = float(
                np.mean(per_formula_f1[fname])) if per_formula_f1[fname] else 0.0
        else:
            rho, tau, mean_f1 = 0.0, 0.0, 0.0

        results[fname] = {
            "mean_f1": float(mean_f1),
            "mean_rho": float(rho),
            "mean_tau": float(tau),
            "num_videos": len(per_formula_predictions[fname]),
            "metric_f1_reduction": f1_metric,
            "details": per_formula_details[fname]
        }

    return results


def _aggregate_split_formula_results(split_formula_results):
    if not split_formula_results:
        return {
            "mean_f1": 0.0,
            "mean_rho": 0.0,
            "mean_tau": 0.0,
            "num_videos_avg_per_split": 0.0,
            "num_splits": 0,
            "split_results": [],
            "best_split": None
        }

    mean_f1 = float(np.mean([x["mean_f1"] for x in split_formula_results]))
    mean_rho = float(np.mean([x["mean_rho"] for x in split_formula_results]))
    mean_tau = float(np.mean([x["mean_tau"] for x in split_formula_results]))
    avg_n = float(np.mean([x["num_videos"] for x in split_formula_results]))

    best_split = sorted(
        split_formula_results,
        key=lambda x: (x["mean_f1"], x["mean_rho"], x["mean_tau"]),
        reverse=True
    )[0]

    return {
        "mean_f1": mean_f1,
        "mean_rho": mean_rho,
        "mean_tau": mean_tau,
        "num_videos_avg_per_split": avg_n,
        "num_splits": len(split_formula_results),
        "split_results": split_formula_results,
        "best_split": best_split
    }


def run_exam_score_evaluation(
    scores_root="/root/tfnet/data/scores",
    scene_score_source="gpt5",
    summe_h5_path="/root/autodl-tmp/datasets/eccv16_dataset_summe_google_pool5.h5",
    tvsum_h5_path="/root/autodl-tmp/datasets/eccv16_dataset_tvsum_google_pool5.h5",
    video_name_dict_path="/root/tfnet/data/video_name_dict.json",
    split_root="/root/autodl-tmp/datasets/splits",
    split_count=5,
    output_dir="/root/tfnet/data/scroe/exam_score",
    output_file_name=None,
    apply_temporal_smoothing=True,
    temporal_smoothing_norm="none",
    alpha_scene_frame=1.0
):
    scene_score_summe_path = os.path.join(
        scores_root, "scene_score", scene_score_source, "summe_scene_scores.json")
    scene_score_tvsum_path = os.path.join(
        scores_root, "scene_score", scene_score_source, "tvsum_scene_scores.json")
    frame_scene_summe_path = os.path.join(
        scores_root, "frame_scene_contribution", "summe.json")
    frame_scene_tvsum_path = os.path.join(
        scores_root, "frame_scene_contribution", "tvsum.json")
    frame_video_summe_path = os.path.join(
        scores_root, "frame_video_contribution", "summe.json")
    frame_video_tvsum_path = os.path.join(
        scores_root, "frame_video_contribution", "tvsum.json")
    summe_split_path = os.path.join(
        split_root, f"summe_splits_{split_count}.json")
    tvsum_split_path = os.path.join(
        split_root, f"tvsum_splits_{split_count}.json")

    required_paths = [
        scene_score_summe_path,
        scene_score_tvsum_path,
        frame_scene_summe_path,
        frame_scene_tvsum_path,
        frame_video_summe_path,
        frame_video_tvsum_path,
        summe_h5_path,
        tvsum_h5_path,
        video_name_dict_path,
        summe_split_path,
        tvsum_split_path,
    ]

    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Required files are missing: {missing}")

    scene_score_summe = _load_json(scene_score_summe_path)
    scene_score_tvsum = _load_json(scene_score_tvsum_path)
    frame_scene_summe = _load_json(frame_scene_summe_path)
    frame_scene_tvsum = _load_json(frame_scene_tvsum_path)
    frame_video_summe = _load_json(frame_video_summe_path)
    frame_video_tvsum = _load_json(frame_video_tvsum_path)
    summe_splits = _load_json(summe_split_path)
    tvsum_splits = _load_json(tvsum_split_path)

    summe_dataset = _load_h5_dataset_as_dict(summe_h5_path)
    tvsum_dataset = _load_h5_dataset_as_dict(tvsum_h5_path)

    summe_mapping = _build_video_key_mapping(
        "summe",
        summe_dataset,
        video_name_dict_path
    )
    tvsum_mapping = _build_video_key_mapping(
        "tvsum",
        tvsum_dataset,
        video_name_dict_path
    )

    formula_names = ["s_mul_fs_add_fv", "s_mul_fs", "s_only", "s_add_fs"]

    summe_split_formula_results = {f: [] for f in formula_names}
    for split_id, split_info in enumerate(summe_splits, start=1):
        split_eval = _evaluate_one_dataset(
            dataset_name="summe",
            dataset_dict=summe_dataset,
            scene_scores_data=scene_score_summe,
            frame_scene_data=frame_scene_summe,
            frame_video_data=frame_video_summe,
            video_mapping=summe_mapping,
            apply_temporal_smoothing=apply_temporal_smoothing,
            temporal_smoothing_norm=temporal_smoothing_norm,
            eval_video_keys=set(split_info.get("test_keys", [])),
            alpha_scene_frame=alpha_scene_frame
        )
        for formula in formula_names:
            summe_split_formula_results[formula].append({
                "split_id": split_id,
                "test_keys": split_info.get("test_keys", []),
                "mean_f1": split_eval[formula]["mean_f1"],
                "mean_rho": split_eval[formula]["mean_rho"],
                "mean_tau": split_eval[formula]["mean_tau"],
                "num_videos": split_eval[formula]["num_videos"]
            })

    tvsum_split_formula_results = {f: [] for f in formula_names}
    for split_id, split_info in enumerate(tvsum_splits, start=1):
        split_eval = _evaluate_one_dataset(
            dataset_name="tvsum",
            dataset_dict=tvsum_dataset,
            scene_scores_data=scene_score_tvsum,
            frame_scene_data=frame_scene_tvsum,
            frame_video_data=frame_video_tvsum,
            video_mapping=tvsum_mapping,
            apply_temporal_smoothing=apply_temporal_smoothing,
            temporal_smoothing_norm=temporal_smoothing_norm,
            eval_video_keys=set(split_info.get("test_keys", [])),
            alpha_scene_frame=alpha_scene_frame
        )
        for formula in formula_names:
            tvsum_split_formula_results[formula].append({
                "split_id": split_id,
                "test_keys": split_info.get("test_keys", []),
                "mean_f1": split_eval[formula]["mean_f1"],
                "mean_rho": split_eval[formula]["mean_rho"],
                "mean_tau": split_eval[formula]["mean_tau"],
                "num_videos": split_eval[formula]["num_videos"]
            })

    summe_results = {
        formula: {
            **_aggregate_split_formula_results(summe_split_formula_results[formula]),
            "metric_f1_reduction": "max"
        }
        for formula in formula_names
    }

    tvsum_results = {
        formula: {
            **_aggregate_split_formula_results(tvsum_split_formula_results[formula]),
            "metric_f1_reduction": "avg"
        }
        for formula in formula_names
    }

    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for dataset_name, dataset_results in [("summe", summe_results), ("tvsum", tvsum_results)]:
        for formula_name, values in dataset_results.items():
            summary_rows.append({
                "dataset": dataset_name,
                "formula": formula_name,
                "mean_f1": values["mean_f1"],
                "mean_rho": values["mean_rho"],
                "mean_tau": values["mean_tau"],
                "num_videos": values["num_videos_avg_per_split"],
                "num_splits": values["num_splits"]
            })

    best_by_dataset = {}
    for dataset_name in ["summe", "tvsum"]:
        candidates = [r for r in summary_rows if r["dataset"] == dataset_name]
        if not candidates:
            continue
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (x["mean_f1"], x["mean_rho"], x["mean_tau"]),
            reverse=True
        )
        best_by_dataset[dataset_name] = candidates_sorted[0]

    output_data = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "description": "Evaluate frame-score composition experiments from data/scores with four formulas.",
            "temporal_smoothing": {
                "enabled": bool(apply_temporal_smoothing),
                "norm": temporal_smoothing_norm,
                "method": "scene-level cosine transition smoothing"
            },
            "alpha_scene_frame": float(alpha_scene_frame),
            "score_formulas": {
                "s_mul_fs_add_fv": "f_j = a * (s_i/100 * f_s_j) + f_v_j",
                "s_mul_fs": "f_j = a * (s_i/100 * f_s_j)",
                "s_only": "f_j = s_i/100",
                "s_add_fs": "f_j = a * (s_i/100) + f_s_j"
            },
            "input_paths": {
                "scores_root": scores_root,
                "scene_score_source": scene_score_source,
                "summe_h5_path": summe_h5_path,
                "tvsum_h5_path": tvsum_h5_path,
                "video_name_dict_path": video_name_dict_path,
                "summe_split_path": summe_split_path,
                "tvsum_split_path": tvsum_split_path
            },
            "notes": [
                "scene score is normalized by dividing 100",
                "a is an adjustable coefficient for scene-score terms in s_mul_fs_add_fv, s_mul_fs and s_add_fs",
                "F1 reduction uses max for SumMe and avg for TVSum",
                "Rank correlation uses Spearman rho and Kendall tau-b",
                "Metrics are computed on each split test set, then averaged across splits",
                "best_split in each formula is selected by (mean_f1, mean_rho, mean_tau)"
            ]
        },
        "summary": {
            "rows": summary_rows,
            "best_by_dataset": best_by_dataset
        },
        "results": {
            "summe": summe_results,
            "tvsum": tvsum_results
        }
    }

    if output_file_name is None:
        output_file_name = "exam_evaluation_results_smooth.json" if apply_temporal_smoothing else "exam_evaluation_results_no_smooth.json"

    output_path = os.path.join(output_dir, output_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_path, output_data


def run_exam_score_evaluation_dual(
    scores_root="/root/tfnet/data/scores",
    scene_score_source="gpt5",
    summe_h5_path="/root/autodl-tmp/datasets/eccv16_dataset_summe_google_pool5.h5",
    tvsum_h5_path="/root/autodl-tmp/datasets/eccv16_dataset_tvsum_google_pool5.h5",
    video_name_dict_path="/root/tfnet/data/video_name_dict.json",
    split_root="/root/autodl-tmp/datasets/splits",
    split_count=5,
    output_dir="/root/tfnet/data/scroe/exam_score",
    temporal_smoothing_norm="none",
    alpha_scene_frame=1.0
):
    no_smooth_path, no_smooth_data = run_exam_score_evaluation(
        scores_root=scores_root,
        scene_score_source=scene_score_source,
        summe_h5_path=summe_h5_path,
        tvsum_h5_path=tvsum_h5_path,
        video_name_dict_path=video_name_dict_path,
        split_root=split_root,
        split_count=split_count,
        output_dir=output_dir,
        output_file_name="exam_evaluation_results_no_smooth.json",
        apply_temporal_smoothing=False,
        temporal_smoothing_norm=temporal_smoothing_norm,
        alpha_scene_frame=alpha_scene_frame
    )

    smooth_path, smooth_data = run_exam_score_evaluation(
        scores_root=scores_root,
        scene_score_source=scene_score_source,
        summe_h5_path=summe_h5_path,
        tvsum_h5_path=tvsum_h5_path,
        video_name_dict_path=video_name_dict_path,
        split_root=split_root,
        split_count=split_count,
        output_dir=output_dir,
        output_file_name="exam_evaluation_results_smooth.json",
        apply_temporal_smoothing=True,
        temporal_smoothing_norm=temporal_smoothing_norm,
        alpha_scene_frame=alpha_scene_frame
    )

    compare_output = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "description": "Side-by-side summary for no_smooth vs smooth split-based evaluation"
        },
        "paths": {
            "no_smooth": no_smooth_path,
            "smooth": smooth_path
        },
        "summary": {
            "no_smooth": no_smooth_data.get("summary", {}),
            "smooth": smooth_data.get("summary", {})
        }
    }

    compare_path = os.path.join(
        output_dir, "exam_evaluation_results_compare.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(compare_output, f, ensure_ascii=False, indent=2)

    return {
        "no_smooth": no_smooth_path,
        "smooth": smooth_path,
        "compare": compare_path
    }


def run_exam_score_alpha_sweep_dual(
    scores_root="/root/tfnet/data/scores",
    scene_score_source="gpt5",
    summe_h5_path="/root/autodl-tmp/datasets/eccv16_dataset_summe_google_pool5.h5",
    tvsum_h5_path="/root/autodl-tmp/datasets/eccv16_dataset_tvsum_google_pool5.h5",
    video_name_dict_path="/root/tfnet/data/video_name_dict.json",
    split_root="/root/autodl-tmp/datasets/splits",
    split_count=5,
    output_dir="/root/tfnet/data/scroe/exam_score",
    temporal_smoothing_norm="none",
    start=0.0,
    end=1.0,
    step=0.1
):
    os.makedirs(output_dir, exist_ok=True)

    alphas = []
    current = float(start)
    while current <= float(end) + 1e-9:
        alphas.append(round(current, 1))
        current += float(step)

    sweep_rows = []

    for alpha in alphas:
        no_smooth_name = f"exam_evaluation_results_no_smooth_a_{alpha:.1f}.json"
        smooth_name = f"exam_evaluation_results_smooth_a_{alpha:.1f}.json"

        no_smooth_path, no_smooth_data = run_exam_score_evaluation(
            scores_root=scores_root,
            scene_score_source=scene_score_source,
            summe_h5_path=summe_h5_path,
            tvsum_h5_path=tvsum_h5_path,
            video_name_dict_path=video_name_dict_path,
            split_root=split_root,
            split_count=split_count,
            output_dir=output_dir,
            output_file_name=no_smooth_name,
            apply_temporal_smoothing=False,
            temporal_smoothing_norm=temporal_smoothing_norm,
            alpha_scene_frame=alpha
        )

        smooth_path, smooth_data = run_exam_score_evaluation(
            scores_root=scores_root,
            scene_score_source=scene_score_source,
            summe_h5_path=summe_h5_path,
            tvsum_h5_path=tvsum_h5_path,
            video_name_dict_path=video_name_dict_path,
            split_root=split_root,
            split_count=split_count,
            output_dir=output_dir,
            output_file_name=smooth_name,
            apply_temporal_smoothing=True,
            temporal_smoothing_norm=temporal_smoothing_norm,
            alpha_scene_frame=alpha
        )

        sweep_rows.append({
            "alpha": alpha,
            "no_smooth": {
                "path": no_smooth_path,
                "best_by_dataset": no_smooth_data.get("summary", {}).get("best_by_dataset", {})
            },
            "smooth": {
                "path": smooth_path,
                "best_by_dataset": smooth_data.get("summary", {}).get("best_by_dataset", {})
            }
        })

    def _pick_best(rows, smoothing_key, dataset_name):
        candidates = []
        for row in rows:
            info = row.get(smoothing_key, {}).get(
                "best_by_dataset", {}).get(dataset_name)
            if info is None:
                continue
            candidates.append({
                "alpha": row["alpha"],
                "formula": info.get("formula"),
                "mean_f1": info.get("mean_f1", 0.0),
                "mean_rho": info.get("mean_rho", 0.0),
                "mean_tau": info.get("mean_tau", 0.0)
            })
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda x: (x["mean_f1"], x["mean_rho"], x["mean_tau"]),
            reverse=True
        )[0]

    summary = {
        "best_alpha": {
            "no_smooth": {
                "summe": _pick_best(sweep_rows, "no_smooth", "summe"),
                "tvsum": _pick_best(sweep_rows, "no_smooth", "tvsum")
            },
            "smooth": {
                "summe": _pick_best(sweep_rows, "smooth", "summe"),
                "tvsum": _pick_best(sweep_rows, "smooth", "tvsum")
            }
        }
    }

    sweep_output = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "description": "Alpha sweep for scene-score*frame-scene term: a in [0.0, 1.0], step=0.1",
            "alpha_range": {
                "start": start,
                "end": end,
                "step": step,
                "values": alphas
            }
        },
        "rows": sweep_rows,
        "summary": summary
    }

    sweep_path = os.path.join(
        output_dir, "exam_evaluation_results_alpha_sweep.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_output, f, ensure_ascii=False, indent=2)

    return {
        "sweep_summary": sweep_path,
        "num_alphas": len(alphas)
    }


# ============================================================
# Demo
# ============================================================

def _demo_run():
    total_frames = 30
    change_points = np.array([[0, 9], [10, 19], [20, 29]])
    frames_per_segment = [10, 10, 10]
    sampled_positions = np.array([0, 5, 10, 15, 20, 30])
    predicted_scores = np.array([0.2, 0.8, 0.3, 0.7, 0.1])

    summary, segment_scores, selected_segments = build_frame_summary_from_segments(
        predicted_scores,
        change_points,
        total_frames,
        frames_per_segment,
        sampled_positions,
        summary_ratio=0.5,
        selection_method="knapsack"
    )

    print("segment_scores:", segment_scores)
    print("selected_segments:", selected_segments)
    print("summary length:", int(summary.sum()))

    human_summaries = np.zeros((2, total_frames))
    human_summaries[0, 8:12] = 1
    human_summaries[1, 20:24] = 1

    f1, p, r = evaluate_f1_frame_summary(summary, human_summaries)
    print("F1 / P / R:", f1, p, r)

    human_scores = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
    rho, tau = evaluate_rank_correlation(predicted_scores, human_scores)
    print("Spearman / Kendall:", rho, tau)


if __name__ == "__main__":
    try:
        output_paths = run_exam_score_evaluation_dual()
        print("Saved exam evaluation results:")
        for key, value in output_paths.items():
            print(f"- {key}: {value}")

        alpha_sweep_info = run_exam_score_alpha_sweep_dual()
        print("Saved alpha sweep summary:")
        print(f"- sweep_summary: {alpha_sweep_info['sweep_summary']}")
        print(f"- num_alphas: {alpha_sweep_info['num_alphas']}")

        _, out_data = run_exam_score_evaluation(
            output_file_name="exam_evaluation_results_smooth.json",
            apply_temporal_smoothing=True
        )
        for item in out_data["summary"]["rows"]:
            print(
                f"[{item['dataset']}] {item['formula']}: "
                f"F1={item['mean_f1']:.4f}, "
                f"rho={item['mean_rho']:.4f}, "
                f"tau={item['mean_tau']:.4f}, "
                f"n_avg={item['num_videos']:.2f}, "
                f"splits={item['num_splits']}"
            )
    except Exception as e:
        print("Exam score evaluation failed:", str(e))
