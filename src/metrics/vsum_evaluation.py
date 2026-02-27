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

import math
import numpy as np

try:
    from ortools.algorithms import pywrapknapsack_solver as _ortools_knapsack
except Exception:
    try:
        from ortools.algorithms.python import knapsack_solver as _ortools_knapsack
    except Exception:
        _ortools_knapsack = None
from scipy.stats import spearmanr, kendalltau


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
    _demo_run()
