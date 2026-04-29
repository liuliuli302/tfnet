"""
Batch-generate per-video result JSON files and overview plots for all
video summarization experiments.

Outputs:
    figure/result/<dataset>/<video_slug>/results.json
    figure/result/<dataset>/<video_slug>/overview.png
    figure/result/manifest.json
    figure/best/<dataset>/top*.json
    figure/best/<dataset>/top*.png

Usage:
    python figure/plot_score_curves.py
    python figure/plot_score_curves.py --resume
    python figure/plot_score_curves.py --mode pair \
        --summe_video Excavators_river_crossing \
        --tvsum_video i3wAGJaaktw
"""
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.metrics.vsum_evaluation import (  # noqa: E402
    _build_normalized_key_index,
    _build_temporal_smoothing_video_data,
    _build_video_key_mapping,
    _compose_pick_scores,
    _load_h5_dataset_as_dict,
    _normalize_video_name,
    _safe_float,
    build_frame_summary_from_segments,
    evaluate_f1_frame_summary,
    temporal_smoothing_func,
)


SCORE_ROOT = os.path.join(BASE, "data", "scores")
EXPERIMENT_ROOT = os.path.join(BASE, "data", "scroe", "exam_score")
RESULT_ROOT = os.path.join(BASE, "figure", "result")
BEST_ROOT = os.path.join(BASE, "figure", "best")

SUMME_H5 = os.path.join(BASE, "data", "feature",
                        "eccv16_dataset_summe_ViT_L_14.h5")
TVSUM_H5 = os.path.join(BASE, "data", "feature",
                        "eccv16_dataset_tvsum_ViT_L_14.h5")
VIDEO_NAME_DICT = os.path.join(BASE, "data", "video_name_dict.json")

FORMULA_EXPRESSIONS = {
    "s_mul_fs_add_fv": r"$F(p_j)=\alpha\,S_{\mathrm{LLM}}\,S_{FS}+S_{FV}$",
    "s_mul_fs": r"$F(p_j)=\alpha\,S_{\mathrm{LLM}}\,S_{FS}$",
    "s_only": r"$F(p_j)=S_{\mathrm{LLM}}$",
    "s_add_fs": r"$F(p_j)=\alpha\,S_{\mathrm{LLM}}+S_{FS}$",
}

FORMULA_TEXT = {
    "s_mul_fs_add_fv": "f_j = a * (s_i/100 * f_s_j) + f_v_j",
    "s_mul_fs": "f_j = a * (s_i/100 * f_s_j)",
    "s_only": "f_j = s_i/100",
    "s_add_fs": "f_j = a * (s_i/100) + f_s_j",
}

SCENE_SCORE_SOURCE_LABELS = {
    "gpt5": "gpt5",
    "deepseek": "deepseek32",
    "deepseek32": "deepseek32",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_LLM = "#2166AC"
C_FS = "#4DAC26"
C_FV = "#D6604D"
C_FINAL = "#1B1B1B"
C_GT_FILL = "#A8ADB5"
C_GT_LINE = "#6D737C"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _to_rel(path):
    return os.path.relpath(path, BASE).replace(os.sep, "/")


def _slugify(text):
    normalized = _normalize_video_name(text)
    return normalized or "unknown_video"


def _parse_alpha_from_name(file_name):
    match = re.search(r"_a_(\d+(?:\.\d+)?)\.json$", file_name)
    if match:
        return float(match.group(1))
    return 1.0


def _parse_smoothing_from_name(file_name):
    lowered = file_name.lower()
    if "no_smooth" in lowered:
        return False
    if "smooth" in lowered:
        return True
    return True


def _canonical_scene_score_source(value, source_path=""):
    raw = (value or "").strip().lower()
    if raw in SCENE_SCORE_SOURCE_LABELS:
        return SCENE_SCORE_SOURCE_LABELS[raw]

    lowered_path = source_path.lower()
    if "deepseek" in lowered_path:
        return "deepseek32"
    return "gpt5"


def _discover_experiment_files(root_dir):
    experiment_files = []
    for current_root, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if not file_name.endswith(".json"):
                continue
            if not file_name.startswith("exam_evaluation_results"):
                continue
            if "compare" in file_name or "alpha_sweep" in file_name:
                continue
            experiment_files.append(os.path.join(current_root, file_name))

    return sorted(experiment_files, key=lambda item: _to_rel(item))


def _build_dataset_contexts():
    summe_dataset = _load_h5_dataset_as_dict(SUMME_H5)
    tvsum_dataset = _load_h5_dataset_as_dict(TVSUM_H5)

    return {
        "summe": {
            "dataset": summe_dataset,
            "mapping": _build_video_key_mapping("summe", summe_dataset, VIDEO_NAME_DICT),
            "f1_reduction": "max",
        },
        "tvsum": {
            "dataset": tvsum_dataset,
            "mapping": _build_video_key_mapping("tvsum", tvsum_dataset, VIDEO_NAME_DICT),
            "f1_reduction": "avg",
        },
    }


def _load_score_assets(scene_score_sources):
    frame_scene = {
        dataset_name: _load_json(os.path.join(
            SCORE_ROOT, "frame_scene_contribution", f"{dataset_name}.json"))
        for dataset_name in ("summe", "tvsum")
    }
    frame_video = {
        dataset_name: _load_json(os.path.join(
            SCORE_ROOT, "frame_video_contribution", f"{dataset_name}.json"))
        for dataset_name in ("summe", "tvsum")
    }

    assets = {}
    for source in sorted(scene_score_sources):
        assets[source] = {}
        for dataset_name in ("summe", "tvsum"):
            scene_scores = _load_json(
                os.path.join(SCORE_ROOT, "scene_score", source,
                             f"{dataset_name}_scene_scores.json")
            )
            assets[source][dataset_name] = {
                "scene_scores": scene_scores,
                "scene_index": _build_normalized_key_index(scene_scores),
                "frame_scene": frame_scene[dataset_name],
                "frame_scene_index": _build_normalized_key_index(frame_scene[dataset_name]),
                "frame_video": frame_video[dataset_name],
                "frame_video_index": _build_normalized_key_index(frame_video[dataset_name]),
            }
    return assets


def _extract_gt_scores(video_data):
    user_summary = np.asarray(video_data["user_summary"], dtype=np.float32)
    if user_summary.ndim == 1:
        user_summary = user_summary[None, :]

    picks = np.asarray(video_data["picks"], dtype=np.int32)
    if user_summary.shape[1] == 0:
        return np.zeros_like(picks, dtype=np.float32)

    picks = np.clip(picks, 0, user_summary.shape[1] - 1)
    return np.mean(user_summary[:, picks], axis=0).astype(np.float32)


def _align_pick_map_to_array(pick_score_map, picks):
    return np.asarray([
        _safe_float(pick_score_map.get(int(pick), 0.0), 0.0)
        for pick in picks
    ], dtype=np.float32)


def _collect_component_curves(scene_map, frame_scene_data, frame_video_data, picks, alpha_scene_frame):
    pick_to_index = {int(pick): idx for idx, pick in enumerate(
        np.asarray(picks, dtype=np.int32).tolist())}
    s_llm = np.zeros(len(picks), dtype=np.float32)
    s_fs = np.zeros(len(picks), dtype=np.float32)
    s_fv = np.zeros(len(picks), dtype=np.float32)

    frame_video_frames = frame_video_data.get(
        "frames", []) if isinstance(frame_video_data, dict) else []
    fv_by_pick = {
        int(item.get("pick")): _safe_float(item.get("text_sim", 0.0), 0.0)
        for item in frame_video_frames
        if isinstance(item, dict) and item.get("pick") is not None
    }

    for scene_idx, scene_item in enumerate(frame_scene_data):
        scene_score = _safe_float(scene_map.get(
            str(scene_idx), 0.0), 0.0) / 100.0
        frames = scene_item.get("frames", []) if isinstance(
            scene_item, dict) else []
        for frame_item in frames:
            if not isinstance(frame_item, dict) or frame_item.get("pick") is None:
                continue

            pick = int(frame_item["pick"])
            if pick not in pick_to_index:
                continue

            array_index = pick_to_index[pick]
            s_llm[array_index] = scene_score
            s_fs[array_index] = _safe_float(frame_item.get("sim", 0.0), 0.0)
            s_fv[array_index] = fv_by_pick.get(pick, 0.0)

    raw_formula_scores = {
        "s_mul_fs_add_fv": alpha_scene_frame * s_llm * s_fs + s_fv,
        "s_mul_fs": alpha_scene_frame * s_llm * s_fs,
        "s_only": s_llm,
        "s_add_fs": alpha_scene_frame * s_llm + s_fs,
    }

    return {
        "s_llm": s_llm,
        "s_fs": s_fs,
        "s_fv": s_fv,
        "raw_formula_scores": raw_formula_scores,
    }


def _resolve_score_inputs(source_video_name, score_assets):
    exact_name = source_video_name
    normalized_name = _normalize_video_name(source_video_name)

    scene_key = exact_name if exact_name in score_assets["scene_scores"] else score_assets["scene_index"].get(
        normalized_name)
    frame_scene_key = exact_name if exact_name in score_assets["frame_scene"] else score_assets["frame_scene_index"].get(
        normalized_name)
    frame_video_key = exact_name if exact_name in score_assets["frame_video"] else score_assets["frame_video_index"].get(
        normalized_name)

    return {
        "scene_key": scene_key,
        "frame_scene_key": frame_scene_key,
        "frame_video_key": frame_video_key,
        "scene_map": score_assets["scene_scores"].get(scene_key) if scene_key is not None else None,
        "frame_scene": score_assets["frame_scene"].get(frame_scene_key) if frame_scene_key is not None else None,
        "frame_video": score_assets["frame_video"].get(frame_video_key) if frame_video_key is not None else None,
    }


def _experiment_id(config, formula):
    source_stem = os.path.splitext(os.path.basename(config["path"]))[
        0] if config["path"] else "manual"
    smooth_tag = "smooth" if config["temporal_smoothing"]["enabled"] else "nosmooth"
    digest = hashlib.sha1(
        f"{config['relative_source']}|{formula}".encode("utf-8")).hexdigest()[:8]
    source_prefix = _slugify(config["source_group"]) or "experiment"
    return f"{source_prefix}__{source_stem}__{smooth_tag}__a_{config['alpha_scene_frame']:.1f}__{formula}__{digest}"


def _build_label(config, formula):
    smooth_text = "smooth" if config["temporal_smoothing"]["enabled"] else "no-smooth"
    return f"{config['scene_score_source']} | {smooth_text} | a={config['alpha_scene_frame']:.1f} | {formula}"


def _load_experiment_configs():
    experiment_files = _discover_experiment_files(EXPERIMENT_ROOT)
    configs = []
    for path in experiment_files:
        data = _load_json(path)
        if not isinstance(data, dict) or "results" not in data:
            continue

        meta = data.get("meta", {})
        temporal_meta = meta.get("temporal_smoothing", {})
        input_paths = meta.get("input_paths", {})
        relative_source = os.path.relpath(
            path, EXPERIMENT_ROOT).replace(os.sep, "/")
        folder_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)

        alpha_scene_frame = float(
            meta.get("alpha_scene_frame", _parse_alpha_from_name(file_name)))
        temporal_smoothing = {
            "enabled": bool(temporal_meta.get("enabled", _parse_smoothing_from_name(file_name))),
            "norm": str(temporal_meta.get("norm", "none")),
        }
        scene_score_source = _canonical_scene_score_source(
            input_paths.get("scene_score_source"),
            source_path=relative_source,
        )

        configs.append({
            "path": path,
            "relative_source": relative_source,
            "source_group": folder_name,
            "scene_score_source": scene_score_source,
            "alpha_scene_frame": alpha_scene_frame,
            "temporal_smoothing": temporal_smoothing,
        })

    return configs


def _evaluate_video_experiment(source_video_name, video_data, config, formula, score_assets, f1_reduction):
    resolved = _resolve_score_inputs(source_video_name, score_assets)
    scene_map = resolved["scene_map"]
    frame_scene = resolved["frame_scene"]
    frame_video = resolved["frame_video"]

    base_result = {
        "experiment_id": _experiment_id(config, formula),
        "label": _build_label(config, formula),
        "formula": formula,
        "formula_expression": FORMULA_TEXT[formula],
        "scene_score_source": config["scene_score_source"],
        "alpha_scene_frame": float(config["alpha_scene_frame"]),
        "temporal_smoothing": config["temporal_smoothing"],
        "source_file": config["relative_source"],
        "matched_keys": {
            "scene": resolved["scene_key"],
            "frame_scene": resolved["frame_scene_key"],
            "frame_video": resolved["frame_video_key"],
        },
    }

    if scene_map is None or frame_scene is None or frame_video is None:
        return {
            **base_result,
            "status": "missing_inputs",
        }

    picks = np.asarray(video_data["picks"], dtype=np.int32)
    formulas, compose_stats = _compose_pick_scores(
        scene_map,
        frame_scene,
        frame_video,
        alpha_scene_frame=config["alpha_scene_frame"],
    )
    raw_predicted_scores = _align_pick_map_to_array(formulas[formula], picks)

    if config["temporal_smoothing"]["enabled"]:
        smoothing_video_data = _build_temporal_smoothing_video_data(
            picks=picks,
            frame_scene_data=frame_scene,
            pick_score_map=formulas[formula],
        )
        predicted_scores = np.asarray(
            temporal_smoothing_func(
                smoothing_video_data, norm=config["temporal_smoothing"]["norm"]),
            dtype=np.float32,
        )
        if predicted_scores.size != raw_predicted_scores.size:
            predicted_scores = raw_predicted_scores
    else:
        predicted_scores = raw_predicted_scores

    if predicted_scores.size != picks.size:
        return {
            **base_result,
            "status": "length_mismatch",
            "pred_len": int(predicted_scores.size),
            "picks_len": int(picks.size),
        }

    components = _collect_component_curves(
        scene_map=scene_map,
        frame_scene_data=frame_scene,
        frame_video_data=frame_video,
        picks=picks,
        alpha_scene_frame=config["alpha_scene_frame"],
    )

    summary, _, selected_segments = build_frame_summary_from_segments(
        predicted_scores=predicted_scores,
        change_points=np.asarray(video_data["change_points"], dtype=np.int32),
        total_frames=int(video_data["n_frames"]),
        frames_per_segment=np.asarray(
            video_data["n_frame_per_seg"], dtype=np.int32).tolist(),
        sampled_positions=picks,
        summary_ratio=0.15,
        selection_method="knapsack",
    )

    user_summary = np.asarray(video_data["user_summary"], dtype=np.float32)
    f1, precision, recall = evaluate_f1_frame_summary(
        machine_summary=summary,
        human_summaries=user_summary,
        reduction=f1_reduction,
    )

    return {
        **base_result,
        "status": "ok",
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "f1_reduction": f1_reduction,
        "n_picks": int(picks.size),
        "selected_segments": [int(item) for item in selected_segments],
        "predicted_scores": predicted_scores.astype(np.float32).tolist(),
        "raw_predicted_scores": raw_predicted_scores.astype(np.float32).tolist(),
        "component_scores": {
            "s_llm": components["s_llm"].astype(np.float32).tolist(),
            "s_fs": components["s_fs"].astype(np.float32).tolist(),
            "s_fv": components["s_fv"].astype(np.float32).tolist(),
            "raw_formula_scores": components["raw_formula_scores"][formula].astype(np.float32).tolist(),
        },
        "compose_stats": compose_stats,
    }


def _plot_background_gt(ax, t, gt_scores):
    ax.fill_between(t, 0, gt_scores, color=C_GT_FILL, alpha=0.18, zorder=0)
    ax.plot(t, gt_scores, color=C_GT_LINE,
            linewidth=0.75, linestyle="--", zorder=1)


def _draw_curve(ax, t, gt_scores, y_values, colour, label, fill=True):
    _plot_background_gt(ax, t, gt_scores)
    ax.plot(t, y_values, color=colour, linewidth=0.9, zorder=2)
    if fill:
        ax.fill_between(t, 0, y_values, color=colour, alpha=0.12, zorder=1.5)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.tick_params(direction="in", which="both")


def plot_overview(video_result, out_path):
    best_result = video_result.get("best_result")
    if not best_result:
        return

    picks = np.asarray(video_result["sampled_positions"], dtype=np.float32)
    gt_scores = np.asarray(video_result["gt_scores"], dtype=np.float32)
    t = picks / 15.0

    component_scores = best_result["component_scores"]
    s_llm = np.asarray(component_scores["s_llm"], dtype=np.float32)
    s_fs = np.asarray(component_scores["s_fs"], dtype=np.float32)
    s_fv = np.asarray(component_scores["s_fv"], dtype=np.float32)
    predicted_scores = np.asarray(
        best_result["predicted_scores"], dtype=np.float32)

    fig, axes = plt.subplots(
        4, 1, figsize=(7.0, 5.0), sharex=True,
        gridspec_kw={"hspace": 0.12}
    )

    _draw_curve(axes[0], t, gt_scores, s_llm, C_LLM, r"$S_{\mathrm{LLM}}$")
    _draw_curve(axes[1], t, gt_scores, s_fs, C_FS, r"$S_{FS}$")
    _draw_curve(axes[2], t, gt_scores, s_fv, C_FV, r"$S_{FV}$")
    _draw_curve(axes[3], t, gt_scores, predicted_scores, C_FINAL, r"$F(p_j)$")

    axes[-1].set_xlabel("Time (s)")
    dataset_label = "SumMe" if video_result["dataset"] == "summe" else "TVSum"
    fig.suptitle(
        f"{dataset_label}  —  {video_result['video_name'].replace('_', ' ')}",
        fontsize=11, fontweight="bold", y=0.98
    )
    fig.text(
        0.5,
        0.945,
        f"{best_result['label']}  |  F1={best_result['f1']:.4f}",
        ha="center",
        va="top",
        fontsize=8.5,
    )

    fig.align_ylabels(axes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _build_video_result(dataset_name, video_name, video_data, configs, score_assets, f1_reduction):
    gt_scores = _extract_gt_scores(video_data)
    experiments = []
    for config in configs:
        if config["scene_score_source"] not in score_assets:
            continue
        experiment_assets = score_assets[config["scene_score_source"]][dataset_name]
        for formula in FORMULA_TEXT:
            experiments.append(
                _evaluate_video_experiment(
                    source_video_name=video_name,
                    video_data=video_data,
                    config=config,
                    formula=formula,
                    score_assets=experiment_assets,
                    f1_reduction=f1_reduction,
                )
            )

    valid_experiments = [
        item for item in experiments if item.get("status") == "ok"]
    valid_experiments.sort(
        key=lambda item: (item.get("f1", 0.0), item.get(
            "precision", 0.0), item.get("recall", 0.0)),
        reverse=True,
    )
    for index, item in enumerate(valid_experiments, start=1):
        item["rank_in_video"] = index

    invalid_experiments = [
        item for item in experiments if item.get("status") != "ok"]
    all_experiments = valid_experiments + invalid_experiments
    best_result = valid_experiments[0] if valid_experiments else None

    return {
        "dataset": dataset_name,
        "video_name": video_name,
        "plot_axis": {
            "x_label": "Sampled Frame Position",
            "y_label": "Importance Score",
        },
        "video_meta": {
            "n_frames": int(video_data["n_frames"]),
            "n_picks": int(len(video_data["picks"])),
            "num_user_summaries": int(np.asarray(video_data["user_summary"]).shape[0]),
        },
        "sampled_positions": np.asarray(video_data["picks"], dtype=np.int32).tolist(),
        "gt_scores": gt_scores.tolist(),
        "best_result": best_result,
        "experiments": all_experiments,
    }


def _prepare_output_dirs(resume):
    if resume:
        os.makedirs(RESULT_ROOT, exist_ok=True)
        os.makedirs(BEST_ROOT, exist_ok=True)
        return

    for path in (RESULT_ROOT, BEST_ROOT):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def _write_best_dataset_summary(dataset_name, manifest_rows):
    dataset_best_dir = os.path.join(BEST_ROOT, dataset_name)
    os.makedirs(dataset_best_dir, exist_ok=True)

    top_rows = sorted(
        manifest_rows,
        key=lambda item: item["best_f1"],
        reverse=True,
    )[:5]

    summary = []
    for rank, row in enumerate(top_rows, start=1):
        source_plot = os.path.join(BASE, row["overview_plot"])
        source_json = os.path.join(BASE, row["result_json"])
        dst_name = _slugify(row["video_name"])
        dst_plot = os.path.join(
            dataset_best_dir, f"top{rank:02d}_{dst_name}.png")
        dst_json = os.path.join(
            dataset_best_dir, f"top{rank:02d}_{dst_name}.json")
        shutil.copy2(source_plot, dst_plot)
        shutil.copy2(source_json, dst_json)
        summary.append({
            "rank": rank,
            "video_key": row["video_key"],
            "video_name": row["video_name"],
            "best_f1": row["best_f1"],
            "best_label": row["best_label"],
            "source_plot": row["overview_plot"],
            "copied_plot": _to_rel(dst_plot),
            "copied_json": _to_rel(dst_json),
        })

    _save_json(os.path.join(dataset_best_dir, "summary.json"), summary)


def generate_all_results(resume=False):
    configs = _load_experiment_configs()
    scene_score_sources = {item["scene_score_source"] for item in configs}
    dataset_contexts = _build_dataset_contexts()
    score_assets = _load_score_assets(scene_score_sources)

    _prepare_output_dirs(resume=resume)

    manifest = {
        "num_experiment_files": len(configs),
        "datasets": {
            "summe": [],
            "tvsum": [],
        },
    }

    for dataset_name in ("summe", "tvsum"):
        context = dataset_contexts[dataset_name]
        dataset_dict = context["dataset"]
        mapping = context["mapping"]
        f1_reduction = context["f1_reduction"]

        for video_key in sorted(dataset_dict.keys(), key=lambda item: int(item.split("_")[1])):
            video_name = mapping[video_key]
            video_slug = _slugify(video_name)
            result_dir = os.path.join(RESULT_ROOT, dataset_name, video_slug)
            result_json_path = os.path.join(result_dir, "results.json")
            overview_plot_path = os.path.join(result_dir, "overview.png")

            if resume and os.path.exists(result_json_path) and os.path.exists(overview_plot_path):
                video_result = _load_json(result_json_path)
            else:
                video_result = _build_video_result(
                    dataset_name=dataset_name,
                    video_name=video_name,
                    video_data=dataset_dict[video_key],
                    configs=configs,
                    score_assets=score_assets,
                    f1_reduction=f1_reduction,
                )
                video_result["video_key"] = video_key
                _save_json(result_json_path, video_result)
                if video_result.get("best_result") is not None:
                    plot_overview(video_result, overview_plot_path)

            if "video_key" not in video_result:
                video_result["video_key"] = video_key
                _save_json(result_json_path, video_result)

            if not os.path.exists(overview_plot_path) and video_result.get("best_result") is not None:
                plot_overview(video_result, overview_plot_path)

            best_result = video_result.get("best_result") or {}
            manifest["datasets"][dataset_name].append({
                "video_key": video_key,
                "video_name": video_name,
                "result_json": _to_rel(result_json_path),
                "overview_plot": _to_rel(overview_plot_path),
                "best_f1": float(best_result.get("f1", 0.0)),
                "best_label": best_result.get("label"),
            })

    _save_json(os.path.join(RESULT_ROOT, "manifest.json"), manifest)

    for dataset_name in ("summe", "tvsum"):
        _write_best_dataset_summary(
            dataset_name, manifest["datasets"][dataset_name])

    return manifest


def _build_single_video_result(dataset_name, video_name, scene_score_source, alpha_scene_frame, smoothing_enabled):
    dataset_contexts = _build_dataset_contexts()
    dataset_dict = dataset_contexts[dataset_name]["dataset"]
    mapping = dataset_contexts[dataset_name]["mapping"]
    reverse_mapping = {value: key for key, value in mapping.items()}
    video_key = reverse_mapping.get(video_name) or reverse_mapping.get(
        video_name.replace("_", " "))
    if video_key is None:
        normalized_target = _normalize_video_name(video_name)
        for mapped_key, mapped_name in mapping.items():
            if _normalize_video_name(mapped_name) == normalized_target:
                video_key = mapped_key
                video_name = mapped_name
                break
    if video_key is None:
        raise KeyError(f"Video not found in {dataset_name}: {video_name}")

    config = {
        "path": "",
        "relative_source": f"manual/{scene_score_source}",
        "source_group": "manual",
        "scene_score_source": scene_score_source,
        "alpha_scene_frame": alpha_scene_frame,
        "temporal_smoothing": {
            "enabled": smoothing_enabled,
            "norm": "none",
        },
    }
    score_assets = _load_score_assets({scene_score_source})
    video_result = _build_video_result(
        dataset_name=dataset_name,
        video_name=mapping[video_key],
        video_data=dataset_dict[video_key],
        configs=[config],
        score_assets=score_assets,
        f1_reduction=dataset_contexts[dataset_name]["f1_reduction"],
    )
    video_result["video_key"] = video_key
    return video_result


def generate_pair_plots(summe_video, tvsum_video, scene_score_source="deepseek32", alpha_scene_frame=0.2, smoothing_enabled=False):
    summe_result = _build_single_video_result(
        dataset_name="summe",
        video_name=summe_video,
        scene_score_source=scene_score_source,
        alpha_scene_frame=alpha_scene_frame,
        smoothing_enabled=smoothing_enabled,
    )
    tvsum_result = _build_single_video_result(
        dataset_name="tvsum",
        video_name=tvsum_video,
        scene_score_source=scene_score_source,
        alpha_scene_frame=alpha_scene_frame,
        smoothing_enabled=smoothing_enabled,
    )

    plot_overview(summe_result, os.path.join(
        BASE, "figure", "score_curve_summe.pdf"))
    plot_overview(tvsum_result, os.path.join(
        BASE, "figure", "score_curve_tvsum.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "pair"], default="batch")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing per-video outputs when possible")
    parser.add_argument(
        "--summe_video", default="Excavators_river_crossing", help="SumMe video name")
    parser.add_argument("--tvsum_video", default="i3wAGJaaktw",
                        help="TVSum video name (YouTube id)")
    parser.add_argument("--scene_score_source", default="deepseek32",
                        choices=["gpt5", "deepseek32"], help="Scene-score source for pair mode")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Alpha coefficient for pair mode")
    parser.add_argument("--smooth", action="store_true",
                        help="Enable temporal smoothing in pair mode")
    args = parser.parse_args()

    if args.mode == "pair":
        generate_pair_plots(
            summe_video=args.summe_video,
            tvsum_video=args.tvsum_video,
            scene_score_source=args.scene_score_source,
            alpha_scene_frame=args.alpha,
            smoothing_enabled=args.smooth,
        )
        return

    manifest = generate_all_results(resume=args.resume)
    print(f"Saved manifest to {os.path.join(RESULT_ROOT, 'manifest.json')}")
    print(f"Processed {manifest['num_experiment_files']} experiment files")


if __name__ == "__main__":
    main()
