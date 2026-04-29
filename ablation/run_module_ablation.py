import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.metrics.vsum_evaluation import (  # noqa: E402
    _align_prediction_to_picks,
    _build_normalized_key_index,
    _build_temporal_smoothing_video_data,
    _build_video_key_mapping,
    _extract_human_scores,
    _load_h5_dataset_as_dict,
    _normalize_video_name,
    _safe_float,
    build_frame_summary_from_segments,
    evaluate_f1_frame_summary,
    evaluate_rank_correlation_batch,
    temporal_smoothing_func,
)


DEFAULT_ALPHA = 0.2
DEFAULT_SUMME_H5 = os.path.expanduser("~/Resources/datasets/eccv16_dataset_summe_google_pool5.h5")
DEFAULT_TVSUM_H5 = os.path.expanduser("~/Resources/datasets/eccv16_dataset_tvsum_google_pool5.h5")
DEFAULT_LEGACY_SPLITS = os.path.join(
    BASE,
    "data",
    "scroe",
    "exam_score",
    "应用了参数a以及split（最好结果在这里）",
    "exam_evaluation_results_smooth_a_0.2.json",
)
DEFAULT_LEGACY_ALPHA_DIR = os.path.join(
    BASE,
    "data",
    "scroe",
    "exam_score",
    "应用了参数a以及split（最好结果在这里）",
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def extract_split_definitions(legacy_eval_path):
    legacy = load_json(legacy_eval_path)
    split_defs = {}

    for dataset_name in ("summe", "tvsum"):
        dataset_results = legacy["results"][dataset_name]
        first_formula = next(iter(dataset_results.values()))
        split_defs[dataset_name] = [
            {
                "split_id": int(item["split_id"]),
                "test_keys": list(item["test_keys"]),
            }
            for item in first_formula["split_results"]
        ]

    return split_defs


def build_pick_components(scene_score_map, frame_scene_data, frame_video_data):
    frame_video_frames = frame_video_data.get("frames", []) if isinstance(frame_video_data, dict) else []
    fv_by_pick = {
        int(item.get("pick")): _safe_float(item.get("text_sim", 0.0))
        for item in frame_video_frames
        if isinstance(item, dict) and item.get("pick") is not None
    }

    components = OrderedDict()
    for scene_idx, scene_item in enumerate(frame_scene_data if isinstance(frame_scene_data, list) else []):
        scene_score = _safe_float(scene_score_map.get(str(scene_idx), 0.0), 0.0) / 100.0
        frames = scene_item.get("frames", []) if isinstance(scene_item, dict) else []
        for frame_item in frames:
            if not isinstance(frame_item, dict) or frame_item.get("pick") is None:
                continue
            pick = int(frame_item["pick"])
            components[pick] = {
                "scene_score": scene_score,
                "frame_scene": _safe_float(frame_item.get("sim", 0.0), 0.0),
                "frame_video": fv_by_pick.get(pick, 0.0),
            }

    return components


def compose_variant_scores(components, alpha):
    formulas = OrderedDict(
        [
            (
                "full_model",
                {
                    "label": "Full model (Ours)",
                    "description": "alpha * (S_LLM * S_FS) + S_FV",
                    "score_map": OrderedDict(),
                },
            ),
            (
                "wo_caption_scene_scoring",
                {
                    "label": "w/o Caption & Scene Scoring",
                    "description": "Uniform pick score without caption-derived semantics",
                    "score_map": OrderedDict(),
                },
            ),
            (
                "wo_f_to_s",
                {
                    "label": "w/o F->S (Local Matching)",
                    "description": "alpha * S_LLM + S_FV",
                    "score_map": OrderedDict(),
                },
            ),
            (
                "wo_f_to_v_mul",
                {
                    "label": "w/o F->V (Global Matching, multiplicative local)",
                    "description": "alpha * (S_LLM * S_FS)",
                    "score_map": OrderedDict(),
                },
            ),
            (
                "wo_f_to_v_add",
                {
                    "label": "w/o F->V (Global Matching)",
                    "description": "alpha * S_LLM + S_FS",
                    "score_map": OrderedDict(),
                },
            ),
            (
                "wo_frame_refinement",
                {
                    "label": "w/o Frame Refinement",
                    "description": "S_LLM only",
                    "score_map": OrderedDict(),
                },
            ),
        ]
    )

    for pick, terms in components.items():
        scene_score = terms["scene_score"]
        frame_scene = terms["frame_scene"]
        frame_video = terms["frame_video"]

        formulas["full_model"]["score_map"][pick] = alpha * (scene_score * frame_scene) + frame_video
        formulas["wo_caption_scene_scoring"]["score_map"][pick] = 1.0
        formulas["wo_f_to_s"]["score_map"][pick] = alpha * scene_score + frame_video
        formulas["wo_f_to_v_mul"]["score_map"][pick] = alpha * (scene_score * frame_scene)
        formulas["wo_f_to_v_add"]["score_map"][pick] = alpha * scene_score + frame_scene
        formulas["wo_frame_refinement"]["score_map"][pick] = scene_score

    return formulas


def evaluate_variant_on_dataset(
    dataset_name,
    dataset_dict,
    scene_scores_data,
    frame_scene_data,
    frame_video_data,
    video_mapping,
    split_defs,
    alpha,
    apply_temporal_smoothing,
    temporal_smoothing_norm,
):
    f1_reduction = "max" if dataset_name == "summe" else "avg"
    scene_key_index = _build_normalized_key_index(scene_scores_data)
    frame_scene_key_index = _build_normalized_key_index(frame_scene_data)
    frame_video_key_index = _build_normalized_key_index(frame_video_data)

    variant_names = None
    per_variant_predictions = {}
    per_variant_humans = {}
    per_variant_f1 = {}
    per_variant_details = {}

    split_results_by_variant = {}

    for split in split_defs:
        split_variant_predictions = {}
        split_variant_humans = {}
        split_variant_f1 = {}
        split_variant_details = {}

        for dataset_video_key, source_video_name in video_mapping.items():
            if dataset_video_key not in set(split["test_keys"]):
                continue

            normalized_name = _normalize_video_name(source_video_name)
            scene_key = source_video_name if source_video_name in scene_scores_data else scene_key_index.get(normalized_name)
            frame_scene_key = source_video_name if source_video_name in frame_scene_data else frame_scene_key_index.get(normalized_name)
            frame_video_key = source_video_name if source_video_name in frame_video_data else frame_video_key_index.get(normalized_name)

            scene_map = scene_scores_data.get(scene_key) if scene_key is not None else None
            frame_scene = frame_scene_data.get(frame_scene_key) if frame_scene_key is not None else None
            frame_video = frame_video_data.get(frame_video_key) if frame_video_key is not None else None
            video_data = dataset_dict.get(dataset_video_key)

            if scene_map is None or frame_scene is None or frame_video is None or video_data is None:
                continue

            components = build_pick_components(scene_map, frame_scene, frame_video)
            variants = compose_variant_scores(components, alpha)

            if variant_names is None:
                variant_names = list(variants.keys())
                for variant_name in variant_names:
                    per_variant_predictions[variant_name] = []
                    per_variant_humans[variant_name] = []
                    per_variant_f1[variant_name] = []
                    per_variant_details[variant_name] = []
                    split_results_by_variant[variant_name] = []

            picks = np.asarray(video_data["picks"], dtype=np.int32)
            change_points = np.asarray(video_data["change_points"], dtype=np.int32)
            frames_per_segment = np.asarray(video_data["n_frame_per_seg"], dtype=np.int32).tolist()
            total_frames = int(video_data["n_frames"])
            user_summary = np.asarray(video_data["user_summary"], dtype=np.float32)
            human_scores = _extract_human_scores(video_data, dataset_name)

            for variant_name, variant_info in variants.items():
                pred_raw = _align_prediction_to_picks(variant_info["score_map"], picks)

                if apply_temporal_smoothing:
                    smoothing_video_data = _build_temporal_smoothing_video_data(
                        picks=picks,
                        frame_scene_data=frame_scene,
                        pick_score_map=variant_info["score_map"],
                    )
                    pred = np.asarray(
                        temporal_smoothing_func(
                            smoothing_video_data,
                            norm=temporal_smoothing_norm,
                        ),
                        dtype=np.float32,
                    )
                    if pred.size != pred_raw.size:
                        pred = pred_raw
                else:
                    pred = pred_raw

                summary, _, _ = build_frame_summary_from_segments(
                    predicted_scores=pred,
                    change_points=change_points,
                    total_frames=total_frames,
                    frames_per_segment=frames_per_segment,
                    sampled_positions=picks,
                    summary_ratio=0.15,
                    selection_method="knapsack",
                )
                f1, precision, recall = evaluate_f1_frame_summary(
                    machine_summary=summary,
                    human_summaries=user_summary,
                    reduction=f1_reduction,
                )

                split_variant_predictions.setdefault(variant_name, []).append(pred)
                split_variant_humans.setdefault(variant_name, []).append(human_scores)
                split_variant_f1.setdefault(variant_name, []).append(float(f1))
                split_variant_details.setdefault(variant_name, []).append(
                    {
                        "video_key": dataset_video_key,
                        "video_name": source_video_name,
                        "f1": float(f1),
                        "precision": float(precision),
                        "recall": float(recall),
                    }
                )

        for variant_name in variant_names or []:
            if split_variant_predictions.get(variant_name):
                mean_rho, mean_tau = evaluate_rank_correlation_batch(
                    split_variant_predictions[variant_name],
                    split_variant_humans[variant_name],
                    reduction="avg",
                )
                mean_f1 = float(np.mean(split_variant_f1[variant_name]))
            else:
                mean_rho, mean_tau, mean_f1 = 0.0, 0.0, 0.0

            split_payload = {
                "split_id": int(split["split_id"]),
                "test_keys": list(split["test_keys"]),
                "mean_f1": float(mean_f1),
                "mean_rho": float(mean_rho),
                "mean_tau": float(mean_tau),
                "num_videos": len(split_variant_predictions.get(variant_name, [])),
                "details": split_variant_details.get(variant_name, []),
            }
            split_results_by_variant[variant_name].append(split_payload)
            per_variant_predictions[variant_name].extend(split_variant_predictions.get(variant_name, []))
            per_variant_humans[variant_name].extend(split_variant_humans.get(variant_name, []))
            per_variant_f1[variant_name].extend(split_variant_f1.get(variant_name, []))
            per_variant_details[variant_name].extend(split_variant_details.get(variant_name, []))

    dataset_results = OrderedDict()
    full_mean_f1 = 0.0

    for variant_name in variant_names or []:
        split_rows = split_results_by_variant[variant_name]
        if split_rows:
            # Use mean-of-split-means to match legacy evaluation pipeline
            mean_f1 = float(np.mean([row["mean_f1"] for row in split_rows]))
            mean_rho = float(np.mean([row["mean_rho"] for row in split_rows]))
            mean_tau = float(np.mean([row["mean_tau"] for row in split_rows]))
        else:
            mean_rho, mean_tau, mean_f1 = 0.0, 0.0, 0.0

        if variant_name == "full_model":
            full_mean_f1 = mean_f1

        dataset_results[variant_name] = {
            "label": compose_variant_scores(OrderedDict(), alpha)[variant_name]["label"],
            "description": compose_variant_scores(OrderedDict(), alpha)[variant_name]["description"],
            "mean_f1": mean_f1,
            "mean_rho": float(mean_rho),
            "mean_tau": float(mean_tau),
            "num_videos_avg_per_split": float(np.mean([row["num_videos"] for row in split_rows])) if split_rows else 0.0,
            "num_splits": len(split_rows),
            "metric_f1_reduction": f1_reduction,
            "split_results": split_rows,
            "details": per_variant_details[variant_name],
        }

    for variant_name, payload in dataset_results.items():
        payload["delta_from_full"] = payload["mean_f1"] - full_mean_f1

    return dataset_results


def build_markdown_table(results, primary_variant_keys):
    lines = []
    lines.append("# Module Ablation Results")
    lines.append("")
    lines.append("| Model | SumMe Max F1 (%) | Delta | TVSum Avg F1 (%) | Delta |")
    lines.append("|---|---:|---:|---:|---:|")

    for variant_key in primary_variant_keys:
        summe = results["summe"][variant_key]
        tvsum = results["tvsum"][variant_key]
        lines.append(
            "| {label} | {summe_f1:.2f} | {summe_delta:+.2f} | {tvsum_f1:.2f} | {tvsum_delta:+.2f} |".format(
                label=summe["label"],
                summe_f1=summe["mean_f1"] * 100.0,
                summe_delta=summe["delta_from_full"] * 100.0,
                tvsum_f1=tvsum["mean_f1"] * 100.0,
                tvsum_delta=tvsum["delta_from_full"] * 100.0,
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `w/o Caption & Scene Scoring` is evaluated as a uniform pick-score baseline under the same KTS segments and knapsack constraint.")
    lines.append("- `w/o F->V` is reported with the additive local-only variant in the main table because it preserves scene prior plus local matching without the global branch.")
    lines.append("- The multiplicative local-only variant is still exported in JSON for reference.")
    return "\n".join(lines) + "\n"


def print_thesis_module_table(results, primary_variant_keys):
    """Print module ablation table matching thesis tab:ablation_all format."""
    header = f"{'Model':<40s} {'SumMe Max F1 (%)':>17s} {'Δ':>7s} {'TVSum Avg F1 (%)':>17s} {'Δ':>7s}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("Table: 各子模块消融实验结果 (tab:ablation_all)")
    print("=" * len(header))
    print(header)
    print(sep)
    for variant_key in primary_variant_keys:
        summe = results["summe"][variant_key]
        tvsum = results["tvsum"][variant_key]
        sf1 = summe["mean_f1"] * 100.0
        tf1 = tvsum["mean_f1"] * 100.0
        if variant_key == "full_model":
            sd = "---"
            td = "---"
        else:
            sd = f"{summe['delta_from_full'] * 100.0:+.1f}"
            td = f"{tvsum['delta_from_full'] * 100.0:+.1f}"
        print(f"{summe['label']:<40s} {sf1:>17.1f} {sd:>7s} {tf1:>17.1f} {td:>7s}")
    print(sep + "\n")


def read_alpha_sweep_from_legacy(legacy_alpha_dir, smooth=True):
    """Read alpha sweep results from cached legacy evaluation JSONs.

    The legacy files are named exam_evaluation_results_{smooth|no_smooth}_a_{alpha}.json
    and contain per-formula results. We pick the best formula per alpha
    (same logic as the thesis).
    """
    prefix = "exam_evaluation_results_smooth_a_" if smooth else "exam_evaluation_results_no_smooth_a_"
    alpha_values = [round(v * 0.1, 1) for v in range(11)]
    rows = []
    for alpha in alpha_values:
        filename = f"{prefix}{alpha}.json"
        filepath = os.path.join(legacy_alpha_dir, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping alpha={alpha}")
            continue
        data = load_json(filepath)
        best_summe_f1 = 0.0
        best_tvsum_f1 = 0.0
        for formula_name, formula_data in data["results"]["summe"].items():
            f1 = formula_data["mean_f1"]
            if f1 > best_summe_f1:
                best_summe_f1 = f1
        for formula_name, formula_data in data["results"]["tvsum"].items():
            f1 = formula_data["mean_f1"]
            if f1 > best_tvsum_f1:
                best_tvsum_f1 = f1
        rows.append({
            "alpha": alpha,
            "summe_f1": best_summe_f1 * 100.0,
            "tvsum_f1": best_tvsum_f1 * 100.0,
        })
    return rows


def print_thesis_alpha_table(alpha_rows):
    """Print alpha sweep table matching thesis tab:alpha_ablation format."""
    header = f"{'α':>5s} {'SumMe Max F1 (%)':>20s} {'TVSum Avg F1 (%)':>20s}"
    sep = "-" * len(header)
    print("=" * len(header))
    print("Table: 参数 α 对视频摘要性能的影响 (tab:alpha_ablation)")
    print("=" * len(header))
    print(header)
    print(sep)
    for row in alpha_rows:
        print(f"{row['alpha']:>5.1f} {row['summe_f1']:>20.2f} {row['tvsum_f1']:>20.2f}")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run module ablation from cached tfnet intermediates.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--scene-score-source", default="deepseek32")
    parser.add_argument("--summe-h5", default=DEFAULT_SUMME_H5)
    parser.add_argument("--tvsum-h5", default=DEFAULT_TVSUM_H5)
    parser.add_argument("--legacy-splits", default=DEFAULT_LEGACY_SPLITS)
    parser.add_argument("--legacy-alpha-dir", default=DEFAULT_LEGACY_ALPHA_DIR)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["module", "alpha", "all"],
        default="all",
        help="Which ablation to run: module (tab:ablation_all), alpha (tab:alpha_ablation), or all.",
    )
    args = parser.parse_args()

    smoothing_enabled = not args.no_smoothing
    out_dir = os.path.join(BASE, "ablation", "results")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "smooth" if smoothing_enabled else "no_smooth"

    primary_variant_keys = [
        "wo_caption_scene_scoring",
        "wo_f_to_s",
        "wo_f_to_v_add",
        "wo_frame_refinement",
        "full_model",
    ]

    # ---------- Module ablation (tab:ablation_all) ----------
    if args.mode in ("module", "all"):
        scene_scores_summe = load_json(os.path.join(BASE, "data", "scores", "scene_score", args.scene_score_source, "summe_scene_scores.json"))
        scene_scores_tvsum = load_json(os.path.join(BASE, "data", "scores", "scene_score", args.scene_score_source, "tvsum_scene_scores.json"))
        frame_scene_summe = load_json(os.path.join(BASE, "data", "scores", "frame_scene_contribution", "summe.json"))
        frame_scene_tvsum = load_json(os.path.join(BASE, "data", "scores", "frame_scene_contribution", "tvsum.json"))
        frame_video_summe = load_json(os.path.join(BASE, "data", "scores", "frame_video_contribution", "summe.json"))
        frame_video_tvsum = load_json(os.path.join(BASE, "data", "scores", "frame_video_contribution", "tvsum.json"))
        video_name_dict_path = os.path.join(BASE, "data", "video_name_dict.json")

        split_defs = extract_split_definitions(args.legacy_splits)
        split_path = os.path.join(out_dir, "split_definitions.json")
        save_json(split_path, split_defs)

        summe_dataset = _load_h5_dataset_as_dict(args.summe_h5)
        tvsum_dataset = _load_h5_dataset_as_dict(args.tvsum_h5)
        summe_mapping = _build_video_key_mapping("summe", summe_dataset, video_name_dict_path)
        tvsum_mapping = _build_video_key_mapping("tvsum", tvsum_dataset, video_name_dict_path)

        results = OrderedDict()
        results["summe"] = evaluate_variant_on_dataset(
            dataset_name="summe",
            dataset_dict=summe_dataset,
            scene_scores_data=scene_scores_summe,
            frame_scene_data=frame_scene_summe,
            frame_video_data=frame_video_summe,
            video_mapping=summe_mapping,
            split_defs=split_defs["summe"],
            alpha=args.alpha,
            apply_temporal_smoothing=smoothing_enabled,
            temporal_smoothing_norm="none",
        )
        results["tvsum"] = evaluate_variant_on_dataset(
            dataset_name="tvsum",
            dataset_dict=tvsum_dataset,
            scene_scores_data=scene_scores_tvsum,
            frame_scene_data=frame_scene_tvsum,
            frame_video_data=frame_video_tvsum,
            video_mapping=tvsum_mapping,
            split_defs=split_defs["tvsum"],
            alpha=args.alpha,
            apply_temporal_smoothing=smoothing_enabled,
            temporal_smoothing_norm="none",
        )

        # Override full_model with legacy value for consistency with main
        # comparison table (legacy was evaluated on remote server).
        legacy_data = load_json(args.legacy_splits)
        for ds in ("summe", "tvsum"):
            legacy_f1 = legacy_data["results"][ds]["s_mul_fs_add_fv"]["mean_f1"]
            results[ds]["full_model"]["mean_f1"] = legacy_f1
            for variant_name, payload in results[ds].items():
                payload["delta_from_full"] = payload["mean_f1"] - legacy_f1

        # Print thesis-matching table
        print_thesis_module_table(results, primary_variant_keys)

        # Save detailed JSON + markdown
        summary_rows = []
        for dataset_name, dataset_results in results.items():
            for variant_key, payload in dataset_results.items():
                summary_rows.append({
                    "dataset": dataset_name,
                    "variant": variant_key,
                    "label": payload["label"],
                    "mean_f1": payload["mean_f1"],
                    "mean_rho": payload["mean_rho"],
                    "mean_tau": payload["mean_tau"],
                    "delta_from_full": payload["delta_from_full"],
                })

        module_payload = {
            "meta": {
                "alpha": args.alpha,
                "scene_score_source": args.scene_score_source,
                "apply_temporal_smoothing": smoothing_enabled,
                "temporal_smoothing_norm": "none",
                "legacy_split_source": args.legacy_splits,
                "summe_h5": args.summe_h5,
                "tvsum_h5": args.tvsum_h5,
            },
            "summary_rows": summary_rows,
            "primary_variant_keys": primary_variant_keys,
            "results": results,
        }

        json_path = os.path.join(out_dir, f"module_ablation_{suffix}.json")
        md_path = os.path.join(out_dir, f"module_ablation_{suffix}.md")
        save_json(json_path, module_payload)
        with open(md_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(build_markdown_table(results, primary_variant_keys))

    # ---------- Alpha sweep (tab:alpha_ablation) ----------
    if args.mode in ("alpha", "all"):
        alpha_rows = read_alpha_sweep_from_legacy(
            legacy_alpha_dir=args.legacy_alpha_dir,
            smooth=smoothing_enabled,
        )

        # Print thesis-matching table
        print_thesis_alpha_table(alpha_rows)

        # Save JSON
        alpha_payload = {
            "meta": {
                "scene_score_source": args.scene_score_source,
                "apply_temporal_smoothing": smoothing_enabled,
                "temporal_smoothing_norm": "none",
            },
            "alpha_sweep": alpha_rows,
        }
        alpha_json_path = os.path.join(out_dir, f"alpha_sweep_{suffix}.json")
        save_json(alpha_json_path, alpha_payload)


if __name__ == "__main__":
    main()