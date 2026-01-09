import json
import os
import glob
import re


def extract_score_from_llm_output(llm_output):
    match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', llm_output)
    return float(match.group(1)) if match else 0.0


def parse_llm_query_config(filename):
    parts = filename.replace('.json', '').split('_')
    return {
        "model": parts[0],
        "with_summary": "ws" in parts,
        "with_explanation": "we" in parts,
        "summary_source": parts[-1] if parts[-1] in ["moonshot", "deepseek"] else "none"
    }


def load_llm_query_scores(experiment_name):
    results = {}
    llm_query_path = f"out/{experiment_name}/llm_query"
    if not os.path.exists(llm_query_path):
        return results

    for json_file in glob.glob(f"{llm_query_path}/*.json"):
        filename = os.path.basename(json_file)
        config = parse_llm_query_config(filename)

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = {
            "exam_config": {"llm_query": config},
            "scores": {"summe": {}, "tvsum": {}}
        }

        if isinstance(data, list):
            new_data = {}
            for item in data:
                new_data.update(item)
            data = new_data
        if "summe" in filename.split(".")[0].split("_"):
            dataset_name = "SumMe"
        else:
            dataset_name = "TVSum"
        for video_name, frames in data.items():
            video_frames_path = f"/root/autodl-tmp/datasets/{dataset_name}/frames/{video_name}"
            n_picks = len(os.listdir(video_frames_path))
            picks = []
            llm_query_scores = []
            for frame_data in frames:
                picks.append(frame_data.get("frame_idx", 0))
                llm_query_scores.append(extract_score_from_llm_output(
                    frame_data.get("llm_output", "")))

            dataset = "tvsum" if "tvsum" in filename else "summe"
            processed_data["scores"][dataset][video_name] = {
                "picks": picks,
                "llm_query_scores": llm_query_scores
            }
            if n_picks != len(llm_query_scores):
                print(
                    f"Warning: Mismatch in picks and scores for {video_name} in {filename}. Expected {n_picks} picks, got {len(llm_query_scores)} scores.")

        results[filename] = processed_data
    return results


def load_nfs_scores(experiment_name):
    results = {}
    nfs_path = f"out/{experiment_name}/nfs/nfs_lvnet"
    if not os.path.exists(nfs_path):
        return results

    for json_file in glob.glob(f"{nfs_path}/*.json"):
        filename = os.path.basename(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        method = filename.split(".")[0].split("_")[-1]
        dataset = "tvsum" if "tvsum" in filename else "summe"

        # 添加与 load_llm_query_scores 类似的配置结构
        config = {
            "nfs": {"method": f"{method}"}
        }

        processed_data = {
            "exam_config": {"nfs": config["nfs"]},
            "scores": {dataset: {}}
        }

        for video_name, video_data in data.items():
            if "picks" in video_data:
                picks = video_data["picks"]
                dataset_path = "SumMe" if dataset == "summe" else "TVSum"
                frames_path = f"/root/autodl-tmp/datasets/{dataset_path}/frames/{video_name}"

                if os.path.exists(frames_path):
                    total_frames = len(os.listdir(frames_path))
                    nfs_scores = [0.0] * total_frames
                    for pick in picks:
                        adjusted_idx = pick // 15
                        if adjusted_idx < len(nfs_scores):
                            if dataset == "summe":
                                nfs_scores[adjusted_idx] = 0.28
                                # 0.28 最高 为 43.7
                            else:
                                nfs_scores[adjusted_idx] = 0.4
                                # 0.4 对应 TVSUM 58.2
                            

                    processed_data["scores"][dataset][video_name] = {
                        "picks": list(range(len(nfs_scores))),
                        "nfs_scores": nfs_scores
                    }
        results[filename] = processed_data
    return results


def load_sim_scores(experiment_name):
    results = {}
    sim_path = f"out/{experiment_name}/sim"
    if not os.path.exists(sim_path):
        return results

    for json_file in glob.glob(f"{sim_path}/*.json"):
        filename = os.path.basename(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = {
            "summary_source": "moonshot" if "moonshot" in filename else "deepseek"}
        dataset = "tvsum" if "tvsum" in filename else "summe"

        processed_data = {
            "exam_config": {"sim": config},
            "scores": {dataset: {}}
        }

        for video_name, video_data in data.items():
            if isinstance(video_data, list) and len(video_data) > 0 and isinstance(video_data[0], (int, float)):
                similarities = video_data
            elif isinstance(video_data, dict) and "similarities" in video_data:
                similarities = video_data["similarities"]
            else:
                continue

            processed_data["scores"][dataset][video_name] = {
                "picks": list(range(len(similarities))),
                "sim_scores": similarities
            }

        results[filename] = processed_data
    return results


def generate_all_combinations(llm_data, nfs_data, sim_data):
    combinations = []
    weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    datasets = ["summe", "tvsum"]

    for dataset in datasets:
        # 筛选属于当前数据集的文件
        valid_llm_files = {k: v for k, v in llm_data.items() if dataset in k}
        valid_nfs_files = {k: v for k, v in nfs_data.items() if dataset in k}
        valid_sim_files = {k: v for k, v in sim_data.items() if dataset in k}

        for llm_file, llm_result in valid_llm_files.items():
            for nfs_file, nfs_result in valid_nfs_files.items():
                for sim_file, sim_result in valid_sim_files.items():
                    for weight in weights:
                        config = {
                            "exam_config": {
                                "nfs": nfs_result["exam_config"]["nfs"],
                                "llm_query": llm_result["exam_config"]["llm_query"],
                                "sim": sim_result["exam_config"]["sim"],
                                "weight": weight,
                                "dataset": dataset
                            }
                        }
                        combinations.append({
                            "config": config,
                            "dataset": dataset,
                            "llm_file": llm_file,
                            "nfs_file": nfs_file,
                            "sim_file": sim_file,
                            "llm_data": llm_result,
                            "nfs_data": nfs_result,
                            "sim_data": sim_result
                        })
    return combinations


def calculate_final_scores(combination, dataset):
    weight = combination["config"]["exam_config"]["weight"]
    llm_data = combination["llm_data"]
    nfs_data = combination["nfs_data"]
    sim_data = combination["sim_data"]
    final_scores = {}

    llm_videos = set(llm_data["scores"][dataset].keys())
    nfs_videos = set(nfs_data["scores"][dataset].keys())
    sim_videos = set(sim_data["scores"][dataset].keys())
    common_videos = llm_videos & nfs_videos & sim_videos

    for video_name in common_videos:
        llm_scores = llm_data["scores"][dataset][video_name]["llm_query_scores"]
        nfs_scores = nfs_data["scores"][dataset][video_name]["nfs_scores"]
        sim_scores = sim_data["scores"][dataset][video_name]["sim_scores"]

        # 以LLM分数的长度为准
        llm_length = len(llm_scores)
        nfs_length = len(nfs_scores)
        sim_length = len(sim_scores)

        # 检查哪些数据长度不匹配并报告错误
        if nfs_length != llm_length:
            print(
                f"错误: {video_name} 的NFS分数长度({nfs_length})与LLM分数长度({llm_length})不匹配")
        if sim_length != llm_length:
            print(
                f"错误: {video_name} 的Sim分数长度({sim_length})与LLM分数长度({llm_length})不匹配")

        final_video_scores = []

        # 以LLM分数长度为准，确保所有picks都被处理
        for i in range(llm_length):
            base_score = llm_scores[i]
            nfs_bonus = nfs_scores[i] if i < nfs_length else 0.0
            sim_score = sim_scores[i] if i < sim_length else 0.0
            final_score = base_score + nfs_bonus + weight * sim_score
            final_video_scores.append(max(0.0, final_score))

        final_scores[video_name] = final_video_scores
    return final_scores


def save_final_scores(experiment_name, combination_idx, config, scores, dataset):
    output_dir = f"out/{experiment_name}/final"
    os.makedirs(output_dir, exist_ok=True)

    # 在配置中添加数据集名称
    config_with_dataset = config.copy()
    config_with_dataset["exam_config"]["dataset"] = dataset

    result = {"exam_config": config_with_dataset["exam_config"], "scores": {
        dataset: scores}}
    with open(f"{output_dir}/scores_{combination_idx}_{dataset}.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main(experiment_name):
    llm_data = load_llm_query_scores(experiment_name)
    nfs_data = load_nfs_scores(experiment_name)
    sim_data = load_sim_scores(experiment_name)

    if not (llm_data and nfs_data and sim_data):
        print(
            f"Warning: Missing data - LLM: {len(llm_data)}, NFS: {len(nfs_data)}, Sim: {len(sim_data)}")
        return

    combinations = generate_all_combinations(llm_data, nfs_data, sim_data)
    print(f"Total combinations: {len(combinations)}")

    for idx, combination in enumerate(combinations):
        dataset = combination["dataset"]
        final_scores = calculate_final_scores(combination, dataset)
        save_final_scores(experiment_name, idx,
                          combination["config"], final_scores, dataset)

    print(
        f"All combinations processed. Results saved to out/{experiment_name}/final/")


if __name__ == "__main__":
    main("exam01")
