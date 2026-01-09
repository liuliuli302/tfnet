import os
import json
import h5py
import numpy as np
from evaluation_metrics import get_corr_coeff


def load_dataset(dataset_path):
    """Load HDF5 dataset into nested dict of numpy arrays."""
    dataset = {}
    with h5py.File(dataset_path, 'r') as f:
        for key in f.keys():
            dataset[key] = {}
            for sub_key in f[key].keys():
                dataset[key][sub_key] = f[key][sub_key][...]
    return dataset


def get_video_keys_sorted(dataset, scores_data, dataset_name):
    """
    Map dataset keys (video_1, ...) to actual video names for JSON alignment.
    """

    if dataset_name == 'tvsum':
        try:
            video_name_dict_path = '/root/tfnet/data/video_name_dict.json'
            with open(video_name_dict_path, 'r', encoding='utf-8') as f:
                video_name_dict = json.load(f)
            key_mapping = {}
            for video_name, video_id in video_name_dict.items():
                key_mapping[video_id] = video_name
            print(f"建立 {dataset_name} 数据集映射: {len(key_mapping)} 个视频 (使用video_name_dict.json)")
            return key_mapping
        except Exception as e:
            print(f"警告: 无法读取video_name_dict.json: {e}, 回退到原始方法")

    if dataset_name == 'summe':
        video_dir = '/root/autodl-tmp/datasets/SumMe/videos'
    elif dataset_name == 'tvsum':
        video_dir = '/root/autodl-tmp/datasets/TVSum/videos'
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

    try:
        video_files = [f[:-4] for f in os.listdir(video_dir) if f.endswith('.mp4')]
        video_files.sort()
        if dataset_name == 'tvsum':
            video_files.reverse()
    except Exception as e:
        print(f"警告: 无法读取视频目录 {video_dir}: {e}")
        video_files = sorted(scores_data.keys())
        if dataset_name == 'tvsum':
            video_files.reverse()

    dataset_keys = list(dataset.keys())
    dataset_keys.sort(key=lambda x: int(x.split('_')[1]))

    key_mapping = {}
    for idx, dataset_key in enumerate(dataset_keys):
        if idx < len(video_files):
            key_mapping[dataset_key] = video_files[idx]

    print(f"建立 {dataset_name} 数据集映射: {len(key_mapping)} 个视频")
    return key_mapping


def extract_user_scores_for_video(d, dataset_name):
    """
    Extract per-frame user importance scores for a single video.
    """

    if dataset_name == 'tvsum':
        if 'user_scores' in d:
            us = d['user_scores']
            return [us[i, :] for i in range(us.shape[0])]
        elif 'user_summary' in d:
            us = d['user_summary']
            return [us[i, :] for i in range(us.shape[0])]
        else:
            raise ValueError("TVSum 缺少 user_scores 或 user_summary")

    elif dataset_name == 'summe':
        if 'user_scores' in d:
            return d['user_scores']
        elif 'user_summary' in d:
            return d['user_summary']
        else:
            raise ValueError("SumMe 缺少 user_scores 或 user_summary")

    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")


def evaluate_json_with_corr(json_filename, dataset, dataset_name, scores_data):
    """
    Compute mean Spearman and Kendall correlation for one JSON file over one dataset.
    """

    key_mapping = get_video_keys_sorted(dataset, scores_data, dataset_name)

    pred_imp_scores = []
    videos = []
    detailed = []
    user_scores_collection = []

    # ----------------------------
    # 遍历每个视频
    # ----------------------------
    for video_key, original_key in key_mapping.items():
        if original_key not in scores_data:
            detailed.append({'video_key': video_key, 'original_key': original_key, 'status': 'missing_scores'})
            continue

        video_data = scores_data[original_key]
        if not isinstance(video_data, list):
            detailed.append({'video_key': video_key, 'original_key': original_key, 'status': 'bad_format'})
            continue

        d = dataset[video_key]
        positions = d['picks']
        X = np.array(video_data)

        if len(X) != len(positions):
            detailed.append({'video_key': video_key, 'original_key': original_key,
                             'status': f'len_mismatch({len(X)} vs {len(positions)})'})
            continue

        X = np.squeeze(X).astype(np.float32)
        pred_imp_scores.append(X)
        videos.append(video_key)

        detailed.append({'video_key': video_key, 'original_key': original_key, 'status': 'ok', 'n_frames': len(X)})

        # --------------------------------------------------------
        # user_scores 对齐
        # --------------------------------------------------------
        try:
            us = extract_user_scores_for_video(d, dataset_name)
        except Exception:
            user_scores_collection.append(None)
            continue

        picks_len = len(positions)

        def align(arr):
            arr = np.array(arr)
            frame_len = arr.shape[-1]

            if frame_len == picks_len:
                return arr.astype(np.float32)

            if frame_len > picks_len:
                return arr[..., :picks_len].astype(np.float32)

            pad_width = picks_len - frame_len
            if arr.ndim == 1:
                return np.pad(arr.astype(np.float32), (0, pad_width), mode='edge')
            else:
                return np.pad(arr.astype(np.float32), ((0, 0), (0, pad_width)), mode='edge')

        if dataset_name == 'tvsum':
            aligned = [align(us_i) for us_i in us]
            user_scores_collection.append(aligned)
        else:
            aligned = align(us)
            user_scores_collection.append(aligned)

    # 过滤掉 user_scores 不存在的视频
    filtered_pred = []
    filtered_users = []

    for p, u in zip(pred_imp_scores, user_scores_collection):
        if u is None:
            continue
        filtered_pred.append(p)
        filtered_users.append(u)

    if not filtered_pred:
        return {
            'rho_mean': 0.0,
            'tau_mean': 0.0,
            'num_videos': 0,
            'details': detailed,
        }

    # ----------------------------
    # compute correlation
    # ----------------------------
    rho, tau = get_corr_coeff(
        filtered_pred,
        videos,
        'SumMe' if dataset_name == 'summe' else 'TVSum',
        filtered_users
    )

    return {
        'rho_mean': float(rho),
        'tau_mean': float(tau),
        'num_videos': len(filtered_pred),
        'details': detailed,
    }


def print_top_results(dataset_name, results, top_k=5):
    print(f"\n{'=' * 80}")
    print(f"{dataset_name.upper()} 相关性评估 - 前{top_k}个最好结果")
    print(f"{'=' * 80}")

    sorted_results = sorted(results, key=lambda x: (x['rho_mean'], x['tau_mean']), reverse=True)

    for i, result in enumerate(sorted_results[:top_k]):
        print(f"\n第{i + 1}名:")
        print(f"  文件名: {result['filename']}")
        print(f"  Spearman(ρ): {result['rho_mean']:.4f}")
        print(f"  Kendall(τ): {result['tau_mean']:.4f}")
        print(f"  视频数量: {result['num_videos']}")


def save_top_results(output_dir, summe_results, tvsum_results, top_k=5):
    summe_sorted = sorted(summe_results, key=lambda x: (x['rho_mean'], x['tau_mean']), reverse=True)
    tvsum_sorted = sorted(tvsum_results, key=lambda x: (x['rho_mean'], x['tau_mean']), reverse=True)

    top_results = {
        'summe_top_results': summe_sorted[:top_k],
        'tvsum_top_results': tvsum_sorted[:top_k],
        'summary': {
            'summe': {
                'best_file': summe_sorted[0]['filename'] if summe_sorted else 'N/A',
                'best_rho': summe_sorted[0]['rho_mean'] if summe_sorted else 0,
                'best_tau': summe_sorted[0]['tau_mean'] if summe_sorted else 0,
                'total_experiments': len(summe_results),
            },
            'tvsum': {
                'best_file': tvsum_sorted[0]['filename'] if tvsum_sorted else 'N/A',
                'best_rho': tvsum_sorted[0]['rho_mean'] if tvsum_sorted else 0,
                'best_tau': tvsum_sorted[0]['tau_mean'] if tvsum_sorted else 0,
                'total_experiments': len(tvsum_results),
            }
        }
    }

    output_file = os.path.join(output_dir, "top_results_method_2.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(top_results, f, indent=2, ensure_ascii=False, default=str)
    return output_file


def main():
    results_dir = "/root/tfnet/out/exam01/final"
    summe_path = "/root/autodl-tmp/datasets/SumMe/summe.h5"
    tvsum_path = "/root/autodl-tmp/datasets/TVSum/tvsum.h5"

    if not os.path.exists(results_dir):
        print(f"错误: 结果目录不存在: {results_dir}")
        return
    if not os.path.exists(summe_path):
        print(f"错误: SumMe 数据集不存在: {summe_path}")
        return
    if not os.path.exists(tvsum_path):
        print(f"错误: TVSum 数据集不存在: {tvsum_path}")
        return

    print("开始使用相关性系数评估...")

    summe_dataset = load_dataset(summe_path)
    tvsum_dataset = load_dataset(tvsum_path)

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')
                  and not f.startswith('evaluation_') and not f.startswith('top_')]

    print(f"找到 {len(json_files)} 个JSON文件进行评估")

    summe_results = []
    tvsum_results = []
    all_results = {}

    for filename in json_files:
        json_path = os.path.join(results_dir, filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        except Exception as e:
            print(f"错误: 无法加载文件 {filename}: {e}")
            continue

        scores_data = result_data.get('scores', {})
        suffix = filename.split(".")[0].split("_")[-1].lower()

        # ------------------- SumMe -------------------
        if 'summe' in scores_data:
            summe_eval = evaluate_json_with_corr(filename, summe_dataset, 'summe', scores_data['summe'])
            summe_results.append({
                'filename': filename,
                'rho_mean': summe_eval['rho_mean'],
                'tau_mean': summe_eval['tau_mean'],
                'num_videos': summe_eval['num_videos'],
                'details': summe_eval['details'],
            })
            all_results[f"{filename}_summe_method_2"] = summe_eval
            print(f"summe - Spearman: {summe_eval['rho_mean']:.4f}, Kendall: {summe_eval['tau_mean']:.4f} ({summe_eval['num_videos']} 个视频)")
        elif suffix == 'summe':
            print(f"提示: 文件 {filename} 指示 summe，但未提供 'scores.summe' 字段")

        # ------------------- TVSum -------------------
        if 'tvsum' in scores_data:
            tvsum_eval = evaluate_json_with_corr(filename, tvsum_dataset, 'tvsum', scores_data['tvsum'])
            tvsum_results.append({
                'filename': filename,
                'rho_mean': tvsum_eval['rho_mean'],
                'tau_mean': tvsum_eval['tau_mean'],
                'num_videos': tvsum_eval['num_videos'],
                'details': tvsum_eval['details'],
            })
            all_results[f"{filename}_tvsum_method_2"] = tvsum_eval
            print(f"tvsum - Spearman: {tvsum_eval['rho_mean']:.4f}, Kendall: {tvsum_eval['tau_mean']:.4f} ({tvsum_eval['num_videos']} 个视频)")
        elif suffix == 'tvsum':
            print(f"提示: 文件 {filename} 指示 tvsum，但未提供 'scores.tvsum' 字段")

    print_top_results("SumMe", summe_results, top_k=5)
    print_top_results("TVSum", tvsum_results, top_k=5)

    top_results_file = save_top_results(results_dir, summe_results, tvsum_results, top_k=5)

    output_file = os.path.join(results_dir, "evaluation_results_method_2.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n详细相关性评估结果已保存到: {output_file}")
    print(f"前5名结果已保存到: {top_results_file}")


if __name__ == "__main__":
    main()
