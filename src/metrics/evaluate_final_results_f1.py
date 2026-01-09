import os
import json
import h5py
import numpy as np
from src.util.eval import generate_summary, evaluate_summary


def load_dataset(dataset_path):
    """加载数据集"""
    dataset = {}
    with h5py.File(dataset_path, 'r') as f:
        for key in f.keys():
            dataset[key] = {}
            for sub_key in f[key].keys():
                dataset[key][sub_key] = f[key][sub_key][...]
    return dataset


def get_video_keys_sorted(dataset, scores_data, dataset_name):
    """获取正确的视频键映射，基于实际的视频文件名顺序"""

    # 如果是tvsum数据集，使用video_name_dict.json文件
    if dataset_name == 'tvsum':
        try:
            video_name_dict_path = '/root/tfnet/data/video_name_dict.json'
            with open(video_name_dict_path, 'r', encoding='utf-8') as f:
                video_name_dict = json.load(f)

            # 反向映射：从video_id到video_name
            key_mapping = {}
            for video_name, video_id in video_name_dict.items():
                key_mapping[video_id] = video_name

            print(
                f"建立 {dataset_name} 数据集映射: {len(key_mapping)} 个视频 (使用video_name_dict.json)")
            return key_mapping

        except Exception as e:
            print(f"警告: 无法读取video_name_dict.json: {e}, 回退到原始方法")

    # summe数据集或tvsum回退方案
    # 根据数据集名称确定视频文件目录
    if dataset_name == 'summe':
        video_dir = '/root/autodl-tmp/datasets/SumMe/videos'
    elif dataset_name == 'tvsum':
        video_dir = '/root/autodl-tmp/datasets/TVSum/videos'
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

    # 获取视频文件名列表（去除.mp4扩展名）并排序
    import os
    try:
        video_files = [f[:-4]
                       for f in os.listdir(video_dir) if f.endswith('.mp4')]
        video_files.sort()  # 按字母序排序

        # 如果是tvsum数据集，则倒序
        if dataset_name == 'tvsum':
            video_files.reverse()

    except Exception as e:
        print(f"警告: 无法读取视频目录 {video_dir}: {e}")
        # 如果无法读取视频目录，回退到使用scores数据的键
        video_files = sorted(scores_data.keys())
        if dataset_name == 'tvsum':
            video_files.reverse()

    # 数据集中的键格式为 video_1, video_2, ..., video_N
    # 按数字排序获取正确的顺序
    dataset_keys = list(dataset.keys())
    dataset_keys.sort(key=lambda x: int(x.split('_')[1]))  # 按video_后的数字排序

    # 建立从数据集键到实际视频名称的映射
    key_mapping = {}
    for idx, dataset_key in enumerate(dataset_keys):
        if idx < len(video_files):
            key_mapping[dataset_key] = video_files[idx]

    print(f"建立 {dataset_name} 数据集映射: {len(key_mapping)} 个视频")

    return key_mapping


def evaluate_final_results(results_dir, summe_path, tvsum_path):
    """
    评估 final 目录下的所有得分 JSON 文件

    Args:
        results_dir: 结果目录路径 (out/exam01/final)
        summe_path: SumMe 数据集路径
        tvsum_path: TVSum 数据集路径
    """
    # 加载数据集
    summe_dataset = load_dataset(summe_path)
    tvsum_dataset = load_dataset(tvsum_path)

    # 存储所有结果
    all_results = {}

    # 分别存储每个数据集的结果，用于排序
    summe_results = []
    tvsum_results = []

    # 遍历结果目录中的所有 JSON 文件
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')
                  and not f.startswith('evaluation_') and not f.startswith('top_')]
    print(f"找到 {len(json_files)} 个JSON文件进行评估")

    for filename in json_files:
        json_path = os.path.join(results_dir, filename)

        # 加载 JSON 结果
        try:
            with open(json_path, 'r') as f:
                result_data = json.load(f)
        except Exception as e:
            print(f"错误: 无法加载文件 {filename}: {e}")
            continue

        # print(f"\n评估文件: {filename}")

        # 从JSON中提取得分数据
        scores_data = result_data.get('scores', {})
        dataset_name = filename.split(".")[0].split("_")[-1]
        # 分别评估 SumMe 和 TVSum
        if dataset_name == "summe":
            dataset_name, dataset, metric = ('summe', summe_dataset, 'max')
        else:
            dataset_name, dataset, metric = ('tvsum', tvsum_dataset, 'avg')
        # for dataset_name, dataset, metric in [
        #     ('summe', summe_dataset, 'max'),
        #     ('tvsum', tvsum_dataset, 'avg')
        # ]:

        dataset_scores = scores_data[dataset_name]

        # 获取视频键映射（传入数据集名称以获取正确的视频名称）
        key_mapping = get_video_keys_sorted(
            dataset, dataset_scores, dataset_name)

        fms = []
        video_scores = []

        # 遍历该数据集的所有视频
        for video_key, original_key in key_mapping.items():
            if original_key not in dataset_scores:
                print(f"警告: {original_key} 不在结果文件中")
                continue

            # 获取机器生成的激活分数
            video_data = dataset_scores[original_key]
            if isinstance(video_data, list):
                X = np.array(video_data)
            else:
                print(f"错误: {original_key} 的数据格式不正确")
                continue

            # 获取数据集中的信息
            d = dataset[video_key]

            if 'change_points' not in d:
                print(f"错误: 数据集/视频 {video_key} 中没有变化点")
                continue

            cps = d['change_points']
            num_frames = d['n_frames'] if isinstance(
                d['n_frames'], int) else d['n_frames'][()]
            nfps = d['n_frame_per_seg'].tolist() if hasattr(
                d['n_frame_per_seg'], 'tolist') else d['n_frame_per_seg']
            positions = d['picks']
            user_summary = d['user_summary']

            # 检查数据长度一致性
            if len(X) != len(positions):
                print(
                    f"警告: {original_key} 的分数数组长度({len(X)})与positions长度({len(positions)})不匹配，跳过该视频")
                continue

            # 归一化分数到[0,1]
            if np.max(X) == np.min(X):
                probs = np.zeros_like(X)
            else:
                probs = (X - np.min(X)) / (np.max(X) - np.min(X))

            # 生成机器摘要
            machine_summary, _, _ = generate_summary(
                probs, cps, num_frames, nfps, positions)

            # 评估摘要
            fm, _, _ = evaluate_summary(
                machine_summary, user_summary, metric)

            fms.append(fm)
            video_scores.append(
                [len(video_scores) + 1, video_key, original_key, f"{fm:.1%}"])

        # 计算平均 F-measure
        mean_fm = np.mean(fms) if fms else 0.0

        # 存储结果
        result_key = f"{filename}_{dataset_name}"
        all_results[result_key] = {
            'filename': filename,
            'dataset': dataset_name,
            'metric': metric,
            'mean_f_measure': mean_fm,
            'video_scores': video_scores,
            'num_videos': len(fms)
        }

        # 添加到对应数据集的结果列表中
        result_entry = {
            'filename': filename,
            'f1_score': mean_fm,
            'num_videos': len(fms),
            'detailed_scores': video_scores
        }

        if dataset_name == 'summe':
            summe_results.append(result_entry)
        else:  # tvsum
            tvsum_results.append(result_entry)

        print(f"{dataset_name} - 平均 F-measure: {mean_fm:.1%} ({len(fms)} 个视频)")

    return all_results, summe_results, tvsum_results


def print_top_results(dataset_name, results, top_k=5):
    """打印每个数据集的前K个最好结果"""
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} 数据集 - 前{top_k}个最好结果:")
    print(f"{'='*80}")

    # 按F1分数排序
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)

    for i, result in enumerate(sorted_results[:top_k]):
        print(f"\n第{i+1}名:")
        print(f"  文件名: {result['filename']}")
        print(f"  F1分数: {result['f1_score']:.1%}")
        print(f"  视频数量: {result['num_videos']}")

        # 显示前几个视频的详细分数
        if result['detailed_scores']:
            print("  详细分数 (前5个视频):")
            for score in result['detailed_scores'][:5]:
                print(f"    {score[1]} ({score[2]}): {score[3]}")
            if len(result['detailed_scores']) > 5:
                print(f"    ... 还有 {len(result['detailed_scores']) - 5} 个视频")


def save_top_results(output_dir, summe_results, tvsum_results, top_k=5):
    """保存前K个最好结果到文件"""
    # 按F1分数排序
    summe_sorted = sorted(
        summe_results, key=lambda x: x['f1_score'], reverse=True)
    tvsum_sorted = sorted(
        tvsum_results, key=lambda x: x['f1_score'], reverse=True)

    top_results = {
        'summe_top_results': summe_sorted[:top_k],
        'tvsum_top_results': tvsum_sorted[:top_k],
        'summary': {
            'summe': {
                'best_f1': summe_sorted[0]['f1_score'] if summe_sorted else 0,
                'best_file': summe_sorted[0]['filename'] if summe_sorted else 'N/A',
                'total_experiments': len(summe_results)
            },
            'tvsum': {
                'best_f1': tvsum_sorted[0]['f1_score'] if tvsum_sorted else 0,
                'best_file': tvsum_sorted[0]['filename'] if tvsum_sorted else 'N/A',
                'total_experiments': len(tvsum_results)
            }
        }
    }

    # 保存到文件
    output_file = os.path.join(output_dir, "top_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(top_results, f, indent=2, ensure_ascii=False, default=str)

    return output_file


def main():
    """主函数"""
    results_dir = "/root/tfnet/out/exam01/final"
    summe_path = "/root/autodl-tmp/datasets/SumMe/summe.h5"
    tvsum_path = "/root/autodl-tmp/datasets/TVSum/tvsum.h5"

    # 检查路径是否存在
    if not os.path.exists(results_dir):
        print(f"错误: 结果目录不存在: {results_dir}")
        return

    if not os.path.exists(summe_path):
        print(f"错误: SumMe 数据集不存在: {summe_path}")
        return

    if not os.path.exists(tvsum_path):
        print(f"错误: TVSum 数据集不存在: {tvsum_path}")
        return

    print("开始评估...")

    # 执行评估
    try:
        all_results, summe_results, tvsum_results = evaluate_final_results(
            results_dir, summe_path, tvsum_path)
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"评估完成！")
    print(f"SumMe结果数量: {len(summe_results)}")
    print(f"TVSum结果数量: {len(tvsum_results)}")

    # 打印前5个最好结果
    print_top_results("SumMe", summe_results, top_k=5)
    print_top_results("TVSum", tvsum_results, top_k=5)

    # 保存结果
    top_results_file = save_top_results(
        results_dir, summe_results, tvsum_results, top_k=5)

    # 生成总结报告
    print(f"\n{'='*80}")
    print("总结报告")
    print(f"{'='*80}")

    print(f"总共评估了 {len(summe_results)} 个SumMe实验和 {len(tvsum_results)} 个TVSum实验")

    if summe_results:
        best_summe = max(summe_results, key=lambda x: x['f1_score'])
        print(
            f"SumMe最佳结果: {best_summe['f1_score']:.1%} (文件: {best_summe['filename']})")

    if tvsum_results:
        best_tvsum = max(tvsum_results, key=lambda x: x['f1_score'])
        print(
            f"TVSum最佳结果: {best_tvsum['f1_score']:.1%} (文件: {best_tvsum['filename']})")

    # 保存详细结果到文件
    output_file = os.path.join(results_dir, "evaluation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n详细结果已保存到: {output_file}")
    print(f"前5名结果已保存到: {top_results_file}")


if __name__ == "__main__":
    main()
