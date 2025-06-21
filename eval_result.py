#!/usr/bin/env python3
# filepath: /root/tfnet/find_best_weights.py
import json
import numpy as np
from eval.evaluator import Evaluator
from pathlib import Path

def create_evaluator(dataset_name, dataset_path):
    """创建评估器"""
    if 'experiment_cache/tfnet' in dataset_path:
        dataset_path = dataset_path.replace('/experiment_cache/tfnet', '/data')
    return Evaluator(dataset_name=dataset_name, data_dir=str(Path(dataset_path).parent), 
                    eval_method="max" if dataset_name == "SumMe" else "avg")

def extract_scores(scores_data):
    """从分数数据中提取数值数组"""
    if isinstance(scores_data, dict):
        # 如果是字典，尝试提取 'values' 字段
        if 'values' in scores_data:
            values = scores_data['values']
            if isinstance(values, list):
                return np.array(values)
            else:
                # 如果 values 不是列表，可能只有一个值，需要重复到所需长度
                return np.array([values])
        else:
            # 如果没有 'values' 字段，尝试提取所有数值
            numeric_values = []
            for key, value in scores_data.items():
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
            return np.array(numeric_values) if numeric_values else np.array([0.0])
    else:
        # 如果是列表，直接转换
        return np.array(scores_data)

def evaluate_weighted_scores(step1_data, step2_data, weight_a, segment_type="seg_2"):
    """评估加权分数"""
    results = {}
    
    for dataset_name in step1_data['datasets']:
        dataset1 = step1_data['datasets'][dataset_name]
        dataset2 = step2_data['datasets'][dataset_name]
        
        evaluator = create_evaluator(dataset_name, dataset1['dataset_path'])
        pred_scores = {}
        
        for video_name in dataset1['videos']:
            if video_name in dataset2 and video_name in evaluator.dataset.video_name_dict:
                # 提取分数
                scores1 = extract_scores(dataset1['videos'][video_name]['scores'])
                
                # 从step2数据中提取指定分段类型的相似度分数
                if segment_type in dataset2[video_name]:
                    scores2 = np.array(dataset2[video_name][segment_type]['similarities'])
                else:
                    continue  # 如果没有该分段类型，跳过该视频
                
                # 如果 similarities 只有一个值，需要扩展到与 scores1 相同的长度
                if len(scores2) == 1 and len(scores1) > 1:
                    scores2 = np.full(len(scores1), scores2[0])
                elif len(scores1) == 1 and len(scores2) > 1:
                    scores1 = np.full(len(scores2), scores1[0])
                
                # 确保两个数组长度一致
                min_len = min(len(scores1), len(scores2))
                scores1 = scores1[:min_len]
                scores2 = scores2[:min_len]
                
                # 加权平均
                weighted_scores = weight_a * scores1 + (1 - weight_a) * scores2
                
                video_key = evaluator.dataset.video_name_dict[video_name]
                pred_scores[video_key] = weighted_scores.tolist()
        
        if pred_scores:
            mean_f1 = evaluator.evaluate_batch(pred_scores)
            results[dataset_name] = mean_f1
    
    return results

def evaluate_step2_similarity_scores(step2_data, segment_type="seg_2"):
    """评估step2生成的相似度分数"""
    results = {}
    
    for dataset_name in step2_data['datasets']:
        dataset_videos = step2_data['datasets'][dataset_name]
        
        # 根据数据集名称推断数据集路径
        if dataset_name == "SumMe":
            dataset_path = "/root/autodl-tmp/data/SumMe"
        elif dataset_name == "TVSum":
            dataset_path = "/root/autodl-tmp/data/TVSum"
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
            
        evaluator = create_evaluator(dataset_name, dataset_path)
        pred_scores = {}
        
        processed_videos = 0
        for video_name in dataset_videos:
            if video_name in evaluator.dataset.video_name_dict:
                video_data = dataset_videos[video_name]
                
                # 使用指定的分段类型
                if segment_type in video_data:
                    similarities = video_data[segment_type]['similarities']
                    
                    if similarities:  # 确保有相似度分数
                        video_key = evaluator.dataset.video_name_dict[video_name]
                        pred_scores[video_key] = similarities
                        processed_videos += 1
                else:
                    print(f"Segment type {segment_type} not found for video {video_name}")
        
        if pred_scores:
            mean_f1 = evaluator.evaluate_batch(pred_scores)
            results[dataset_name] = {
                'f1_score': mean_f1,
                'processed_videos': processed_videos,
                'total_videos': len(dataset_videos)
            }
            print(f"{dataset_name}: F1={mean_f1:.4f} ({processed_videos}/{len(dataset_videos)} videos)")
        else:
            print(f"No valid scores found for {dataset_name}")
    
    return results

def evaluate_all_step2_segments(step2_data):
    """评估所有分段类型的相似度分数"""
    segment_types = ["seg_2", "seg_4", "seg_6", "seg_8", "seg_16", "seg_32"]
    all_results = {}
    
    print("Evaluating all segment types:")
    print("=" * 50)
    
    for seg_type in segment_types:
        print(f"\nEvaluating {seg_type}:")
        results = evaluate_step2_similarity_scores(step2_data, seg_type)
        all_results[seg_type] = results
        
        # 打印该分段类型的结果摘要
        if results:
            avg_f1 = np.mean([r['f1_score'] for r in results.values()])
            print(f"{seg_type} average F1: {avg_f1:.4f}")
    
    # 找出每个数据集的最佳分段类型
    print("\n" + "=" * 50)
    print("Best F1 scores by dataset and segment type:")
    
    # 获取所有数据集名称
    dataset_names = set()
    for seg_type, results in all_results.items():
        if results:
            dataset_names.update(results.keys())
    
    best_by_dataset = {}
    
    for dataset_name in dataset_names:
        print(f"\n{dataset_name}:")
        best_f1 = 0
        best_seg = None
        
        for seg_type in segment_types:
            if seg_type in all_results and all_results[seg_type] and dataset_name in all_results[seg_type]:
                f1_score = all_results[seg_type][dataset_name]['f1_score']
                print(f"  {seg_type}: {f1_score:.4f}")
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_seg = seg_type
        
        if best_seg:
            best_by_dataset[dataset_name] = {'segment_type': best_seg, 'f1_score': best_f1}
            print(f"  Best: {best_seg} (F1: {best_f1:.4f})")
    
    print("\n" + "=" * 50)
    print("Summary - Best segment type for each dataset:")
    for dataset_name, best_info in best_by_dataset.items():
        print(f"{dataset_name}: {best_info['segment_type']} (F1: {best_info['f1_score']:.4f})")
    
    return all_results, best_by_dataset

def evaluate_step3_redundancy_scores(step3_data, redundancy_type="mean_similarity"):
    """评估step3生成的冗余度分数"""
    results = {}
    
    for dataset_name in step3_data['datasets']:
        dataset_videos = step3_data['datasets'][dataset_name]
        
        # 根据数据集名称推断数据集路径
        if dataset_name == "SumMe":
            dataset_path = "/root/autodl-tmp/data/SumMe"
        elif dataset_name == "TVSum":
            dataset_path = "/root/autodl-tmp/data/TVSum"
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
            
        evaluator = create_evaluator(dataset_name, dataset_path)
        pred_scores = {}
        
        processed_videos = 0
        for video_name in dataset_videos:
            if video_name in evaluator.dataset.video_name_dict:
                video_data = dataset_videos[video_name]
                
                if redundancy_type in video_data and 'scores' in video_data[redundancy_type]:
                    redundancy_scores = np.array(video_data[redundancy_type]['scores'])
                    # 使用1-冗余度作为重要性分数
                    importance_scores = 1 - redundancy_scores
                    
                    video_key = evaluator.dataset.video_name_dict[video_name]
                    pred_scores[video_key] = importance_scores.tolist()
                    processed_videos += 1
        
        if pred_scores:
            mean_f1 = evaluator.evaluate_batch(pred_scores)
            results[dataset_name] = {
                'f1_score': mean_f1,
                'processed_videos': processed_videos,
                'total_videos': len(dataset_videos)
            }
            print(f"{dataset_name}: F1={mean_f1:.4f} ({processed_videos}/{len(dataset_videos)} videos)")
        else:
            print(f"No valid scores found for {dataset_name}")
    
    return results

def evaluate_three_step_weighted_scores(step1_data, step2_data, step3_data, weight_a, weight_b, weight_c, segment_type="seg_2", redundancy_type="mean_similarity"):
    """评估三步加权分数 (weight_a * step1 + weight_b * step2 + weight_c * (1-step3))"""
    results = {}
    
    for dataset_name in step1_data['datasets']:
        if dataset_name not in step2_data['datasets'] or dataset_name not in step3_data['datasets']:
            continue
            
        dataset1 = step1_data['datasets'][dataset_name]
        dataset2 = step2_data['datasets'][dataset_name]
        dataset3 = step3_data['datasets'][dataset_name]
        
        evaluator = create_evaluator(dataset_name, dataset1['dataset_path'])
        pred_scores = {}
        
        for video_name in dataset1['videos']:
            if (video_name in dataset2 and video_name in dataset3 and 
                video_name in evaluator.dataset.video_name_dict):
                
                # Step1 scores
                scores1 = extract_scores(dataset1['videos'][video_name]['scores'])
                
                # Step2 scores
                if segment_type in dataset2[video_name]:
                    scores2 = np.array(dataset2[video_name][segment_type]['similarities'])
                else:
                    continue
                
                # Step3 scores (1 - redundancy)
                if redundancy_type in dataset3[video_name] and 'scores' in dataset3[video_name][redundancy_type]:
                    redundancy_scores = np.array(dataset3[video_name][redundancy_type]['scores'])
                    scores3 = 1 - redundancy_scores
                else:
                    continue
                
                # 确保长度一致
                min_len = min(len(scores1), len(scores2), len(scores3))
                scores1 = scores1[:min_len]
                scores2 = scores2[:min_len]
                scores3 = scores3[:min_len]
                
                # 三步加权
                weighted_scores = weight_a * scores1 + weight_b * scores2 + weight_c * (1-scores3)
                
                video_key = evaluator.dataset.video_name_dict[video_name]
                pred_scores[video_key] = weighted_scores.tolist()
        
        if pred_scores:
            mean_f1 = evaluator.evaluate_batch(pred_scores)
            results[dataset_name] = mean_f1
    
    return results

def main():
    # 评估step2的相似度分数
    print("Evaluating Step2 Similarity Scores")
    print("=" * 50)
    
    # 加载step2数据
    with open('resource/step2_simple_results.json') as f:
        step2_data = json.load(f)
    
    # 评估所有分段类型
    step2_results, best_by_dataset = evaluate_all_step2_segments(step2_data)
    
    print("\n" + "=" * 50)
    print("Evaluating Combined Scores (Step1 + Step2 Simple)")
    print("=" * 50)
    
    # 加载step1数据进行组合评估
    try:
        with open('resource/step1_results.json') as f:
            step1_data = json.load(f)
        # 使用新的step2简化结果
        step2_simple_data = step2_data
        
        # 权重范围
        weights = np.arange(0, 1.1, 0.1)
        
        # 为每个数据集使用其最佳分段类型进行组合评估
        print("Testing with best segment type for each dataset:")
        for dataset_name, best_info in best_by_dataset.items():
            best_seg_type = best_info['segment_type']
            print(f"\n{dataset_name} - Using {best_seg_type}:")
            
            dataset_best_results = {}
            
            for weight_a in weights:
                results = evaluate_weighted_scores(step1_data, step2_simple_data, weight_a, segment_type=best_seg_type)
                
                if dataset_name in results:
                    f1_score = results[dataset_name]
                    if weight_a not in dataset_best_results or f1_score > dataset_best_results[weight_a]:
                        dataset_best_results[weight_a] = f1_score
                    print(f"  Weight a={weight_a:.1f}: F1={f1_score:.4f}")
            
            # 找出该数据集的最佳权重
            if dataset_best_results:
                best_weight = max(dataset_best_results.keys(), key=lambda k: dataset_best_results[k])
                best_f1 = dataset_best_results[best_weight]
                print(f"  Best for {dataset_name}: weight_a={best_weight:.1f}, F1={best_f1:.4f}")
        
        # 也测试使用seg_2进行统一比较
        print(f"\nFor comparison - Using seg_2 for all datasets:")
        best_results = {}
        
        for weight_a in weights:
            results = evaluate_weighted_scores(step1_data, step2_simple_data, weight_a, segment_type="seg_2")
            
            for dataset_name, f1_score in results.items():
                if dataset_name not in best_results or f1_score > best_results[dataset_name]['f1']:
                    best_results[dataset_name] = {'weight': weight_a, 'f1': f1_score}
            
            print(f"Weight a={weight_a:.1f}: {', '.join(f'{k}={v:.4f}' for k, v in results.items())}")
        
        # 输出最佳结果
        print("\n最佳权重组合结果 (using seg_2):")
        for dataset_name, result in best_results.items():
            print(f"{dataset_name}: weight_a={result['weight']:.1f}, F1={result['f1']:.4f}")
            
    except FileNotFoundError as e:
        print(f"\nSkipping combined evaluation due to missing file: {e}")
    
    # 评估step3冗余度分数
    print("\n" + "=" * 50)
    print("Evaluating Step3 Redundancy Scores")
    print("=" * 50)
    
    # 加载step3数据
    with open('resource/step3_redundancy_results.json') as f:
        step3_data = json.load(f)
    
    # 评估不同冗余度类型
    redundancy_types = ["mean_similarity", "topk_similarity", "threshold_count"]
    step3_results = {}
    
    for red_type in redundancy_types:
        print(f"\nEvaluating {red_type}:")
        results = evaluate_step3_redundancy_scores(step3_data, red_type)
        step3_results[red_type] = results
        
        if results:
            avg_f1 = np.mean([r['f1_score'] for r in results.values()])
            print(f"{red_type} average F1: {avg_f1:.4f}")
    
    # 三步结合评估
    print("\n" + "=" * 50)
    print("Evaluating Three-Step Combined Scores")
    print("=" * 50)
    
    try:
        with open('resource/step1_results.json') as f:
            step1_data = json.load(f)
        with open('resource/step2_simple_results.json') as f:
            step2_data = json.load(f)
        
        # 测试不同权重组合
        weight_combinations = [
            (0.5, 0.3, 0.2),  # step1主导
            (0.3, 0.5, 0.2),  # step2主导
            (0, 0, 1),  # step3主导
            (0.33, 0.33, 0.34)  # 平均权重
        ]
        
        print("Testing weight combinations (step1, step2, step3):")
        for weight_a, weight_b, weight_c in weight_combinations:
            results = evaluate_three_step_weighted_scores(
                step1_data, step2_data, step3_data, 
                weight_a, weight_b, weight_c, 
                segment_type="seg_2", redundancy_type="mean_similarity"
            )
            
            if results:
                avg_f1 = np.mean(list(results.values()))
                print(f"Weights ({weight_a:.2f}, {weight_b:.2f}, {weight_c:.2f}): Avg F1={avg_f1:.4f}")
                for dataset_name, f1_score in results.items():
                    print(f"  {dataset_name}: {f1_score:.4f}")
            
    except FileNotFoundError as e:
        print(f"Skipping three-step evaluation due to missing file: {e}")
    
    return step2_results
    
    # 额外评估步骤2相似度分数
    print("\n评估步骤2相似度分数:")
    all_step2_results = evaluate_all_step2_segments(step2_data)

if __name__ == "__main__":
    main()