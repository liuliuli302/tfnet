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

def evaluate_weighted_scores(step1_data, step2_data, weight_a):
    """评估加权分数"""
    results = {}
    
    for dataset_name in step1_data['datasets']:
        dataset1 = step1_data['datasets'][dataset_name]
        dataset2 = step2_data['datasets'][dataset_name]
        
        evaluator = create_evaluator(dataset_name, dataset1['dataset_path'])
        pred_scores = {}
        
        for video_name in dataset1['videos']:
            if video_name in dataset2['videos'] and video_name in evaluator.dataset.video_name_dict:
                scores1 = np.array(dataset1['videos'][video_name]['scores'])
                scores2 = np.array(dataset2['videos'][video_name]['similarities'])
                
                # 加权平均
                weighted_scores = weight_a * scores1 + (1 - weight_a) * scores2
                
                video_key = evaluator.dataset.video_name_dict[video_name]
                pred_scores[video_key] = weighted_scores.tolist()
        
        if pred_scores:
            mean_f1 = evaluator.evaluate_batch(pred_scores)
            results[dataset_name] = mean_f1
    
    return results

def main():
    # 加载数据
    with open('resource/step1_results.json') as f:
        step1_data = json.load(f)
    with open('resource/step2_similarity_scores.json') as f:
        step2_data = json.load(f)
    
    # 权重范围
    weights = np.arange(0, 1.1, 0.1)
    
    best_results = {}
    
    # 遍历所有权重
    for weight_a in weights:
        results = evaluate_weighted_scores(step1_data, step2_data, weight_a)
        
        for dataset_name, f1_score in results.items():
            if dataset_name not in best_results or f1_score > best_results[dataset_name]['f1']:
                best_results[dataset_name] = {'weight': weight_a, 'f1': f1_score}
        
        print(f"Weight a={weight_a:.1f}: {', '.join(f'{k}={v:.4f}' for k, v in results.items())}")
    
    # 输出最佳结果
    print("\n最佳结果:")
    for dataset_name, result in best_results.items():
        print(f"{dataset_name}: weight_a={result['weight']:.1f}, F1={result['f1']:.4f}")

if __name__ == "__main__":
    main()