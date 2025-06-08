#!/usr/bin/env python3
"""
评估logs下的结果json文件的性能
"""

from eval.evaluator import Evaluator
from src.dataloader.vs_dataset import VideoSummarizationDataset  # Used by Evaluator
import os
import json
import sys
import argparse
from pathlib import Path

# Add project root and its parent to Python path
project_root = Path(__file__).parent
src_dir = str(project_root / 'src')
eval_dir = str(project_root / 'eval')
sys.path.extend([str(project_root), src_dir, eval_dir])


def find_latest_results():
    """Find the latest step1_results.json file in logs directory"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs directory not found")

    # Find all step1_results.json files
    result_files = list(logs_dir.glob("**/results/step1_results.json"))
    if not result_files:
        raise FileNotFoundError(
            "No step1_results.json found in logs directory")

    # Get the most recent file based on modification time
    return max(result_files, key=lambda p: p.stat().st_mtime)


def find_results_by_version(version: str):
    """Find step1_results.json file for a specific version"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs directory not found")
    
    # Look for the specific version
    version_path = logs_dir / "pipeline_01" / f"version_{version}" / "results" / "step1_results.json"
    if version_path.exists():
        return version_path
    else:
        raise FileNotFoundError(f"No step1_results.json found for version_{version}")


def list_available_versions():
    """List all available versions with results"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    
    pipeline_dir = logs_dir / "pipeline_01"
    if not pipeline_dir.exists():
        return []
    
    versions = []
    for version_dir in pipeline_dir.iterdir():
        if version_dir.is_dir() and version_dir.name.startswith("version_"):
            results_file = version_dir / "results" / "step1_results.json"
            if results_file.exists():
                version_num = version_dir.name.replace("version_", "")
                versions.append(version_num)
    
    return sorted(versions, key=int)


def print_video_scores_sorted(results_file: Path):
    """Print video F1 scores sorted from low to high by dataset"""
    with open(results_file, 'r') as f:
        results = json.load(f)

    dataset_scores = {}
    
    for dataset_name, dataset_info in results['datasets'].items():
        dataset_path = dataset_info['dataset_path']
        if 'experiment_cache/tfnet' in dataset_path:
            dataset_path = dataset_path.replace('/experiment_cache/tfnet', '/data')
        data_dir = Path(dataset_path).parent

        eval_method = "max" if dataset_name == "SumMe" else "avg"
        evaluator = Evaluator(dataset_name=dataset_name, data_dir=str(data_dir), eval_method=eval_method)
        video_name_dict = evaluator.dataset.video_name_dict

        video_scores = []
        for video_name, video_info in dataset_info['videos'].items():
            if video_name in video_name_dict:
                video_key = video_name_dict[video_name]
                video_index = int(video_key.split("_")[1]) - 1
                f1_score, _ = evaluator.evaluate(video_index, video_info['scores'])
                
                video_scores.append({
                    'video_name': video_name,
                    'f1_score': f1_score
                })
        
        # Sort videos by F1 score (low to high)
        video_scores.sort(key=lambda x: x['f1_score'])
        dataset_scores[dataset_name] = {
            'videos': video_scores,
            'eval_method': eval_method
        }

    # Print results by dataset
    for dataset_name, dataset_data in dataset_scores.items():
        video_scores = dataset_data['videos']
        eval_method = dataset_data['eval_method']
        
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name} (评估方法: {eval_method})")
        print(f"{'='*60}")
        print(f"{'排名':<4} {'视频名称':<30} {'F1分数':<8}")
        print(f"{'-'*50}")
        
        for i, video_info in enumerate(video_scores, 1):
            print(f"{i:<4} {video_info['video_name']:<30} {video_info['f1_score']:<8.4f}")
        
        if video_scores:
            scores = [v['f1_score'] for v in video_scores]
            worst_3 = video_scores[:3]
            best_3 = video_scores[-3:]
            
            print(f"\n{dataset_name} 统计信息:")
            print(f"最低F1分数: {min(scores):.4f}")
            print(f"最高F1分数: {max(scores):.4f}")
            print(f"平均F1分数: {sum(scores) / len(scores):.4f}")
            print(f"视频总数: {len(video_scores)}")
            
            print(f"\n表现最差的3个视频:")
            for i, v in enumerate(worst_3, 1):
                print(f"  {i}. {v['video_name']}: {v['f1_score']:.4f}")
                
            print(f"\n表现最好的3个视频:")
            for i, v in enumerate(best_3, 1):
                print(f"  {i}. {v['video_name']}: {v['f1_score']:.4f}")
    
    # Overall summary
    all_scores = []
    for dataset_data in dataset_scores.values():
        all_scores.extend([v['f1_score'] for v in dataset_data['videos']])
    
    if all_scores:
        print(f"\n{'='*60}")
        print("总体统计:")
        print(f"{'='*60}")
        print(f"总视频数: {len(all_scores)}")
        print(f"总体平均F1分数: {sum(all_scores) / len(all_scores):.4f}")
        print(f"总体最低F1分数: {min(all_scores):.4f}")
        print(f"总体最高F1分数: {max(all_scores):.4f}")


def evaluate_results(results_file: Path):
    """Evaluate the results from the json file"""
    with open(results_file, 'r') as f:
        results = json.load(f)

    total_f1_scores = []
    for dataset_name, dataset_info in results['datasets'].items():
        print(f"\nEvaluating dataset: {dataset_name}")

        dataset_path = dataset_info['dataset_path']
        if 'experiment_cache/tfnet' in dataset_path:
            dataset_path = dataset_path.replace('/experiment_cache/tfnet', '/data')
        data_dir = Path(dataset_path).parent

        eval_method = "max" if dataset_name == "SumMe" else "avg"
        print(f"Using evaluation method: {eval_method}")

        evaluator = Evaluator(dataset_name=dataset_name, data_dir=str(data_dir), eval_method=eval_method)
        video_name_dict = evaluator.dataset.video_name_dict

        pred_scores = {}
        for video_name, video_info in dataset_info['videos'].items():
            if video_name in video_name_dict:
                video_key = video_name_dict[video_name]
                pred_scores[video_key] = video_info['scores']

        mean_f1 = evaluator.evaluate_batch(pred_scores)
        print(f"Dataset {dataset_name} mean F1 score ({eval_method}): {mean_f1:.4f}")
        total_f1_scores.append(mean_f1)

    if total_f1_scores:
        overall_mean = sum(total_f1_scores) / len(total_f1_scores)
        print(f"\nOverall mean F1 score across all datasets: {overall_mean:.4f}")


def main():
    parser = argparse.ArgumentParser(description='评估logs下的结果json文件的性能')
    parser.add_argument('--version', '-v', type=str, 
                       help='指定要评估的version号 (例如: 1, 4, 7)。如果不指定，将使用最新的结果')
    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有可用的version')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='显示每个视频的详细F1分数排序')
    
    args = parser.parse_args()
    
    if args.list:
        available_versions = list_available_versions()
        if available_versions:
            print("可用的版本:")
            for version in available_versions:
                print(f"  version_{version}")
        else:
            print("没有找到可用的版本")
        return
    
    # Find the results file
    if args.version:
        results_file = find_results_by_version(args.version)
        print(f"Found results file for version_{args.version}: {results_file}")
    else:
        results_file = find_latest_results()
        print(f"Found latest results file: {results_file}")
    
    # Evaluate the results
    evaluate_results(results_file)
    
    # Show detailed video scores if requested
    if args.detailed:
        print_video_scores_sorted(results_file)


if __name__ == "__main__":
    main()
