#!/usr/bin/env python3
"""
评估logs下的结果json文件的性能
"""

import os
import json
import sys
from pathlib import Path

# Add project root and its parent to Python path
project_root = Path(__file__).parent
src_dir = str(project_root / 'src')
eval_dir = str(project_root / 'eval')
sys.path.extend([str(project_root), src_dir, eval_dir])

from src.dataloader.vs_dataset import VideoSummarizationDataset  # Used by Evaluator

from eval.evaluator import Evaluator

def find_latest_results():
    """Find the latest step1_results.json file in logs directory"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs directory not found")

    # Find all step1_results.json files
    result_files = list(logs_dir.glob("**/results/step1_results.json"))
    if not result_files:
        raise FileNotFoundError("No step1_results.json found in logs directory")

    # Get the most recent file based on modification time
    return max(result_files, key=lambda p: p.stat().st_mtime)

def evaluate_results(results_file: Path):
    """Evaluate the results from the json file"""
    with open(results_file, 'r') as f:
        results = json.load(f)

    total_f1_scores = []
    for dataset_name, dataset_info in results['datasets'].items():
        print(f"\nEvaluating dataset: {dataset_name}")
        
        # Initialize evaluator for this dataset
        dataset_path = dataset_info['dataset_path']
        # Convert old path to new path if needed
        if 'experiment_cache/tfnet' in dataset_path:
            dataset_path = dataset_path.replace('/experiment_cache/tfnet', '/data')
        data_dir = Path(dataset_path).parent
        
        # Choose evaluation method based on dataset
        eval_method = "max" if dataset_name == "SumMe" else "avg"
        print(f"Using evaluation method: {eval_method}")
        
        evaluator = Evaluator(dataset_name=dataset_name, 
                            data_dir=str(data_dir),
                            eval_method=eval_method)
        
        # Get scores for each video
        pred_scores = {}
        for video_name, video_info in dataset_info['videos'].items():
            scores = video_info['scores']
            indices = video_info['sampled_indices']
            
            # Create video key in expected format (e.g., "video_1" for first video)
            video_num = len(pred_scores) + 1
            video_key = f"video_{video_num}"
            pred_scores[video_key] = scores
            
        # Evaluate all videos in the dataset
        mean_f1 = evaluator.evaluate_batch(pred_scores)
        print(f"Dataset {dataset_name} mean F1 score ({eval_method}): {mean_f1:.4f}")
        total_f1_scores.append(mean_f1)
    
    if total_f1_scores:
        overall_mean = sum(total_f1_scores) / len(total_f1_scores)
        print(f"\nOverall mean F1 score across all datasets: {overall_mean:.4f}")

def main():
    try:
        # Find the latest results file
        results_file = find_latest_results()
        print(f"Found results file: {results_file}")
        
        # Evaluate the results
        evaluate_results(results_file)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
