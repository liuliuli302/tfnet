import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.dataloader.vs_dataset import VideoSummarizationDataset


def knapSack(W, wt, val, n):
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]

    return selected


def generate_single_summary(shot_bound, scores, n_frames, positions):
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(positions) - 1):
        start, end = positions[i], positions[i + 1]
        score = scores[i] if i < len(scores) else 0
        frame_scores[start:end] = score

    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append(frame_scores[shot[0]: shot[1] + 1].mean().item())

    final_max_length = int((shot_bound[-1][1] + 1) * 0.15)
    selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    summary = np.zeros(n_frames, dtype=np.int8)
    for shot in selected:
        summary[shot_bound[shot][0]: shot_bound[shot][1] + 1] = 1

    return summary


def evaluate_single_summary(predicted_summary, user_summary, eval_method):
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        precision = sum(overlapped) / sum(S) if sum(S) else 0
        recall = sum(overlapped) / sum(G) if sum(G) else 0
        if precision + recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    return max(f_scores) if eval_method == "max" else sum(f_scores) / len(f_scores)


# class Evaluator:
#     def __init__(self, dataset_name, data_dir, eval_method):
#         self.dataset = VideoSummarizationDataset(data_dir, dataset_name)
#         self.eval_method = eval_method

#     def evaluate(self, video_index, pred_scores):
#         data = self.dataset[video_index]
#         summary = generate_single_summary(
#             data["change_points"], pred_scores, data["n_frames"], data["picks"]
#         )
#         f_score = evaluate_single_summary(summary, data["user_summary"], self.eval_method)
#         return f_score, summary

#     def evaluate_batch(self, pred_score_dict):
#         all_scores = []
#         for video_i, pred_scores in tqdm(pred_score_dict.items()):
#             idx = int(video_i.split("_")[1]) - 1
#             f1, _ = self.evaluate(idx, pred_scores)
#             all_scores.append(f1)
#         return np.mean(all_scores) if all_scores else 0.0