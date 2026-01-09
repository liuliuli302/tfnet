''''
Courtesy of KaiyangZhou
https://github.com/KaiyangZhou/pytorch-vsumm-reinforce

@article{zhou2017reinforcevsumm,
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao},
   journal={arXiv:1801.00054},
   year={2017}
}

Modifications by Jiri Fajtl
- knapsack replaced with knapsack_ortools
- added evaluate_user_summaries() for user summaries ground truth evaluation
'''
'''
------------------------------------------------
Use dynamic programming (DP) to solve 0/1 knapsack problem
Time complexity: O(nW), where n is number of items and W is capacity

Author: Kaiyang Zhou
Website: https://kaiyangzhou.github.io/
------------------------------------------------
knapsack_dp(values,weights,n_items,capacity,return_all=False)

Input arguments:
  1. values: a list of numbers in either int or float, specifying the values of items
  2. weights: a list of int numbers specifying weights of items
  3. n_items: an int number indicating number of items
  4. capacity: an int number indicating the knapsack capacity
  5. return_all: whether return all info, defaulty is False (optional)

Return:
  1. picks: a list of numbers storing the positions of selected items
  2. max_val: maximum value (optional)
------------------------------------------------
'''
# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W

import numpy as np
from ortools.algorithms import pywrapknapsack_solver
import math
# from knapsack import knapsack_dp

# ===========================================


def knapsack_dp(values, weights, n_items, capacity, return_all=False):
    check_inputs(values, weights, n_items, capacity)

    table = np.zeros((n_items+1, capacity+1), dtype=np.float32)
    keep = np.zeros((n_items+1, capacity+1), dtype=np.float32)

    for i in range(1, n_items+1):
        for w in range(0, capacity+1):
            wi = weights[i-1]  # weight of current item
            vi = values[i-1]  # value of current item
            if (wi <= w) and (vi + table[i-1, w-wi] > table[i-1, w]):
                table[i, w] = vi + table[i-1, w-wi]
                keep[i, w] = 1
            else:
                table[i, w] = table[i-1, w]

    picks = []
    K = capacity

    for i in range(n_items, 0, -1):
        if keep[i, K] == 1:
            picks.append(i)
            K -= weights[i-1]

    picks.sort()
    picks = [x-1 for x in picks]  # change to 0-index

    if return_all:
        max_val = table[n_items, capacity]
        return picks, max_val
    return picks


def check_inputs(values, weights, n_items, capacity):
    # check variable type
    assert (isinstance(values, list))
    assert (isinstance(weights, list))
    assert (isinstance(n_items, int))
    assert (isinstance(capacity, int))
    # check value type
    assert (all(isinstance(val, int) or isinstance(val, float)
            for val in values))
    assert (all(isinstance(val, int) for val in weights))
    # check validity of value
    assert (all(val >= 0 for val in weights))
    assert (n_items > 0)
    assert (capacity > 0)


osolver = pywrapknapsack_solver.KnapsackSolver(
    # pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
    'test')

# values:shot得分列表, weights:每个shot包含的帧数，capacity：摘要所需要的帧数


def knapsack_ortools(values, weights, items, capacity):
    scale = 1000
    values = np.array(values)
    weights = np.array(weights)
    values = (values * scale).astype(int)
    weights = (weights).astype(int)
    capacity = capacity

    osolver.Init(values.tolist(), [weights.tolist()], [capacity])
    computed_value = osolver.Solve()
    packed_items = [x for x in range(0, len(weights))
                    if osolver.BestSolutionContains(x)]

    return packed_items


def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    n_segs = cps.shape[0]  # shot数
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    # positions即picks，被抽样的帧：0,15,30,45...
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate(
            [positions, [n_frames]])  # 把最后一帧序号加入positions中
    # 除最后添加n_frames，其它是每15帧共1个分数，即[0,14]1个分数，[15,29]1个分数
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    # 计算每个shot内所有帧的平均分
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))  # 计算生成摘要的帧数：总帧数 * 摘要比例

    if method == 'knapsack':
        # picks = knapsack_dp(seg_score, nfps, n_segs, limits)
        picks = knapsack_ortools(
            seg_score, nfps, n_segs, limits)  # 使用背包算法返回选中shot序号
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros((1), dtype=np.float32)  # this element should be deleted
    for seg_idx in range(n_segs):  # 选中shot的帧置为1，其它置为0
        nf = nfps[seg_idx]  # nf每个shot的长度
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0)  # delete the first element
    return summary, seg_score, picks  # seg_score为shot的得分，picks为选中的shot


def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users, n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx, :]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':  # tvsum取和每个人工打分f_score的均值
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':  # summe取和每个人工打分f_score的最大值
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]

    return final_f_score, final_prec, final_rec


def evaluate_user_summaries(user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    user_summary = user_summary.astype(np.float32)
    n_users, n_frames = user_summary.shape

    # binarization
    user_summary[user_summary > 0] = 1

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx, :]
        for other_user_idx in range(user_idx+1, n_users):
            other_gt_summary = user_summary[other_user_idx, :]
            overlap_duration = (other_gt_summary * gt_summary).sum()
            precision = overlap_duration / (other_gt_summary.sum() + 1e-8)
            recall = overlap_duration / (gt_summary.sum() + 1e-8)
            if precision == 0 and recall == 0:
                f_score = 0.
            else:
                f_score = (2 * precision * recall) / (precision + recall)
            f_scores.append(f_score)
            prec_arr.append(precision)
            rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]

    return final_f_score, final_prec, final_rec
