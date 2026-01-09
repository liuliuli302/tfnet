import numpy as np
from scipy.stats import spearmanr, kendalltau


def get_corr_coeff(pred_list, video_list, dataset_name, user_scores_list):
    """
    pred_list: List of predicted importance arrays, one per video.
    user_scores_list:
        - For SumMe: per-video 2D numpy array [n_users, n_frames]
        - For TVSum: per-video list of arrays (each [n_frames])
    """

    rho_list = []
    tau_list = []

    for pred, user_scores in zip(pred_list, user_scores_list):

        # --- 处理 SumMe 格式: 2D array ---
        if isinstance(user_scores, np.ndarray):
            # shape = (n_users, n_frames)
            true_scores = np.mean(user_scores, axis=0)

        # --- 处理 TVSum 格式: list of arrays ---
        elif isinstance(user_scores, list):
            try:
                stacked = np.stack(user_scores)
                true_scores = np.mean(stacked, axis=0)
            except Exception as e:
                print(f"[WARN] TVSum user_scores stacking failed: {e}")
                continue

        else:
            print(f"[WARN] Unknown user_scores type: {type(user_scores)}")
            continue

        # --- 计算 Spearman ρ 和 Kendall τ ---
        try:
            rho, _ = spearmanr(pred, true_scores)
            tau, _ = kendalltau(pred, true_scores)

            if rho is not None and not np.isnan(rho):
                rho_list.append(rho)

            if tau is not None and not np.isnan(tau):
                tau_list.append(tau)

        except Exception as e:
            print(f"[WARN] correlation error: {e}")
            continue

    # --- 返回平均值 ---
    if len(rho_list) == 0 or len(tau_list) == 0:
        return 0.0, 0.0

    return np.mean(rho_list), np.mean(tau_list)
