import numpy as np
from evaluation_metrics import get_corr_coeff

# SumMe 多视频示例
def test_summe():
    pred_imp_scores = [
        np.array([0.1, 0.5, 0.9]),
        np.array([0.2, 0.6, 0.8])
    ]
    videos = ["1", "2"]
    dataset = "SumMe"
    user_scores = np.array([
        [0.2, 0.4, 0.8],
        [0.1, 0.5, 0.7]
    ])
    rho, tau = get_corr_coeff(pred_imp_scores, videos, dataset, user_scores)
    print("SumMe Test:")
    print("Spearman:", rho)
    print("Kendall:", tau)

# TVSum 多视频示例
def test_tvsum():
    pred_imp_scores = [
        np.array([0.3, 0.6, 0.2]),
        np.array([0.7, 0.1, 0.5])
    ]
    videos = ["video_1", "video_2"]
    dataset = "TVSum"
    user_scores = [
        [np.array([0.2, 0.5, 0.3]), np.array([0.4, 0.7, 0.1])],
        [np.array([0.6, 0.2, 0.8]), np.array([0.5, 0.3, 0.7])]
    ]
    rho, tau = get_corr_coeff(pred_imp_scores, videos, dataset, user_scores)
    print("TVSum Test:")
    print("Spearman:", rho)
    print("Kendall:", tau)

if __name__ == "__main__":
    test_summe()
    print()
    test_tvsum()
