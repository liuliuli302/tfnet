"""
视频摘要流程
1 非均匀帧采样
2 新的QueryLLM的流程
3 相似度计算
4 评估
"""

from model.non_uniform_frame_sampling import temporal_scene_clustering_used
import warnings
warnings.filterwarnings("ignore")


def main():
    frames_dir_summe = "/root/autodl-tmp/data/SumMe/frames"
    frames_dir_tvsum = "/root/autodl-tmp/data/TVSum/frames"
    out_dir = "/root/tfnet/out"
    batch_size = 4

    temporal_scene_clustering_used(frames_dir_summe, out_dir, batch_size)
    temporal_scene_clustering_used(frames_dir_tvsum, out_dir, batch_size)


if __name__ == "__main__":
    main()
