from dataclasses import dataclass
import json
import os
import h5py
import yaml
import torch
from torch.utils.data import Dataset
from src.config.config import BasicConfig


@dataclass
class VideoSummarizationDatasetConfig(BasicConfig):
    dataset_name: str
    dataset_dir: str
    video_dir: str
    h5file_path: str
    split_files: dict
    video_name_mapping_file_path: str
    video_extension: str


class VideoSummarizationDataset(Dataset):

    def __init__(self, config: VideoSummarizationDatasetConfig):
        """
        Args:
            config (VideoSummarizationDatasetConfig): 配置对象
        """
        # 加载基本配置
        self.config = config
        self.dataset_name = config.dataset_name
        self.dataset_dir = config.dataset_dir
        self.video_dir = config.video_dir
        self.h5file_path = config.h5file_path
        self.split_files = config.split_files
        self.video_name_mapping_file_path = config.video_name_mapping_file_path
        self.video_extension = config.video_extension

        # 加载video_name_mapping
        self.video_name_mapping = {}
        with open(self.video_name_mapping_file_path, 'r') as f:
            self.video_name_mapping = json.load(f)

        # 初始化数据列表
        self.data_list = {}
        with h5py.File(self.h5file_path, 'r') as h5file:
            h5_dict = self.h5_to_dict(h5file)
            keys = list(h5_dict.keys())
            for key in keys:
                # TVSum数据集的h5文件中没有video_name，需要通过name mapping
                if self.dataset_name == "TVSum":
                    h5_dict[key]['video_name'] = self.video_name_mapping.get(
                        key)
                else:
                    h5_dict[key]['video_name'] = h5_dict[key]['video_name'].decode(
                        'utf-8')
                h5_dict[key]['video_path'] = os.path.join(
                    self.video_dir,
                    h5_dict[key]['video_name'] + self.video_extension
                )
            self.data_list = h5_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[f'video_{idx+1}']
        return data_item

    @staticmethod
    def h5_to_dict(obj):
        out = {}
        for k, v in obj.items():
            if isinstance(v, h5py.Dataset):
                out[k] = v[()]            # 读取为 numpy 数组或标量
            elif isinstance(v, h5py.Group):
                out[k] = VideoSummarizationDataset.h5_to_dict(v)    # 递归子组
        return out


if __name__ == "__main__":
    summe_config = VideoSummarizationDatasetConfig.load_config_from_file(
        "/root/tfnet/configs/dataset/summe.yaml")

    summe_dataset = VideoSummarizationDataset(config=summe_config)
    print(type(summe_dataset[0]))
    print(f"Dataset size: {len(summe_dataset)}")
