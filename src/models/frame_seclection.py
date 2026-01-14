import os
import glob
import json
import random
from typing import List, Tuple, Dict, Any, Optional

import clip
import natsort
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

from src.config.config import BasicConfig


class FrameSelectionDataset(Dataset):
    def __init__(self, img_list: List[str], preprocess):
        self.img_list = img_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx: int):
        img_path = self.img_list[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.copy()
        return self.preprocess(img)


class LVNetFrameSelectorConfig(BasicConfig):
    def __init__(
        self,
        frames_dir: str,
        out_dir: str,
        dataset_name: str,
        batch_size: int = 64,
        frame_interval: int = 1,
        divlam: float = 8.0,
        num_picks: int = 18,
        min_frames: int = 5,
        num_workers: int = 8,
        device: Optional[str] = None,
        clip_model: str = "ViT-B/32",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frames_dir = frames_dir
        self.out_dir = out_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.divlam = divlam
        self.num_picks = num_picks
        self.min_frames = min_frames
        self.num_workers = num_workers
        self.clip_model = clip_model
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")


class LVNetFrameSelector:
    """
    Frame selector based on LVNet non-uniform sampling.
    Encapsulates model loading, frame scoring, and result persistence.
    """

    def __init__(self, config: LVNetFrameSelectorConfig):
        if not isinstance(config, LVNetFrameSelectorConfig):
            raise TypeError("config must be LVNetFrameSelectorConfig")
        self.config = config
        self.device = torch.device(config.device)
        self.resnet18_pretrained = None
        self.preprocess = None
        self._load_models()

    def _load_models(self):
        resnet18_pretrained = models.resnet18(pretrained=True)
        resnet18_pretrained.fc = torch.nn.Identity()
        resnet18_pretrained.avgpool = torch.nn.Identity()
        self.resnet18_pretrained = resnet18_pretrained.to(self.device)
        self.resnet18_pretrained.eval()
        _, self.preprocess = clip.load(
            self.config.clip_model, device=self.device)

    def _compute_features(self, img_list: List[str]) -> torch.Tensor:
        dataset = FrameSelectionDataset(img_list, self.preprocess)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        features = []
        for imgtensor in loader:
            imgtensor = imgtensor.to(self.device)
            feats = self.resnet18_pretrained(imgtensor)
            features.append(feats.cpu())
        return torch.cat(features, dim=0)

    @torch.inference_mode()
    def select_frames(self, folder: str) -> Tuple[List[str], List[List[int]]]:
        img_list = natsort.natsorted(glob.glob(os.path.join(folder, "*.jpg")))
        img_list = img_list[:: self.config.frame_interval]
        if not img_list:
            return [], []

        featuremap = self._compute_features(img_list)
        frame_num = featuremap.shape[0]
        dist_list = []
        for img_feat in featuremap:
            dist_list.append(torch.mean(torch.sqrt(
                (featuremap - img_feat) ** 2), dim=-1))
        dist_list = torch.concat(dist_list).reshape(frame_num, frame_num)

        idx_list = list(range(frame_num))
        loop_idx = 0
        out_frames: List[str] = []
        output_results: List[List[int]] = []

        while len(idx_list) > self.config.min_frames:
            dist_idx = idx_list.pop(0)
            data = dist_list[dist_idx, idx_list].softmax(dim=-1)
            mu, std = torch.mean(data), torch.std(data)
            threshold = mu - std * (np.exp(1 - loop_idx / self.config.divlam))
            pop_idx_list = torch.where(data < threshold)[
                0].detach().cpu().numpy()
            result = list(np.array(idx_list)[pop_idx_list])
            result.append(dist_idx)
            output_results.append(result)

            if len(result) > self.config.num_picks:
                idx_result_list = sorted(
                    random.sample(result, self.config.num_picks))
            else:
                idx_result_list = sorted(result)

            img_arr = np.array(img_list)
            out_frames.extend(img_arr[np.array(idx_result_list)].tolist())

            loop_idx += 1
            for pop_idx in reversed(pop_idx_list):
                idx_list.pop(pop_idx)

        return out_frames, output_results

    @torch.inference_mode()
    def run(self) -> Dict[str, Dict[str, Any]]:
        output_results: Dict[str, Dict[str, Any]] = {}
        for video_name in tqdm(sorted(os.listdir(self.config.frames_dir)), desc=f"Processing NFS for {self.config.dataset_name}"):
            video_frames_dir = os.path.join(self.config.frames_dir, video_name)
            if not os.path.isdir(video_frames_dir):
                continue
            out_frames, output_result = self.select_frames(video_frames_dir)
            picks = [int(path.split("/")[-1].split(".")[0])
                     for path in out_frames]
            output_results[video_name] = {
                "frame_paths": out_frames,
                "picks": picks,
                "raw_indices": output_result,
            }

        self._save_results(output_results)
        torch.cuda.empty_cache()
        return output_results

    def _save_results(self, output_results: Dict[str, Dict[str, Any]]):
        out_dir = os.path.join(self.config.out_dir, "nfs_lvnet")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_json = os.path.join(
            out_dir, f"{self.config.dataset_name}_nfs_lvnet.json")
        with open(out_json, "w") as f:
            json.dump(self._convert_to_builtin_type(
                output_results), f, indent=4)

    @staticmethod
    def _convert_to_builtin_type(obj: Any):
        if isinstance(obj, dict):
            return {k: LVNetFrameSelector._convert_to_builtin_type(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [LVNetFrameSelector._convert_to_builtin_type(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(LVNetFrameSelector._convert_to_builtin_type(v) for v in obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj


@torch.inference_mode()
def nfs_from_lvnet(
    frames_dir: str,
    out_dir: str,
    batch_size: int,
    dataset_name: str,
    frame_interval: int,
    divlam: float,
    num_picks: int = 18,
    min_frames: int = 5,
    num_workers: int = 8,
    device: Optional[str] = None,
):
    """
    Backward-compatible helper that instantiates the selector and runs sampling.
    """
    selector_config = LVNetFrameSelectorConfig(
        frames_dir=frames_dir,
        out_dir=out_dir,
        dataset_name=dataset_name,
        batch_size=batch_size,
        frame_interval=frame_interval,
        divlam=divlam,
        num_picks=num_picks,
        min_frames=min_frames,
        num_workers=num_workers,
        device=device,
    )
    selector = LVNetFrameSelector(selector_config)
    return selector.run()


if __name__ == "__main__":
    # Example usage: non-uniform sampling on SumMe and TVSum
    nfs_from_lvnet(
        frames_dir="/root/autodl-tmp/datasets/SumMe/frames",
        out_dir="/root/tfnet/data/nfs",
        batch_size=64,
        dataset_name="summe",
        frame_interval=1,
        divlam=8,
    )

    nfs_from_lvnet(
        frames_dir="/root/autodl-tmp/datasets/TVSum/frames",
        out_dir="/root/tfnet/data/nfs",
        batch_size=64,
        dataset_name="tvsum",
        frame_interval=1,
        divlam=8,
    )
