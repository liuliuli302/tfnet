import os
import glob
import json
from tqdm import tqdm
import natsort
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import clip
from torchvision import models
import re


class loading_img(Dataset):
    def __init__(self, img_list, preprocess):
        self.img_list = img_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.copy()
        return self.preprocess(img)


@torch.inference_mode()
def select_frames(folder, preprocess, resnet18_pretrained, device, batch_size, frame_interval, divlam):
    img_list = natsort.natsorted(glob.glob(f"{folder}/*.jpg"))
    img_list = img_list[::frame_interval]
    img_set = loading_img(img_list, preprocess)
    img_loader = DataLoader(img_set, batch_size=batch_size,
                            shuffle=False, num_workers=8)
    features = []
    with torch.no_grad():
        for imgtensor in img_loader:
            imgtensor = imgtensor.to(device)
            feats = resnet18_pretrained(imgtensor)
            features.append(feats.cpu())
    featuremap = torch.cat(features, dim=0)

    with torch.no_grad():
        frame_num = featuremap.shape[0]
        dist_list = []
        for img_feat in featuremap:
            dist_list.append(torch.mean(torch.sqrt(
                (featuremap-img_feat)**2), dim=-1))
        dist_list = torch.concat(dist_list).reshape(frame_num, frame_num)

        idx_list = [_ for _ in range(frame_num)]
        loop_idx = 0
        out_frames = []
        output_results = []

        while len(idx_list) > 5:
            dist_idx = idx_list.pop(0)

            data = dist_list[dist_idx, idx_list].softmax(dim=-1)
            mu, std = torch.mean(data), torch.std(data)
            pop_idx_list = torch.where(
                data < mu-std*(np.exp(1-loop_idx/divlam)))[0].detach().cpu().numpy()
            result = list(np.array(idx_list)[pop_idx_list])
            result.append(dist_idx)
            output_results.append(result)

            num_picks = 18
            if len(result) > num_picks:
                idx_result_list = sorted(random.sample(result, num_picks))
                img_list = np.array(img_list)
                idx_result_list = np.array(idx_result_list)
                out_frames.extend(img_list[idx_result_list])
            else:
                idx_result_list = sorted(result)
                img_list = np.array(img_list)
                idx_result_list = np.array(idx_result_list)
                out_frames.extend(img_list[idx_result_list])

            loop_idx += 1

            for pop_idx in reversed(pop_idx_list):
                idx_list.pop(pop_idx)

    return out_frames, output_results


@torch.inference_mode()
def nfs_from_lvnet(
    frames_dir, out_dir, batch_size, dataset_name, frame_interval, divlam
):
    # Too Many Frames, not all Useful: Efficient Strategies for Long-Form Video QA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet18_pretrained = models.resnet18(pretrained=True).to(device)
    resnet18_pretrained.fc = torch.nn.Identity()
    resnet18_pretrained.avgpool = torch.nn.Identity()
    resnet18_pretrained.eval()
    model, preprocess = clip.load("ViT-B/32", device=device)

    output_results = {}

    for video_name in tqdm(os.listdir(frames_dir), desc=f"Processing NFS for {dataset_name}"):
        video_frames_dir = os.path.join(frames_dir, video_name)
        out_frames, output_result = select_frames(
            video_frames_dir, preprocess, resnet18_pretrained, device, batch_size, frame_interval, divlam)
        picks = [int(path.split('/')[-1].split('.')[0]) for path in out_frames]
        output_results[video_name] = {
            "frame_paths": out_frames,
            "picks": picks,
        }

    def convert_to_builtin_type(obj):
        if isinstance(obj, dict):
            return {k: convert_to_builtin_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_builtin_type(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_builtin_type(v) for v in obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        else:
            return obj
    out_dir = os.path.join(out_dir, "nfs_lvnet")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_json = os.path.join(out_dir, f"{dataset_name}_nfs_lvnet.json")

    with open(out_json, "w") as f:
        json.dump(convert_to_builtin_type(output_results), f, indent=4)

    del out_frames, output_result
    torch.cuda.empty_cache()


if __name__ == "__main__":

    # 使用文献方法进行非均匀采样
    nfs_from_lvnet(
        frames_dir="/root/autodl-tmp/datasets/SumMe/frames",
        out_dir="/root/tfnet/data/nfs",
        batch_size=64,
        dataset_name="summe",
        frame_interval=1,
        divlam=8
    )

    nfs_from_lvnet(
        frames_dir="/root/autodl-tmp/datasets/TVSum/frames",
        out_dir="/root/tfnet/data/nfs",
        batch_size=64,
        dataset_name="tvsum",
        frame_interval=1,
        divlam=8
    )
