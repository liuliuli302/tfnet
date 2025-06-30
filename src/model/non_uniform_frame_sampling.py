# Lib
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
# import config

# base
dict_api = {
    "api_key": "ADD",
}
base_dir = "your_path"  # your data path ex) img_folder_path/egoschema


# scene clustering
divlam = 12
f_path = "your_path"  # files from keywords dir
q_path = "your_path"  # files from questions dir
a_path = "your_path"
# your img folder path ex) img_folder_path/egoschema/frames_900_4531/q_uid/image_sec_millisec.jpg
img_folder = "your_path"


# coarse key frame detector
maximgslen = 32
limit_keywords = 25
concatname = "LVnet"
modelpath = "your_path"  # model path
# recommend using the same path with scene clustering answer path
question_path = "your_path"
# kwkfmatching is not necessary.
answerpath = f"{base_dir}/kwkfmatching/kf_{concatname}.jsonl"
# kwkfmatching is not necessary.
concatdir = f"{base_dir}/kwkfmatching/concatimg_{concatname}"


# fine key frame detector
kf_vlm = "gpt-4o"
kf_temp = None
kf_num_select = 3
kf_num_input_imgs = 32
# recommend using the same path with coarse key frame detector answer path
kf_question_path = "your_path"
# kf_VLM is not necessary.
kf_answer_path = f"{base_dir}/kf_VLM/kf_VLM{kf_num_input_imgs}sel{kf_num_select}_{kf_question_path.split('/')[-1].split('.')[0]}.jsonl"


# fine key frame detector refine
refine_num_group = 4
refine_kflen = 12
# kf_VLM is not necessary.
refine_output_path = f"{base_dir}/kf_VLM/refine/" + \
    kf_answer_path.split('/')[-1]


class loading_img(Dataset):
    def __init__(self, img_list, preprocess):
        self.img_list = img_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # 保证 RGB 格式
            img = img.copy()  # 强制载入图像到内存，释放文件句柄
        return self.preprocess(img)


# select frames
@torch.inference_mode()
def select_frames(folder, preprocess, resnet18_pretrained, device, batch_size, frame_interval):
    # for folder in folder_list:
    img_list = natsort.natsorted(glob.glob(f"{folder}/*.jpg"))
    # 每隔 frame_interval 帧采样
    # frame_interval = 15
    img_list = img_list[::frame_interval]
    img_feats = []

    # 旧版本，会导致显存爆炸
    # img_set = loading_img(img_list, preprocess)
    # img_loader = DataLoader(img_set, batch_size=batch_size,
    #                         shuffle=False, num_workers=16)
    # for imgtensor in img_loader:
    #     img_feats.append(imgtensor)
    # img_feats = torch.concat(img_feats, dim=0).to(device)

    # 新版本，修改为batch的方式
    img_set = loading_img(img_list, preprocess)
    img_loader = DataLoader(img_set, batch_size=batch_size,
                            shuffle=False, num_workers=8)
    features = []
    with torch.no_grad():
        for imgtensor in img_loader:
            imgtensor = imgtensor.to(device)
            feats = resnet18_pretrained(imgtensor)
            features.append(feats.cpu())  # 可直接保存在CPU，避免占GPU
    featuremap = torch.cat(features, dim=0)

    with torch.no_grad():
        # 旧版本
        # featuremap = resnet18_pretrained(img_feats)
        frame_num = featuremap.shape[0]

        # 旧版本，一个一个计算距离，太慢了
        dist_list = []
        for img_feat in featuremap:
            dist_list.append(torch.mean(torch.sqrt(
                (featuremap-img_feat)**2), dim=-1))
        dist_list = torch.concat(dist_list).reshape(frame_num, frame_num)

        # 新版本，使用cdist计算距离
        # with torch.no_grad():
        #     dist_list = torch.cdist(featuremap, featuremap, p=2)

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


def temporal_scene_clustering_org():
    # Init
    random.seed(10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet18_pretrained = models.resnet18(pretrained=True).to(device)
    resnet18_pretrained.fc = torch.nn.Identity()
    resnet18_pretrained.avgpool = torch.nn.Identity()
    resnet18_pretrained.eval()

    model, preprocess = clip.load("ViT-B/32", device=device)

    objs_acts = f_path
    questions = q_path

    questions = [json.loads(q)
                 for q in open(os.path.expanduser(questions), "r")]
    objs_acts = [json.loads(q)
                 for q in open(os.path.expanduser(objs_acts), "r")]

    answer_path = os.path.expanduser(a_path)
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)
    ans_file = open(answer_path, "w")

    output_results = []
    for question in tqdm.tqdm(questions):
        test_token = True

        for objs_act in objs_acts:
            if objs_act['q_uid'] == question['q_uid']:
                question['Object'] = objs_act["Activity"]
                question['Activity'] = objs_act["Activity"]

                folder_list = glob.glob(
                    f"{img_folder}/{question['q_uid']}/")
                out_frames, output_result = select_frames(
                    folder_list, preprocess, resnet18_pretrained, device, batch_size)
                output_results.append(output_result)
                question['filepath'] = out_frames

                ans_file.write(json.dumps(question) + "\n")
                test_token = False
                break


@torch.inference_mode()
def temporal_scene_clustering_used(frames_dir, output_dir, batch_size, dataset_name, frame_interval):
    # Init
    random.seed(10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet18_pretrained = models.resnet18(pretrained=True).to(device)
    resnet18_pretrained.fc = torch.nn.Identity()
    resnet18_pretrained.avgpool = torch.nn.Identity()
    resnet18_pretrained.eval()
    model, preprocess = clip.load("ViT-B/32", device=device)

    output_results = {}
    for video_name in tqdm(os.listdir(frames_dir), desc="Processing NFS for videos"):
        video_frames_dir = os.path.join(frames_dir, video_name)
        out_frames, output_result = select_frames(
            video_frames_dir, preprocess, resnet18_pretrained, device, batch_size, frame_interval)
        output_results[video_name] = {
            "out_frames": out_frames,
            "output_result": output_result,
            "num_frame_picks": len(out_frames)
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
    # 保存为json
    # with open(f"{output_dir}/nfs_output.json", "w") as f:
    #     json.dump(output_results, f, indent=4)
    with open(f"{output_dir}/{dataset_name}_nfs_output.json", "w") as f:
        json.dump(convert_to_builtin_type(output_results), f, indent=4)
        
    # 释放显存
    del out_frames, output_result
    torch.cuda.empty_cache()


def nfs_by_literature_01():
    """
    来自文献
    Too Many Frames, Not All Useful: Efficient Strategies for Long-Form Video QA
    """
    pass


def nfs_by_bro():
    """
    来自Bro
    """
    pass
