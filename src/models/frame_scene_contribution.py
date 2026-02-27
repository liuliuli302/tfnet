from src.config.config import BasicConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from typing import List
from dataclasses import dataclass
import json
import os


@dataclass
class FrameSceneContributionConfig(BasicConfig):
    datasets: List[str]
    frame_caption_dir: str
    scene_caption_dir: str
    output_dir: str

    text_model_name: str = "Qwen/Qwen3-Embedding-8B"
    max_length: int = 512


class FrameSceneContribution:
    def __init__(self, config: FrameSceneContributionConfig):
        self.config = config
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, but GPU is required for this pipeline.")
        self.device = "cuda"
        self.encoder = SentenceTransformer(
            config.text_model_name, trust_remote_code=True, device=self.device)

    def _get_frame_scene_contribution(self):
        os.makedirs(self.config.output_dir, exist_ok=True)

        for d in tqdm(self.config.datasets, desc="Datasets", leave=True):
            fc = json.load(
                open(f"{self.config.frame_caption_dir}/{d}_frame_captions.json"))
            sc = json.load(
                open(f"{self.config.scene_caption_dir}/{d}_scene_captions.json"))

            out = {}

            for vid in tqdm(sc, desc=f"Videos[{d}]", leave=False):
                if vid not in fc:
                    continue

                picks = fc[vid]["picks"]
                caps = fc[vid]["captions"]
                pick2cap = dict(zip(picks, caps))

                video_res = []

                for scene in tqdm(sc[vid].values(), desc=f"Scenes[{vid}]", leave=False):
                    scene_cap = scene["scene_caption"]
                    scene_emb = self.encoder.encode(
                        scene_cap,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                    )
                    if scene_emb is None:
                        continue

                    frames = []
                    for p in scene["picks"]:
                        cap = pick2cap.get(p)
                        if not cap:
                            continue
                        emb = self.encoder.encode(
                            cap,
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                        )
                        if emb is None:
                            continue
                        sim = float(self.encoder.similarity(
                            emb, scene_emb).item())
                        frames.append({"pick": p, "sim": round(sim, 6)})

                    video_res.append({
                        "scene_caption": scene_cap,
                        "frames": frames
                    })

                out[vid] = video_res

            json.dump(
                out, open(f"{self.config.output_dir}/{d}.json", "w"), indent=2)


if __name__ == "__main__":
    cfg = FrameSceneContributionConfig(
        datasets=["summe"],
        frame_caption_dir="/root/tfnet/data/captions/frame_caption/blip",
        scene_caption_dir="/root/tfnet/data/captions/scene_caption/gpt5",
        output_dir="/root/tfnet/data/scores/frame_scene_contribution",
    )
    FrameSceneContribution(cfg)._get_frame_scene_contribution()

    cfg = FrameSceneContributionConfig(
        datasets=["tvsum"],
        frame_caption_dir="/root/tfnet/data/captions/frame_caption/blip",
        scene_caption_dir="/root/tfnet/data/captions/scene_caption/gpt5",
        output_dir="/root/tfnet/data/scores/frame_scene_contribution",
    )
    FrameSceneContribution(cfg)._get_frame_scene_contribution()
