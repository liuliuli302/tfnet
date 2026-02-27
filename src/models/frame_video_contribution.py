from src.config.config import BasicConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from typing import List
from dataclasses import dataclass
import json
import os


@dataclass
class FrameVideoContributionConfig(BasicConfig):
    datasets: List[str]
    frame_caption_dir: str
    video_caption_dir: str
    output_dir: str

    text_model_name: str = "Qwen/Qwen3-Embedding-8B"
    max_length: int = 512
    resize_short_side: int = 256


class FrameVideoContribution:
    def __init__(self, config: FrameVideoContributionConfig):
        self.config = config
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, but GPU is required for this pipeline.")
        self.device = "cuda"
        self.encoder = SentenceTransformer(
            config.text_model_name, trust_remote_code=True, device=self.device)

    def _get_frame_video_contribution(self):
        os.makedirs(self.config.output_dir, exist_ok=True)

        for d in tqdm(self.config.datasets, desc="Datasets", leave=True):
            fc = json.load(
                open(f"{self.config.frame_caption_dir}/{d}_frame_captions.json"))
            vc = json.load(
                open(f"{self.config.video_caption_dir}/{d}_video_captions.json"))

            out = {}

            for vid in tqdm(fc, desc=f"Videos[{d}]", leave=False):
                if vid not in vc:
                    continue

                picks = fc[vid]["picks"]
                caps = fc[vid]["captions"]
                pick2cap = dict(zip(picks, caps))

                idxs = sorted(set(pick2cap.keys()))

                # ===== text embedding =====
                video_emb = self.encoder.encode(
                    vc[vid],
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                if video_emb is None:
                    continue

                frames = []
                for i in tqdm(idxs, desc=f"Frames[{vid}]", leave=False):
                    cap = pick2cap[i]
                    emb = self.encoder.encode(
                        cap,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                    )
                    if emb is None:
                        continue

                    sim = float(self.encoder.similarity(emb, video_emb).item())

                    frames.append({
                        "pick": i,
                        "text_sim": round(sim, 6),
                    })

                out[vid] = {
                    "video_caption": vc[vid],
                    "frames": frames
                }

            json.dump(
                out, open(f"{self.config.output_dir}/{d}.json", "w"), indent=2)


if __name__ == "__main__":
    cfg = FrameVideoContributionConfig(
        datasets=["summe", "tvsum"],
        frame_caption_dir="/root/tfnet/data/captions/frame_caption/blip",
        video_caption_dir="/root/tfnet/data/captions/video_caption/gpt5",
        output_dir="/root/tfnet/data/scores/frame_video_contribution",
    )
    FrameVideoContribution(cfg)._get_frame_video_contribution()
