from dataclasses import dataclass
import json
import os
import time
from typing import Dict, Any
from tqdm import tqdm
from src.config.config import BasicConfig
from utils.llm_client import LLMBaseClient, OpenAIClient


@dataclass
class VideoCaptionSummarizerConfig(BasicConfig):
    input_file: str
    output_file: str
    system_prompt: str = "This content contains descriptive text for the frame images of a video. Please summarize the main content of the video based on the descriptive text in a linear fashion and do not divide the points.",
    max_retries: int = 3
    retry_delay: int = 5


class VideoCaptionSummarizer:
    """
    Generate per-video summaries from frame captions using an LLM client.
    Mirrors the class-based pattern used in frame_caption.
    """

    def __init__(self, llm_client: LLMBaseClient, config: VideoCaptionSummarizerConfig):
        if not isinstance(config, VideoCaptionSummarizerConfig):
            raise TypeError("config must be VideoCaptionSummarizerConfig")
        self.llm_client = llm_client
        self.config = config

    def _load_captions(self) -> Dict[str, Any]:
        with open(self.config.input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_existing_summaries(self) -> Dict[str, Any]:
        if not os.path.exists(self.config.output_file):
            return {}
        with open(self.config.output_file, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        print(
            f"Found existing output file, loaded {len(summaries)} existing summaries")
        return summaries

    def _build_messages(self, captions):
        file_content = "\n".join(captions)
        return [
            {"role": "system", "content": file_content},
            {"role": "user", "content": self.config.system_prompt},
        ]

    def _summarize_single(self, video_name: str, captions, summaries: Dict[str, Any]):
        messages = self._build_messages(captions)
        for attempt in range(self.config.max_retries):
            try:
                summary = self.llm_client.generate({"messages": messages})
                summaries[video_name] = {
                    "summary": summary.strip(), "frame_count": len(captions)}
                self._persist(summaries)
                return
            except Exception as exc:  # retry on failure
                print(
                    f"Error processing video {video_name} (attempt {attempt + 1}/{self.config.max_retries}): {exc}")
                if attempt < self.config.max_retries - 1:
                    print(f"Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
                else:
                    print(
                        f"Failed to process video {video_name} after {self.config.max_retries} attempts")

    def _persist(self, summaries: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.config.output_file), exist_ok=True)
        with open(self.config.output_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    def caption_frames(self):
        caption_data = self._load_captions()
        summaries = self._load_existing_summaries()

        for video_name, video_data in tqdm(caption_data.items(), desc="Generating summary from frame captions"):
            if video_name in summaries:
                continue
            if "captions" not in video_data:
                print(f"Warning: No captions found for video {video_name}")
                continue
            captions = video_data["captions"]
            self._summarize_single(video_name, captions, summaries)

        print(f"Summaries saved to {self.config.output_file}")
        return summaries


if __name__ == "__main__":
    summe_captions_json = "/root/tfnet/data/captions/frame_caption/blip/summe_captions.json"
    tvsum_captions_json = "/root/tfnet/data/captions/frame_caption/blip/tvsum_captions.json"

    out_dir = "/root/tfnet/data/captions/video_caption/gpt-41"
    os.makedirs(out_dir, exist_ok=True)

    summe_summary_json = os.path.join(out_dir, "summe_summary_gpt41.json")
    tvsum_summary_json = os.path.join(out_dir, "tvsum_summary_gpt41.json")

    deepseek_client = OpenAIClient(
        api_key="", base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    moonshot_client = OpenAIClient(
        api_key="", base_url="https://api.moonshot.cn/v1", model="moonshot-v1-128k")
    gpt4_client = OpenAIClient(
        api_key="", base_url="https://api.openai.com/v1", model="gpt-4")

    VideoCaptionSummarizer(gpt4_client, VideoCaptionSummarizerConfig(
        input_file=summe_captions_json, output_file=summe_summary_json)).run()
    VideoCaptionSummarizer(gpt4_client, VideoCaptionSummarizerConfig(
        input_file=tvsum_captions_json, output_file=tvsum_summary_json)).run()
