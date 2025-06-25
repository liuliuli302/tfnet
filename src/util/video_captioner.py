import os
import abc
import cv2
import base64
import time
import argparse
from typing import List
from tqdm import tqdm
from openai import OpenAI
import google.generativeai as genai
import torch
import copy
import numpy as np
from transformers import AutoTokenizer, AutoProcessor
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from decord import VideoReader, cpu

class VideoCaptioner(abc.ABC):
    """Abstract base class for video captioning models."""

    @abc.abstractmethod
    def caption_video(self, video_path: str) -> str:
        pass

    def caption_videos_from_directory(self, video_dir: str, output_dir: str, 
                                    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        os.makedirs(output_dir, exist_ok=True)
        video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                      if f.lower().endswith(video_extensions)]
        
        for video_path in tqdm(video_files, desc="Captioning"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            caption = self.caption_video(video_path)
            output_path = os.path.join(output_dir, f"{video_name}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(caption)

class GPT4oCaptioner(VideoCaptioner):
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def extract_frames(self, video_path: str, max_frames: int = 10) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_b64)
        
        cap.release()
        return frames

    def caption_video(self, video_path: str) -> str:
        frames = self.extract_frames(video_path)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "Please analyze these video frames and provide a detailed description."}]
        }]
        
        for frame_b64 in frames:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })
        
        response = self.client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=500
        )
        return response.choices[0].message.content

class Gemini25ProCaptioner(VideoCaptioner):
    def __init__(self, api_key: str = None):
        genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def caption_video(self, video_path: str) -> str:
        video_file = genai.upload_file(path=video_path)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(5)
            video_file = genai.get_file(video_file.name)
        
        prompt = "Please analyze this video and provide a detailed description."
        response = self.model.generate_content([prompt, video_file])
        genai.delete_file(video_file.name)
        
        return response.text

class LLaVAVideoCaptioner(VideoCaptioner):
    def __init__(self, model_path: str = None, max_frames_num: int = 64, conv_template: str = "qwen_1_5"):
        self.model_path = model_path or "lmms-lab/LLaVA-Video-7B-Qwen2"
        self.max_frames_num = max_frames_num
        self.conv_template = conv_template
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.device = None

    def load_video(self, video_path: str):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()
        video_time = total_frame_num / video_fps
        
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / video_fps for i in frame_idx]
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames, frame_time_str, video_time

    def load_model(self):
        if self.model is not None:
            return
        
        # Check CUDA compatibility
        device_map = "auto"
        torch_dtype = "bfloat16"
            
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path, None, "llava_qwen", torch_dtype=torch_dtype, device_map=device_map
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on device: {self.device}")

    def caption_video(self, video_path: str, custom_prompt: str = None) -> str:
        if self.model is None:
            self.load_model()
        
        video, frame_time, video_time = self.load_video(video_path)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        
        # Move video to the same device as model
        if hasattr(self, 'device'):
            video = video.to(self.device)
            if self.device.type == 'cuda':
                video = video.half()
        elif torch.cuda.is_available():
            video = video.cuda().half()
        
        time_instruction = f"The video lasts for {video_time:.2f} seconds, with frames at {frame_time}."
        question_text = custom_prompt or "Please describe this video in detail."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{question_text}"
        
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        
        # Move input_ids to the same device as model
        if hasattr(self, 'device'):
            input_ids = input_ids.to(self.device)
        elif torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        with torch.no_grad():
            cont = self.model.generate(input_ids, images=[video], modalities=["video"], 
                                     do_sample=False, temperature=0, max_new_tokens=4096)
        
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

CAPTIONER_MODELS = {
    "gpt4o": GPT4oCaptioner,
    "gemini25pro": Gemini25ProCaptioner,
    "llavavideo": LLaVAVideoCaptioner,
}

def get_captioner(model_name: str, **kwargs) -> VideoCaptioner:
    model_name_lower = model_name.lower()
    captioner_class = CAPTIONER_MODELS.get(model_name_lower)
    if not captioner_class:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(CAPTIONER_MODELS.keys())}")
    return captioner_class(**kwargs)

def caption_videos(input_path: str, model_name: str, output_dir: str, **model_kwargs):
    captioner = get_captioner(model_name, **model_kwargs)
    
    if os.path.isfile(input_path):
        caption = captioner.caption_video(input_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(caption)
    elif os.path.isdir(input_path):
        captioner.caption_videos_from_directory(input_path, output_dir)
    else:
        print(f"Error: {input_path} is not a valid file or directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for videos from a file or folder.")
    parser.add_argument("input_path", type=str, help="Path to the video file or folder containing video files.")
    parser.add_argument("--model", type=str, default="gpt4o",
                        choices=list(CAPTIONER_MODELS.keys()),
                        help="Captioning model to use (default: gpt4o).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save caption files. A subfolder named after the model will be created here.")
    # Add model-specific arguments if needed, e.g., API keys or model paths if not using env vars
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key (overrides OPENAI_API_KEY env var).")
    parser.add_argument("--gemini_api_key", type=str, help="Gemini API key (overrides GEMINI_API_KEY env var).")
    parser.add_argument("--llava_model_path", type=str, help="Path to the local LLaVA-Video model.")

    args = parser.parse_args()

    model_specific_kwargs = {}
    if args.model.lower() == "gpt4o" and args.openai_api_key:
        model_specific_kwargs['api_key'] = args.openai_api_key
    elif args.model.lower() == "gemini25pro" and args.gemini_api_key:
        model_specific_kwargs['api_key'] = args.gemini_api_key
    elif args.model.lower() == "llavavideo" and args.llava_model_path:
        model_specific_kwargs['model_path'] = args.llava_model_path

    caption_videos(args.input_path, args.model, args.output_dir, **model_specific_kwargs)

    print("\\n---")
    script_name = os.path.basename(__file__)
    print("To run this script (example):")
    print(f"python {script_name} /path/to/your/video.mp4 --model gpt4o --output_dir /path/to/output_captions")
    print(f"python {script_name} /path/to/your/video_folder --model gemini25pro --gemini_api_key YOUR_API_KEY --output_dir /path/to/output_captions")
    print(f"python {script_name} /path/to/your/video_folder --model llavavideo --llava_model_path /path/to/llava/model --output_dir /path/to/output_captions")
    print("---")
    print("Installation requirements:")
    print("pip install opencv-python openai google-generativeai")
    print("\\nAPI Key setup:")
    print("export OPENAI_API_KEY='your-openai-key'")
    print("export GEMINI_API_KEY='your-gemini-key'")
    print("\\nNote: LLaVA-Video implementation is simplified. For full functionality, install:")
    print("pip install torch transformers accelerate decord")
