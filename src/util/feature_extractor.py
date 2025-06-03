import os
import glob
import numpy as np
import torch
from PIL import Image
import cv2 # For video processing
from transformers import Blip2Processor, Blip2Model
import tempfile # For temporary directory
import shutil # For removing directory tree
from tqdm import tqdm # 导入 tqdm

# Import from sibling module util
from .video_utils import extract_frames as util_extract_frames

class BLIP2FeatureExtractor:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        """
        Initializes the BLIP-2 feature extractor.

        Args:
            model_name (str): The name of the pre-trained BLIP-2 model from Hugging Face.
            device (str, optional): The device to run the model on ('cuda', 'cpu'). 
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading BLIP-2 model: {model_name} on device: {self.device}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set model to evaluation mode

    @torch.no_grad()
    def extract_visual_features(self, video_path_or_folder, output_folder, frame_skip=15):
        """
        Extracts visual features from a single video or a folder of videos
        using video_utils.extract_frames.

        Args:
            video_path_or_folder (str): Path to a single video file or a folder containing videos.
            output_folder (str): Folder to save the extracted .npy features.
            frame_skip (int): Interval for frame extraction passed to video_utils.extract_frames.
        """
        os.makedirs(output_folder, exist_ok=True)

        if os.path.isfile(video_path_or_folder):
            video_paths = [video_path_or_folder]
        elif os.path.isdir(video_path_or_folder):
            supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            video_paths = []
            for ext in supported_extensions:
                video_paths.extend(glob.glob(os.path.join(video_path_or_folder, f"*{ext}")))
                video_paths.extend(glob.glob(os.path.join(video_path_or_folder, f"*{ext.upper()}")))
            video_paths = sorted(list(set(video_paths)))
        else:
            print(f"Error: {video_path_or_folder} is not a valid file or directory.")
            return

        if not video_paths:
            print(f"No video files found in {video_path_or_folder}.")
            return

        for video_path in video_paths:
            print(f"Processing video: {video_path}")
            temp_frame_dir = None # Initialize to ensure it's cleaned up
            try:
                # Create a temporary directory to store frames
                temp_frame_dir = tempfile.mkdtemp(prefix="blip2_frames_")
                print(f"Created temporary frame directory: {temp_frame_dir}")

                # Use video_utils.extract_frames
                # Note: video_utils.extract_frames saves frames as images, not returns PIL objects
                util_extract_frames(video_path, temp_frame_dir, frame_interval=frame_skip)

                # List the extracted frame images
                extracted_frame_files = sorted(glob.glob(os.path.join(temp_frame_dir, "*.jpg"))) # Assuming .jpg, adjust if different
                
                if not extracted_frame_files:
                    print(f"No frames extracted by video_utils.extract_frames for {video_path} into {temp_frame_dir}. Skipping.")
                    continue

                video_features_list = []
                for frame_file_path in extracted_frame_files:
                    frame_pil = Image.open(frame_file_path).convert("RGB")
                    inputs = self.processor(images=frame_pil, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**inputs) 
                    # 正确处理 BLIP-2 模型输出
                    if hasattr(image_features, 'last_hidden_state'):
                        features = image_features.last_hidden_state.squeeze(0).cpu().numpy()
                    elif hasattr(image_features, 'pooler_output'):
                        features = image_features.pooler_output.squeeze(0).cpu().numpy()
                    else:
                        # 如果是张量，直接处理
                        features = image_features.squeeze(0).cpu().numpy()
                    video_features_list.append(features)
                
                if not video_features_list:
                    print(f"No features extracted for video {video_path} after processing frames. Skipping.")
                    continue

                final_video_features = np.array(video_features_list)
                
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_folder, f"{base_name}.npy")
                np.save(output_path, final_video_features)
                print(f"Saved visual features for {video_path} to {output_path} with shape {final_video_features.shape}")

            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
            finally:
                # Clean up the temporary directory
                if temp_frame_dir and os.path.exists(temp_frame_dir):
                    shutil.rmtree(temp_frame_dir)
                    print(f"Cleaned up temporary frame directory: {temp_frame_dir}")

    @torch.no_grad()
    def extract_textual_features(self, text_path_or_folder, output_folder):
        """
        Extracts textual features from a single .txt file or a folder of .txt files.

        Args:
            text_path_or_folder (str): Path to a single .txt file or a folder containing .txt files.
            output_folder (str): Folder to save the extracted .npy features.
        """
        os.makedirs(output_folder, exist_ok=True)

        if os.path.isfile(text_path_or_folder):
            if not text_path_or_folder.lower().endswith(".txt"):
                print(f"Error: {text_path_or_folder} is not a .txt file.")
                return
            text_paths = [text_path_or_folder]
        elif os.path.isdir(text_path_or_folder):
            text_paths = glob.glob(os.path.join(text_path_or_folder, "*.txt"))
            text_paths.extend(glob.glob(os.path.join(text_path_or_folder, "*.TXT"))) # Case-insensitive
            text_paths = sorted(list(set(text_paths))) # Remove duplicates and sort
        else:
            print(f"Error: {text_path_or_folder} is not a valid file or directory.")
            return

        if not text_paths:
            print(f"No .txt files found in {text_path_or_folder}.")
            return

        for text_path in tqdm(text_paths, desc="提取文本特征"):
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            if not text_content:
                continue

            inputs = self.processor(text=text_content, return_tensors="pt").to(self.device)
            text_features = self.model.get_text_features(**inputs)
            
            # 正确处理 BLIP-2 模型输出
            if hasattr(text_features, 'last_hidden_state'):
                features = text_features.last_hidden_state.squeeze(0).cpu().numpy()
            elif hasattr(text_features, 'pooler_output'):
                features = text_features.pooler_output.squeeze(0).cpu().numpy()
            else:
                # 如果是张量，直接处理
                features = text_features.squeeze(0).cpu().numpy()

            base_name = os.path.splitext(os.path.basename(text_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}.npy")
            np.save(output_path, features)

    @torch.no_grad()
    def extract_visual_features_from_frames(self, frames_folder, output_folder, interval=15):
        """
        从已提取的帧文件夹中提取视觉特征，使用经过 language_projection 的特征以与文本特征对齐。
        
        Args:
            frames_folder (str): 包含已提取帧的文件夹路径
            output_folder (str): 保存提取特征的输出文件夹
            interval (int): 帧间隔，每隔多少帧提取一次特征
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 查找所有视频的帧文件夹
        video_frame_dirs = [d for d in os.listdir(frames_folder) if os.path.isdir(os.path.join(frames_folder, d))]
        
        if not video_frame_dirs:
            print(f"在 {frames_folder} 中未找到视频帧文件夹。")
            return
            
        for video_name in tqdm(video_frame_dirs, desc="提取视觉特征"):
            video_frame_dir = os.path.join(frames_folder, video_name)
            
            # 获取所有帧文件并排序
            frame_files = sorted([f for f in os.listdir(video_frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if not frame_files:
                continue
            
            # 根据间隔选择帧
            selected_frames = frame_files[::interval]  # 每隔 interval 帧选择一帧
            
            video_features_list = []
            for frame_file in tqdm(selected_frames, desc=f"{video_name}", leave=False):
                frame_path = os.path.join(video_frame_dir, frame_file)
                frame_pil = Image.open(frame_path).convert("RGB")
                inputs = self.processor(images=frame_pil, return_tensors="pt").to(self.device)
                
                # 通过完整的 BLIP-2 pipeline 获取经过 projection 的视觉特征
                # 1. 视觉编码器
                vision_outputs = self.model.vision_model(**inputs)
                image_embeds = vision_outputs.last_hidden_state
                
                # 2. QFormer 处理
                image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_outputs = self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                
                # 3. Language projection - 将特征投影到与文本相同的空间 (2560维)
                projected_features = self.model.language_projection(query_outputs.last_hidden_state)
                
                # 输出形状: (1, 32, 2560) -> (32, 2560) 
                features = projected_features.squeeze(0).cpu().numpy()
                video_features_list.append(features)
            
            if not video_features_list:
                continue
            
            # 保存所有帧的特征：形状为 (num_frames, num_tokens, feature_dim)
            final_video_features = np.array(video_features_list)
            output_path = os.path.join(output_folder, f"{video_name}.npy")
            np.save(output_path, final_video_features)
            
            print(f"已保存视觉特征: {video_name}.npy, 形状: {final_video_features.shape}")

    @torch.no_grad()
    def extract_caption_features(self, captions_folder, output_folder):
        """
        从字幕文件夹中提取文本特征。
        
        Args:
            captions_folder (str): 包含字幕文件的文件夹路径
            output_folder (str): 保存提取特征的输出文件夹
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 查找所有txt字幕文件
        caption_files = []
        for file in os.listdir(captions_folder):
            if file.lower().endswith('.txt'):
                caption_files.append(os.path.join(captions_folder, file))
        
        if not caption_files:
            print(f"在 {captions_folder} 中未找到 .txt 字幕文件")
            return
        
        caption_files = sorted(caption_files)
        print(f"找到 {len(caption_files)} 个字幕文件")
        
        # 处理每个字幕文件
        for caption_file in tqdm(caption_files, desc="提取字幕特征"):
            try:
                # 读取字幕内容
                with open(caption_file, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
                
                if not caption_text:
                    print(f"字幕文件 {caption_file} 为空，跳过")
                    continue
                
                # 处理文本输入 - 使用 BLIP-2 的语言模型
                inputs = self.processor(text=caption_text, return_tensors="pt").to(self.device)
                
                # 使用 BLIP-2 模型的语言模型获取文本特征
                with torch.no_grad():
                    text_outputs = self.model.language_model(**inputs, output_hidden_states=True)
                
                # 从语言模型输出中提取特征，保持 (N, D) 的形状
                if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states:
                    # 使用最后一层的隐藏状态，保持 (N, D) 形状
                    last_hidden_state = text_outputs.hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)
                    features = last_hidden_state.squeeze(0).cpu().numpy()  # shape: (seq_len, hidden_dim)
                elif hasattr(text_outputs, 'last_hidden_state'):
                    # 如果没有 hidden_states，使用 last_hidden_state
                    features = text_outputs.last_hidden_state.squeeze(0).cpu().numpy()  # shape: (seq_len, hidden_dim)
                else:
                    # 如果以上都没有，尝试直接处理输出
                    print(f"警告: 无法识别的输出格式: {type(text_outputs)}")
                    continue
                
                # 保存特征
                base_name = os.path.splitext(os.path.basename(caption_file))[0]
                output_path = os.path.join(output_folder, f"{base_name}.npy")
                np.save(output_path, features)
                
                print(f"已保存字幕特征: {base_name}.npy, 形状: {features.shape}")
                
            except Exception as e:
                print(f"处理字幕文件 {caption_file} 时出错: {e}")
                continue

if __name__ == '__main__':
    # This is an example of how to use the extractor.
    # Users should adapt this to their specific file paths and needs.
    
    print("Starting BLIP-2 Feature Extractor example...")
    
    # --- IMPORTANT ---
    # 1. Ensure you have installed all necessary packages:
    #    pip install transformers torch Pillow opencv-python numpy sentencepiece
    # 2. The first time you run this, the BLIP-2 model (around 5-15GB depending on version) 
    #    will be downloaded. This requires a stable internet connection and sufficient disk space.
    # 3. Processing videos can be computationally intensive.
    
    try:
        # You can specify a different BLIP-2 model if needed, e.g., "Salesforce/blip2-flan-t5-xl"
        # Smaller models like "Salesforce/blip2-opt-2.7b" (default) are faster but might be less accurate.
        # Larger models like "Salesforce/blip2-opt-6.7b" or "Salesforce/blip2-flan-t5-xxl" require more resources.
        extractor = BLIP2FeatureExtractor(model_name="Salesforce/blip2-opt-2.7b") 
    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        print("Please check error messages and ensure model/dependencies are correctly set up.")
        print("Skipping example usage.")
        exit()

    # --- Example for Visual Features ---
    print("\n--- Visual Feature Extraction Example ---")
    example_video_folder = "example_videos_for_blip2" # Create this folder
    example_video_output_folder = "output_visual_features_blip2" # Will be created
    os.makedirs(example_video_folder, exist_ok=True)
    
    # Create a dummy MP4 file path. 
    # For a real test, place an actual small .mp4 video file here.
    dummy_video_path = os.path.join(example_video_folder, "sample_video.mp4")
    
    # To test, create a small dummy video file (e.g., using ffmpeg) or place one manually.
    # For example, using ffmpeg:
    # ffmpeg -f lavfi -i testsrc=duration=2:size=128x72:rate=10 -c:v libx264 -t 2 example_videos_for_blip2/sample_video.mp4
    if not os.path.exists(dummy_video_path):
        print(f"INFO: Dummy video {dummy_video_path} not found.")
        print(f"Please place a sample video (e.g., sample_video.mp4) in '{example_video_folder}' to test visual feature extraction.")
        print("You can create a test video using ffmpeg: ffmpeg -f lavfi -i testsrc=duration=2:size=128x72:rate=10 -c:v libx264 -t 2 example_videos_for_blip2/sample_video.mp4")
    else:
        if os.path.getsize(dummy_video_path) > 0:
            print(f"Attempting to process dummy video: {dummy_video_path}")
            extractor.extract_visual_features(dummy_video_path, example_video_output_folder, frame_skip=10) # Reduced skip for small video
            
            # Example for processing a folder of videos (if you have more in example_video_folder)
            # print(f"\nAttempting to process video folder: {example_video_folder}")
            # extractor.extract_visual_features(example_video_folder, example_video_output_folder, frame_skip=15)
        else:
            print(f"Skipping visual feature extraction for single file as {dummy_video_path} is empty.")


    # --- Example for Textual Features ---
    print("\n--- Textual Feature Extraction Example ---")
    example_text_folder = "example_texts_for_blip2" # Create this folder
    example_text_output_folder = "output_textual_features_blip2" # Will be created
    os.makedirs(example_text_folder, exist_ok=True)
    
    dummy_text_path1 = os.path.join(example_text_folder, "sample1.txt")
    with open(dummy_text_path1, 'w', encoding='utf-8') as f:
        f.write("A beautiful landscape with mountains and a lake.")
    
    dummy_text_path2 = os.path.join(example_text_folder, "another_sample.txt")
    with open(dummy_text_path2, 'w', encoding='utf-8') as f:
        f.write("A futuristic city skyline at sunset.")

    print(f"Processing dummy text file: {dummy_text_path1}")
    extractor.extract_textual_features(dummy_text_path1, example_text_output_folder)
    
    print(f"\nProcessing dummy text folder: {example_text_folder}")
    extractor.extract_textual_features(example_text_folder, example_text_output_folder)

    print("\nExample usage finished.")
    print(f"Check '{example_video_output_folder}' for visual features (if a video was processed).")
    print(f"Check '{example_text_output_folder}' for textual features.")
    print("Remember to clean up dummy folders/files if needed (e.g., example_videos_for_blip2, output_visual_features_blip2, etc.).")
