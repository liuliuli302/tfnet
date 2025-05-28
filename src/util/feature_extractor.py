import os
import glob
import numpy as np
import torch
from PIL import Image
import cv2 # For video processing
from transformers import Blip2Processor, Blip2Model
import tempfile # For temporary directory
import shutil # For removing directory tree

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
        try:
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2Model.from_pretrained(model_name).to(self.device)
            self.model.eval() # Set model to evaluation mode
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Please ensure you have a working internet connection to download the model,")
            print("or that the model is available in your local Hugging Face cache.")
            print("Required packages: transformers, torch, Pillow, opencv-python, numpy, sentencepiece (for some BLIP-2 models)")
            raise

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
                    try:
                        frame_pil = Image.open(frame_file_path).convert("RGB")
                        inputs = self.processor(images=frame_pil, return_tensors="pt").to(self.device)
                        image_features = self.model.get_image_features(**inputs) 
                        video_features_list.append(image_features.squeeze(0).cpu().numpy())
                    except Exception as e_frame:
                        print(f"Error processing frame {frame_file_path}: {e_frame}")
                        continue # Skip problematic frames
                
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
                    try:
                        shutil.rmtree(temp_frame_dir)
                        print(f"Cleaned up temporary frame directory: {temp_frame_dir}")
                    except Exception as e_clean:
                        print(f"Error cleaning up temporary directory {temp_frame_dir}: {e_clean}")

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

        for text_path in text_paths:
            print(f"Processing text file: {text_path}")
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                if not text_content:
                    print(f"Text file {text_path} is empty. Skipping.")
                    continue

                inputs = self.processor(text=text_content, return_tensors="pt").to(self.device)
                text_features = self.model.get_text_features(**inputs)
                
                final_text_features = text_features.squeeze(0).cpu().numpy()

                base_name = os.path.splitext(os.path.basename(text_path))[0]
                output_path = os.path.join(output_folder, f"{base_name}.npy")
                np.save(output_path, final_text_features)
                print(f"Saved textual features for {text_path} to {output_path} with shape {final_text_features.shape}")

            except Exception as e:
                print(f"Error processing text file {text_path}: {e}")

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
