import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a video file and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
        frame_interval (int): Interval at which to extract frames (e.g., 1 for every frame, 10 for every 10th frame).
                              Defaults to 1.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0
    success = True

    while success:
        success, frame = video_capture.read()
        if success:
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1
            frame_count += 1
        else:
            # This handles the case where read() fails at the end of the video
            # or due to an actual error.
            if frame_count == 0: # No frames were read at all
                 print(f"Error: No frames could be read from video {video_path}. It might be corrupted or an unsupported format.")
            break # Exit loop if no frame is read or at the end of the video

    video_capture.release()

def extract_frames_from_directory(video_dir, output_folder, frame_interval=1):
    """
    Extracts frames from all video files in a directory and saves them as images.
    Each video's frames are saved in a separate subfolder named after the video file.

    Args:
        video_dir (str): Path to the directory containing video files.
        output_folder (str): Path to the folder where frame subfolders will be created.
        frame_interval (int): Interval at which to extract frames (e.g., 1 for every frame, 10 for every 10th frame).
                              Defaults to 1.
    """
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found at {video_dir}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    # Find all video files in the directory
    video_files = []
    for file in os.listdir(video_dir):
        if os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(file)
    
    # Process each video file with progress bar
    for i, video_file in enumerate(tqdm(video_files, desc="Extract frames"), 1):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # Remove extension
        video_output_folder = os.path.join(output_folder, video_name)
        # Extract frames for this video
        extract_frames(video_path, video_output_folder, frame_interval)

if __name__ == '__main__':
    # Example usage (optional, for testing the function directly)
    # Create a dummy video file for testing if you don't have one.
    # For example, using ffmpeg: ffmpeg -f lavfi -i testsrc=duration=5:size=1280x720:rate=30 test_video.mp4
    
    # Test single video extraction
    test_video_path = "test_video.mp4"
    test_output_folder = "extracted_frames"
    
    if os.path.exists(test_video_path):
        print(f"Testing single video extraction from '{test_video_path}' to '{test_output_folder}'")
        extract_frames(test_video_path, test_output_folder, frame_interval=30)
    else:
        print(f"Test video '{test_video_path}' not found. Skipping single video test.")
    
    # Test directory extraction
    test_video_dir = "videos"  # Directory containing multiple videos
    test_batch_output = "batch_extracted_frames"
    
    if os.path.exists(test_video_dir):
        print(f"\nTesting batch video extraction from '{test_video_dir}' to '{test_batch_output}'")
        extract_frames_from_directory(test_video_dir, test_batch_output, frame_interval=30)
    else:
        print(f"Test video directory '{test_video_dir}' not found. Skipping batch test.")



def main():
    """Main function to extract frames from all videos in a directory."""
    extract_frames_from_directory(
        video_dir='/root/autodl-tmp/data/SumMe/videos',
        output_folder='/root/autodl-tmp/data/SumMe/frames',
        frame_interval=1
    )
    
    extract_frames_from_directory(
        video_dir='/root/autodl-tmp/data/TVSum/videos',
        output_folder='/root/autodl-tmp/data/TVSum/frames',
        frame_interval=1
    )
    

if __name__ == '__main__':
    main()