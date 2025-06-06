import json

def verify_json_lengths(json_file_path):
    """
    Verifies that for each video in the JSON file, the lengths of 
    'sampled_indices' and 'scores' arrays are consistent, and that all scores
    are valid numbers within the [0, 1] range.

    Args:
        json_file_path (str): The path to the JSON file.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    all_consistent = True
    if 'datasets' in data:
        for dataset_name, dataset_data in data['datasets'].items():
            print(f"Verifying dataset: {dataset_name}")
            if 'videos' in dataset_data:
                for video_name, video_data in dataset_data['videos'].items():
                    if 'sampled_indices' in video_data and 'scores' in video_data:
                        len_indices = len(video_data['sampled_indices'])
                        len_scores = len(video_data['scores'])
                        if len_indices == len_scores:
                            print(f"  Video '{video_name}': Lengths are consistent ({len_indices}).")
                            # Verify individual scores
                            for i, score in enumerate(video_data['scores']):
                                if not isinstance(score, (int, float)):
                                    print(f"  Video '{video_name}': Score at index {i} is not a number (value: {score}).")
                                    all_consistent = False
                                elif not (0 <= score <= 1):
                                    print(f"  Video '{video_name}': Score at index {i} is out of range [0, 1] (value: {score}).")
                                    all_consistent = False
                        else:
                            print(f"  Video '{video_name}': Lengths are INCONSISTENT. "
                                  f"sampled_indices: {len_indices}, scores: {len_scores}")
                            all_consistent = False
                    else:
                        print(f"  Video '{video_name}': Missing 'sampled_indices' or 'scores' field.")
                        all_consistent = False # Or handle as a different kind of issue
            else:
                print(f"Dataset '{dataset_name}' has no 'videos' field.")
    else:
        print("JSON file does not have a 'datasets' field.")
        all_consistent = False

    if all_consistent:
        print("\nAll videos checked have consistent lengths for 'sampled_indices' and 'scores', and all scores are valid (numeric, between 0 and 1).")
    else:
        print("\nFound inconsistencies, missing fields, or invalid scores in one or more videos.")

if __name__ == '__main__':
    # Path to the JSON file to verify
    file_path = '/root/tfnet/logs/pipeline_01/version_1/results/step1_results.json' 
    print(f"Starting verification for: {file_path}")
    verify_json_lengths(file_path)
