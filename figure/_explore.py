"""Explore h5 GT data structure — deep dive into SumMe gtscore."""
import h5py
import numpy as np

h5 = h5py.File('/Users/allets/Resources/datasets/eccv16_dataset_summe_google_pool5.h5', 'r')

# Pick a specific video we plotted: playing_ball (top-1 by F1)
target = "playing ball"
vid = None
for k in h5.keys():
    vn = h5[k]['video_name'][()].decode()
    if vn.lower().replace("_", " ") == target:
        vid = h5[k]
        break
if vid is None:
    # fallback: show all names
    for k in h5.keys():
        print(k, "->", h5[k]['video_name'][()].decode())
    raise RuntimeError("not found")

print(f"Video: {vid['video_name'][()].decode()}")
print(f"n_frames: {int(np.asarray(vid['n_frames']))}")
print(f"picks shape: {np.asarray(vid['picks']).shape}")
print(f"picks[:10]: {np.asarray(vid['picks'])[:10]}")
print(f"picks[-5:]: {np.asarray(vid['picks'])[-5:]}")

gt = np.asarray(vid['gtscore'])
print(f"\ngtscore shape: {gt.shape}")
print(f"gtscore dtype: {gt.dtype}")
print(f"gtscore range: [{gt.min():.4f}, {gt.max():.4f}]")
print(f"gtscore unique values: {len(np.unique(gt))}")
print(f"gtscore unique (sorted): {np.sort(np.unique(gt))}")
print(f"gtscore[:20]: {gt[:20]}")
print(f"gtscore histogram:")
for val in np.sort(np.unique(gt)):
    cnt = np.sum(gt == val)
    print(f"  {val:.4f}: {cnt} frames ({cnt/len(gt)*100:.1f}%)")

us = np.asarray(vid['user_summary'])
print(f"\nuser_summary shape: {us.shape}")
print(f"user_summary dtype: {us.dtype}")
print(f"user_summary range: [{us.min()}, {us.max()}]")
print(f"\nHow gtscore is computed (check if it's mean of user_summary per-pick):")
picks = np.asarray(vid['picks'])
# reconstruct: for each pick, compute mean of user_summary at that frame
n_users = us.shape[0]
gt_reconstructed = np.array([us[:, p].mean() for p in picks])
print(f"reconstructed shape: {gt_reconstructed.shape}")
print(f"reconstructed[:20]: {np.round(gt_reconstructed[:20], 4)}")
print(f"Match actual gtscore? max_diff={np.max(np.abs(gt - gt_reconstructed)):.6f}")

print(f"\n--- Also check Excavators river crossing ---")
for k in h5.keys():
    vn = h5[k]['video_name'][()].decode()
    if "xcavator" in vn:
        vid2 = h5[k]
        break
gt2 = np.asarray(vid2['gtscore'])
us2 = np.asarray(vid2['user_summary'])
picks2 = np.asarray(vid2['picks'])
print(f"Video: {vid2['video_name'][()].decode()}")
print(f"n_frames: {int(np.asarray(vid2['n_frames']))}")
print(f"gtscore shape: {gt2.shape}, unique: {len(np.unique(gt2))}")
print(f"gtscore unique: {np.sort(np.unique(gt2))}")
print(f"user_summary shape: {us2.shape}")
gt2_recon = np.array([us2[:, p].mean() for p in picks2])
print(f"Match gtscore? max_diff={np.max(np.abs(gt2 - gt2_recon)):.6f}")

# Also check TVSum for contrast
print(f"\n--- TVSum contrast (video_1) ---")
h5t = h5py.File('/Users/allets/Resources/datasets/eccv16_dataset_tvsum_google_pool5.h5', 'r')
vt = h5t['video_1']
gtt = np.asarray(vt['gtscore'])
print(f"gtscore shape: {gtt.shape}, unique: {len(np.unique(gtt))}")
print(f"gtscore range: [{gtt.min():.4f}, {gtt.max():.4f}]")
print(f"gtscore[:20]: {np.round(gtt[:20], 4)}")

h5.close()
h5t.close()
if 'n_frames' in v0t:
    print('n_frames:', v0t['n_frames'][...])
h5t.close()
