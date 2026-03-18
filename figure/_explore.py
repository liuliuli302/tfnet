"""Exploration helper – run to inspect data & pick best-visualized videos."""
import json, os, numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(rel):
    with open(os.path.join(BASE, rel)) as f:
        return json.load(f)

def compute_final_scores(dataset, alpha=0.2):
    """Return {video_name: (picks, s_llm, s_fs, s_fv, final)} for a dataset."""
    ss = load(f'data/scores/scene_score/deepseek32/{dataset}_scene_scores.json')
    fs = load(f'data/scores/frame_scene_contribution/{dataset}.json')
    fv = load(f'data/scores/frame_video_contribution/{dataset}.json')

    out = {}
    for vname in fs:
        scenes = fs[vname]
        fv_frames = fv[vname]['frames'] if vname in fv else []
        fv_map = {int(f['pick']): float(f['text_sim']) for f in fv_frames}
        scene_scores_raw = ss.get(vname, {})

        picks, s_llm_arr, s_fs_arr, s_fv_arr, final_arr = [], [], [], [], []
        for si, scene in enumerate(scenes):
            s_score = float(scene_scores_raw.get(str(si), '0')) / 100.0
            for frame in scene['frames']:
                p = int(frame['pick'])
                f_s = float(frame['sim'])
                f_v = fv_map.get(p, 0.0)
                final = alpha * s_score * f_s + f_v
                picks.append(p)
                s_llm_arr.append(s_score)
                s_fs_arr.append(f_s)
                s_fv_arr.append(f_v)
                final_arr.append(final)

        out[vname] = (
            np.array(picks), np.array(s_llm_arr), np.array(s_fs_arr),
            np.array(s_fv_arr), np.array(final_arr)
        )
    return out

# Rank videos by score variance (higher variance = more visually interesting)
for ds in ['summe', 'tvsum']:
    print(f'\n=== {ds.upper()} — variance ranking ===')
    scores = compute_final_scores(ds)
    ranked = sorted(scores.items(), key=lambda kv: -np.std(kv[1][4]))
    for vname, (picks, sl, sf, sv, fn) in ranked[:8]:
        print(f'  {vname:30s}  std={np.std(fn):.4f}  '
              f'n_frames={len(picks)}  '
              f'final_range=[{fn.min():.3f}, {fn.max():.3f}]  '
              f's_llm_range=[{sl.min():.2f}, {sl.max():.2f}]')
