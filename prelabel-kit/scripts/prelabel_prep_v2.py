# Written by Claude (pre-label v2)
"""v2 prep: v1 events.json + sheets, plus 5 fps frames of both cameras so labellers can look anywhere without video.
usage: prelabel_prep_v2.py DATASET EPS [--root DIR] [--cache DIR] [--out DIR]
Videos are downloaded per file from the public Hub dataset and deleted once their episodes are done (disk is tight)."""
import sys, os, json, argparse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, av
from PIL import Image
import prelabel_prep as P

HUB = "https://huggingface.co/datasets/molnrb/{ds}/resolve/main/{path}"
FPS_OUT = 5
BOWL = (250, 40, 550, 340)


def stripe_score(img):
    # hue 28-44 of 255 is the yellow-green stripe; the orange arm reaches ~31 in highlights, so this is only a hint
    hsv = np.asarray(img.convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return int(((h >= 32) & (h <= 44) & (s > 90) & (v > 90)).sum())


def video_path(row, cam):
    k = f"videos/observation.images.{cam}"
    return f"videos/observation.images.{cam}/chunk-{int(row[k + '/chunk_index']):03d}/file-{int(row[k + '/file_index']):03d}.mp4"


def fetch(ds, rel, cache):
    dst = os.path.join(cache, ds, rel)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urllib.request.urlretrieve(HUB.format(ds=ds, path=rel), dst + ".part")
        os.replace(dst + ".part", dst)
    return dst


def decode_episode(path, t_from, length_s, fps, keep):
    """frames whose index is in `keep`, one sequential decode: {frame_index: PIL}"""
    out = {}
    c = av.open(path); st = c.streams.video[0]; st.thread_type = "AUTO"; tb = float(st.time_base)
    c.seek(int(max(0.0, t_from - 1.0) / tb), stream=st, backward=True, any_frame=False)
    for fr in c.decode(st):
        ft = float(fr.pts * tb) - t_from
        if ft < -1e-3: continue
        if ft > length_s: break
        i = int(round(ft * fps))
        if i in keep: out[i] = fr.to_image()
    c.close()
    return out


def nearest(frames, i):
    k = min(frames, key=lambda j: abs(j - i)); return frames[k]


def prep_episode(ds, ep, row, D, fps, cache, out):
    x = D[D.episode_index == ep].sort_values("frame_index")
    S = np.stack(x["observation.state"].values).astype(float)
    ev, leave, home, _ = P.analyse(S, fps)
    L = len(S) / fps
    od = f"{out}/{ds}/{ep}"; os.makedirs(od + "/f", exist_ok=True)
    times = P.pick_times(ev, L, leave, home)
    keep = set(range(0, len(S), fps // 10)) | {int(round(t * fps)) for _, t in times}
    frames = {cam: decode_episode(fetch(ds, video_path(row, cam), cache), float(row[f"videos/observation.images.{cam}/from_timestamp"]), L, fps, keep)
              for cam in ("top", "wrist")}
    step = fps // FPS_OUT
    for i in range(0, len(S), step):
        t = i / fps
        nearest(frames["top"], i).crop(BOWL).save(f"{od}/f/top_{t:06.2f}.jpg", quality=80)
        nearest(frames["wrist"], i).resize((320, 240)).save(f"{od}/f/wrist_{t:06.2f}.jpg", quality=75)
    stripe = {cam: [stripe_score(nearest(frames[cam], i).crop(BOWL) if cam == "top" else nearest(frames[cam], i)) for i in range(0, len(S), fps // 10)]
              for cam in ("top", "wrist")}
    small = {cam: {t: nearest(frames[cam], int(round(t * fps))).resize((320, 240)) for _, t in times} for cam in ("top", "wrist")}
    files = P.sheets(od, small["top"], small["wrist"], times, S, fps)
    nearest(frames["top"], int(0.3 * fps)).crop(BOWL).resize((600, 600)).save(od + "/start_bowl.jpg", quality=88)
    traj = [[round(i / fps, 1)] + [round(float(v)) for v in S[i]] for i in range(0, len(S), fps // 2)]
    json.dump(dict(dataset=ds, episode=ep, fps=fps, length_s=round(L, 2), joints=P.JN, arm_leaves_s=leave and round(leave, 2),
                   arm_home_s=round(home, 2), start_pose=[round(float(v)) for v in S[:5].mean(0)], gripper_events=ev,
                   sheet_frames=[dict(kind=k, t=tt) for k, tt in times], wflex_10hz=[round(float(v)) for v in S[::3, 3]],
                   grip_10hz=[round(float(v), 1) for v in S[::3, 5]], stripe_top_10hz=stripe["top"], stripe_wrist_10hz=stripe["wrist"],
                   sheets=files, traj_half_second=traj), open(od + "/events.json", "w"), indent=1)
    return L, len(ev)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("eps")
    ap.add_argument("--root", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); ds = a.dataset.replace("ds/", "")
    P.ROOT = a.root
    _, E, D, fps = P.load(ds)
    eps = P.parse(a.eps)
    need = {}
    for ep in eps:
        for cam in ("top", "wrist"): need.setdefault(video_path(E.loc[ep], cam), []).append(ep)
    for n, ep in enumerate(eps):
        L, ne = prep_episode(ds, ep, E.loc[ep], D, fps, a.cache, a.out)
        print(ds, ep, f"{L:.1f}s", ne, "events", flush=True)
        for rel, users in need.items():
            if users[-1] == ep or not any(u in eps[n + 1:] for u in users):
                p = os.path.join(a.cache, ds, rel)
                if os.path.exists(p): os.remove(p)


if __name__ == "__main__": main()
