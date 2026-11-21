"""Prep contact sheets + gripper events for pre-labelling.
usage: prelabel_prep.py DATASET EP[,EP|A-B ...] [--out DIR]
DATASET like so101_pick_rb_mixed. Local data only."""
import sys, glob, json, os, argparse
import numpy as np, pandas as pd, av
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.expanduser("~/so101/curation/ds")
OUT = os.path.dirname(os.path.abspath(__file__)) + "/prelabel"
JN = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]

def parse(s):
    r = []
    for p in s.split(","):
        if "-" in p: a, b = p.split("-"); r += range(int(a), int(b) + 1)
        else: r.append(int(p))
    return r

def load(ds):
    base = f"{ROOT}/{ds}"
    e = pd.concat([pd.read_parquet(f) for f in glob.glob(base + "/meta/episodes/*/*.parquet")]).sort_values("episode_index").set_index("episode_index")
    d = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "episode_index", "frame_index"]) for f in sorted(glob.glob(base + "/data/*/*.parquet"))])
    fps = json.load(open(base + "/meta/info.json"))["fps"]
    return base, e, d, fps

def med(x, k=5):
    p = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.array([np.median(p[i:i + k]) for i in range(len(x))])

def analyse(S, fps):
    t = np.arange(len(S)) / fps
    g = med(S[:, 5], 5)
    w = int(0.3 * fps)
    d = np.r_[g[w:] - g[:-w], np.zeros(w)]
    ev = []; i = 0; n = len(g)
    while i < n:
        if abs(d[i]) >= 3:
            sgn = np.sign(d[i]); j = i
            while j < n and np.sign(d[j]) == sgn and abs(d[j]) >= 1.5: j += 1
            a, b = i, min(j + w, n - 1)
            ev.append(dict(kind="close" if sgn < 0 else "open", t=round(float(t[a]), 2), t_end=round(float(t[b]), 2),
                           g_before=round(float(g[a]), 1), g_after=round(float(g[b]), 1),
                           arm=[round(float(v), 0) for v in S[a, :5]]))
            i = j
        else: i += 1
    # merge same-kind events closer than 0.4 s
    m = []
    for e in ev:
        if m and m[-1]["kind"] == e["kind"] and e["t"] - m[-1]["t_end"] < 0.4:
            m[-1]["t_end"] = e["t_end"]; m[-1]["g_after"] = e["g_after"]
        else: m.append(e)
    ev = [e for e in m if abs(e["g_after"] - e["g_before"]) >= 3]
    for e in ev: e["mid_t"] = round((e["t"] + e["t_end"]) / 2, 2)
    # arm leave / home: distance of the 4 big joints (lift, elbow, wflex) to start / end pose
    big = S[:, 1:4]
    ds_ = np.abs(big - big[:5].mean(0)).max(1); de_ = np.abs(big - big[-5:].mean(0)).max(1)
    away = np.where(ds_ > 8)[0]; leave = float(t[away[0]]) if len(away) else None
    far = np.where(de_ > 8)[0]; home = float(t[min(far[-1] + 1, len(t) - 1)]) if len(far) else 0.0
    return ev, leave, home, t

def get_frames(path, f_ts, t_from, t_dur, want, w=320, full=False):
    """decode frames at episode times `want` (secs); returns {t: PIL}"""
    out = {}
    c = av.open(path); st = c.streams.video[0]; st.thread_type = "AUTO"
    tb = float(st.time_base)
    c.seek(int(max(0, t_from + min(want) - 1.0) / tb), stream=st, backward=True, any_frame=False)
    pending = sorted(want); last = None
    for fr in c.decode(st):
        ft = float(fr.pts * tb) - t_from
        while pending and ft >= pending[0] - 1e-3:
            out[pending[0]] = fr.to_image(); pending.pop(0)
        if not pending: break
        last = fr
    for p in pending:  # past the end: use last frame
        if last is not None: out[p] = last.to_image()
    c.close()
    if full: return out
    return {k: v.resize((w, int(v.height * w / v.width))) for k, v in out.items()}

def pick_times(ev, length_s, leave, home):
    ts = [("start", 0.3)]
    for e in ev:
        if e["kind"] == "close":
            ts += [("pre-close", max(0.3, e["t"] - 1.5)), ("close", e["t"]), ("post-close", min(length_s - 0.1, e["t_end"] + 0.6))]
        else: ts.append(("open", e["mid_t"]))
    ts.append(("end", length_s - 0.2))
    ts.sort(key=lambda x: x[1]); res = []
    for k, t in ts:
        if res and t - res[-1][1] < 0.45 and k not in ("end",): continue
        res.append((k, round(t, 2)))
    return res

def sheets(path_dir, top, wr, times, S, fps):
    font = ImageFont.load_default(size=18) if hasattr(ImageFont, "load_default") else None
    W, H = 320, 240; per_row, rows = 2, 5
    files = []
    for si in range(0, len(times), per_row * rows):
        chunk = times[si:si + per_row * rows]
        nr = (len(chunk) + per_row - 1) // per_row
        im = Image.new("RGB", (per_row * 2 * W + 8, nr * (H + 4)), (30, 30, 30))
        for k, (kind, t) in enumerate(chunk):
            r, c = divmod(k, per_row); x0 = c * (2 * W + 8); y0 = r * (H + 4)
            for j, cam in enumerate((top, wr)):
                f = cam.get(t)
                if f is not None: im.paste(f.resize((W, H)), (x0 + j * W, y0))
            dr = ImageDraw.Draw(im)
            fi = min(int(t * fps), len(S) - 1)
            dr.rectangle([x0, y0, x0 + 215, y0 + 24], fill=(0, 0, 0))
            dr.text((x0 + 4, y0 + 2), f"{t:.2f}s {kind} g={S[fi,5]:.0f}", fill=(255, 255, 0), font=font)
            dr.text((x0 + 4, y0 + H - 22), "top" if True else "", fill=(255, 255, 255), font=font)
            dr.text((x0 + W + 4, y0 + H - 22), "wrist", fill=(255, 255, 255), font=font)
        fn = path_dir + ("/sheet.jpg" if si == 0 else f"/sheet_{si // (per_row * rows) + 1}.jpg")
        im.save(fn, quality=82); files.append(os.path.basename(fn))
    return files

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("eps"); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(); ds = a.dataset.replace("ds/", "")
    base, E, D, fps = load(ds)
    for ep in parse(a.eps):
        row = E.loc[ep]; x = D[D.episode_index == ep].sort_values("frame_index")
        S = np.stack(x["observation.state"].values).astype(float)
        ev, leave, home, t = analyse(S, fps)
        L = len(S) / fps
        times = pick_times(ev, L, leave, home)
        od = f"{a.out}/{ds}/{ep}"; os.makedirs(od, exist_ok=True)
        for f in glob.glob(od + "/sheet*.jpg"): os.remove(f)
        cams = {}
        for cam in ("top", "wrist"):
            k = f"videos/observation.images.{cam}"
            p = f"{base}/videos/observation.images.{cam}/chunk-{int(row[k+'/chunk_index']):03d}/file-{int(row[k+'/file_index']):03d}.mp4"
            cams[cam] = get_frames(p, None, float(row[k + "/from_timestamp"]), L, [tt for _, tt in times])
        files = sheets(od, cams["top"], cams["wrist"], times, S, fps)
        # bowl close-up at the start: the striped-ball visibility is judged on this
        k = "videos/observation.images.top"
        p = f"{base}/videos/observation.images.top/chunk-{int(row[k+'/chunk_index']):03d}/file-{int(row[k+'/file_index']):03d}.mp4"
        full = get_frames(p, None, float(row[k + "/from_timestamp"]), L, [0.3], full=True)[0.3]
        full.crop((250, 40, 550, 340)).resize((600, 600)).save(od + "/start_bowl.jpg", quality=88)
        traj = [[round(i / fps, 1)] + [round(float(v)) for v in S[i]] for i in range(0, len(S), int(fps / 2))]
        json.dump(dict(dataset=ds, episode=ep, fps=fps, length_s=round(L, 2), joints=JN, arm_leaves_s=leave and round(leave, 2),
                       arm_home_s=round(home, 2), start_pose=[round(float(v)) for v in S[:5].mean(0)], gripper_events=ev,
                       sheet_frames=[dict(kind=k, t=tt) for k, tt in times], wflex_10hz=[round(float(v)) for v in S[::3, 3]], sheets=files, traj_half_second=traj),
                  open(od + "/events.json", "w"), indent=1)
        print(ep, f"{L:.1f}s", len(ev), "events", files)

if __name__ == "__main__": main()
