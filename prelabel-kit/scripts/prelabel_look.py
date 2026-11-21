# Written by Claude (pre-label v2)
"""Montage of the 5 fps frames saved by prelabel_prep_v2.py, one row per time, top bowl crop | wrist.
usage: prelabel_look.py PL_DIR DATASET EP T0 T1 [STEP] [--cam both|top|wrist] [--size S] [--out FILE]
   or: prelabel_look.py PL_DIR DATASET EP --times 3.2,5.0,7.4 ...
Times snap to the nearest saved frame (0.2 s grid). Prints the output path."""
import os, glob, argparse
from PIL import Image, ImageDraw, ImageFont


def frames(d, cam):
    return {float(os.path.basename(p)[len(cam) + 1:-4]): p for p in glob.glob(f"{d}/f/{cam}_*.jpg")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pl"); ap.add_argument("dataset"); ap.add_argument("ep", type=int)
    ap.add_argument("t0", type=float, nargs="?"); ap.add_argument("t1", type=float, nargs="?"); ap.add_argument("step", type=float, nargs="?", default=0.4)
    ap.add_argument("--times"); ap.add_argument("--cam", default="both"); ap.add_argument("--size", type=int, default=220)
    ap.add_argument("--cols", type=int, default=0); ap.add_argument("--out")
    a = ap.parse_args()
    ds = a.dataset.replace("ds/", ""); d = f"{a.pl}/{ds}/{a.ep}"
    cams = ["top", "wrist"] if a.cam == "both" else [a.cam]
    have = {c: frames(d, c) for c in cams}
    if a.times: want = [float(x) for x in a.times.split(",")]
    else:
        want, t = [], a.t0
        while t <= a.t1 + 1e-6: want.append(round(t, 2)); t += a.step
    keys = sorted(have[cams[0]])
    snap = [min(keys, key=lambda k: abs(k - t)) for t in want]
    S = a.size; tile_w = S * len(cams) + 6
    cols = a.cols or (4 if len(cams) == 2 else 6)
    rows = (len(snap) + cols - 1) // cols
    im = Image.new("RGB", (cols * tile_w, rows * (S + 4)), (25, 25, 25))
    dr = ImageDraw.Draw(im); font = ImageFont.load_default(size=16)
    for n, t in enumerate(snap):
        r, c = divmod(n, cols); x0, y0 = c * tile_w, r * (S + 4)
        for j, cam in enumerate(cams):
            f = Image.open(have[cam][t]); w = S if cam == "top" else S; h = S if cam == "top" else int(S * 0.75)
            im.paste(f.resize((w, h)), (x0 + j * S, y0))
        dr.rectangle([x0, y0, x0 + 62, y0 + 19], fill=(0, 0, 0)); dr.text((x0 + 3, y0 + 1), f"{t:.1f}s", fill=(255, 255, 0), font=font)
    out = a.out or f"{d}/look_{want[0]:.1f}_{want[-1]:.1f}_{a.cam}.jpg"
    im.save(out, quality=82); print(out)


if __name__ == "__main__": main()
