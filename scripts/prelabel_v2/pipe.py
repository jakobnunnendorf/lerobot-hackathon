# Written by Claude: stream a dataset's videos one file at a time (download, extract stripe features at 5 Hz, delete) to keep disk use low
import sys, os, json, glob, subprocess, av, cv2, numpy as np, pandas as pd
name, lo, hi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ROOT = f'dsx/{name}'; HF = f'https://huggingface.co/datasets/molnrb/{name}/resolve/main'
os.makedirs(ROOT + '/meta/episodes/chunk-000', exist_ok=True); os.makedirs(ROOT + '/data/chunk-000', exist_ok=True)
def fetch(rel):
    p = f'{ROOT}/{rel}'
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        subprocess.check_call(['curl', '-sfL', '-o', p, f'{HF}/{rel}'])
    return p
tree = json.loads(subprocess.check_output(['curl', '-s', f'https://huggingface.co/api/datasets/molnrb/{name}/tree/main?recursive=true']))
for f in tree:
    if f['type'] == 'file' and (f['path'].startswith('meta/') or f['path'].startswith('data/')): fetch(f['path'])
meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(ROOT + '/meta/episodes/chunk-000/*.parquet'))]).set_index('episode_index', drop=False)
STRIPE = ((28, 45, 80), (48, 255, 255)); STRIPE_W = ((30, 32, 40), (86, 255, 255))
def masks(bgr, cam):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h, s, v = cv2.split(hsv)
    hcv = cv2.merge([h, cv2.multiply(s, v, scale=1 / 255), v])
    lo_, hi_ = STRIPE if cam == 'top' else STRIPE_W
    st = cv2.inRange(hcv, lo_, hi_)
    red = cv2.inRange(hsv, (0, 110, 90), (8, 255, 255)) | cv2.inRange(hsv, (172, 110, 90), (180, 255, 255))
    blue = cv2.inRange(hsv, (100, 150, 50), (130, 255, 255))
    return int((st > 0).sum()), int((red > 0).sum()), int((blue > 0).sum())
eps = [e for e in meta.index if lo <= e <= hi]
res = json.load(open(out)) if os.path.exists(out) else {}
for cam in ('top', 'wrist'):
    k = f'videos/observation.images.{cam}'
    files = sorted(set(int(meta.loc[e, k + '/file_index']) for e in eps))
    for fi in files:
        todo = [e for e in eps if int(meta.loc[e, k + '/file_index']) == fi and cam not in res.get(str(e), {})]
        if not todo: continue
        p = fetch(f'{k}/chunk-000/file-{fi:03d}.mp4')
        for ep in todo:
            r = meta.loc[ep]; t0 = float(r[k + '/from_timestamp']); L = float(r[k + '/to_timestamp']) - t0
            c = av.open(p); s = c.streams.video[0]; s.thread_type = 'AUTO'; tb = float(s.time_base)
            c.seek(int(max(0, t0 - 0.5) / tb), stream=s, backward=True)
            want = list(np.arange(0, L, 0.2)); rows = []
            for f in c.decode(s):
                ft = float(f.pts * tb) - t0
                if ft < want[0] - 0.01: continue
                while want and ft >= want[0] - 0.01:
                    img = f.to_ndarray(format='bgr24')
                    if cam == 'top': img = img[40:340, 250:550]
                    rows.append([round(want[0], 2), *masks(img, cam)]); want.pop(0)
                if not want: break
            c.close(); res.setdefault(str(ep), {})[cam] = rows
        os.remove(p); json.dump(res, open(out, 'w')); print(name, cam, 'file', fi, 'done', len(todo), flush=True)
