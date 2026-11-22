"""Pretrain a ResNet18 on the curation labels (top and wrist frames) so ACT's image encoder starts with the cues that matter.

    /path/to/python -I scripts/label_encoder/pretrain.py --cache <dir with frames.npz> --out outputs/label_encoder
Heads: striped visibility (top), target ball of the running or next segment, wrist blind (wrist), phase.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import label_data as L

BLIND_POS_WEIGHT = 4.0


class LabelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        net.fc = nn.Identity()
        self.backbone = net
        self.camera = nn.Embedding(2, 32)
        self.trunk = nn.Sequential(nn.Linear(512 + 32, 256), nn.ReLU(), nn.Dropout(0.2))
        self.visibility = nn.Linear(256, 3)
        self.target = nn.Linear(256, 4)
        self.blind = nn.Linear(256, 1)
        self.phase = nn.Linear(256, 4)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image_uint8: torch.Tensor, camera: torch.Tensor) -> dict:
        x = (image_uint8.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        h = self.trunk(torch.cat([self.backbone(x), self.camera(camera)], 1))
        return {"visibility": self.visibility(h), "target": self.target(h), "blind": self.blind(h)[:, 0], "phase": self.phase(h)}

    def train(self, mode: bool = True):
        super().train(mode)
        # Batch-norm statistics stay at ImageNet values: ACT runs this backbone with frozen batch norm.
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


def frame_labels(cache: dict, labels: dict) -> dict:
    n = len(cache["ep"])
    out = {k: np.zeros(n, dtype=np.int64) for k in ("visibility", "target", "blind", "phase")}
    for i in range(n):
        rec, t = labels[int(cache["ep"][i])], float(cache["t"][i])
        out["visibility"][i] = L.visibility_at(rec, t)
        out["target"][i], out["phase"][i] = L.target_and_phase_at(rec, t)
        out["blind"][i] = L.wrist_blind_at(rec, t)
    return out


def augment(x: torch.Tensor) -> torch.Tensor:
    gain = 1 + 0.2 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 2
    shift = 0.1 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 2 * 255
    x = (x.float() * gain + shift).clamp(0, 255)
    return torch.roll(x, shifts=(np.random.randint(-6, 7), np.random.randint(-6, 7)), dims=(1, 2))


def confusion(true, pred, n) -> list:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m.tolist()


def predict_all(model, device, images, camera, batch=256) -> dict:
    model.eval()
    outs = {k: [] for k in ("visibility", "target", "blind", "phase")}
    with torch.no_grad():
        for i in range(0, len(images), batch):
            o = model(torch.from_numpy(images[i : i + batch]).to(device), torch.from_numpy(camera[i : i + batch]).to(device))
            for k in outs:
                outs[k].append((o[k] > 0).long() if k == "blind" else o[k].argmax(1))
    return {k: torch.cat(v).cpu().numpy() for k, v in outs.items()}


def score(pred: dict, true: dict, cam: np.ndarray) -> dict:
    top, wrist = cam == 0, cam == 1
    report = {
        "visibility_top": {"acc": float((pred["visibility"][top] == true["visibility"][top]).mean()),
                           "confusion": confusion(true["visibility"][top], pred["visibility"][top], 3)},
        "target_top": {"acc": float((pred["target"][top] == true["target"][top]).mean()),
                       "confusion": confusion(true["target"][top], pred["target"][top], 4)},
        "target_wrist": {"acc": float((pred["target"][wrist] == true["target"][wrist]).mean()),
                         "confusion": confusion(true["target"][wrist], pred["target"][wrist], 4)},
        "phase_top": {"acc": float((pred["phase"][top] == true["phase"][top]).mean()),
                      "confusion": confusion(true["phase"][top], pred["phase"][top], 4)},
        "phase_wrist": {"acc": float((pred["phase"][wrist] == true["phase"][wrist]).mean()),
                        "confusion": confusion(true["phase"][wrist], pred["phase"][wrist], 4)},
        "wrist_blind": {"acc": float((pred["blind"][wrist] == true["blind"][wrist]).mean()),
                        "confusion": confusion(true["blind"][wrist], pred["blind"][wrist], 2)},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/label_encoder"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    cache = L.load_cache(args.cache)
    labels = L.load_labels()
    y = frame_labels(cache, labels)
    images = np.concatenate([cache["top"], cache["wrist"]])
    camera = np.concatenate([np.zeros(len(cache["ep"]), np.int64), np.ones(len(cache["ep"]), np.int64)])
    y = {k: np.concatenate([v, v]) for k, v in y.items()}
    ep = np.concatenate([cache["ep"], cache["ep"]])
    ho = np.isin(ep, L.HOLDOUT)
    tr_idx, ho_idx = np.where(~ho)[0], np.where(ho)[0]
    print(f"train {len(tr_idx)} frames, hold-out {len(ho_idx)}; device {device}")
    print("train counts", {k: np.bincount(v[tr_idx]).tolist() for k, v in y.items()})

    model = LabelEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    steps = args.epochs * (len(tr_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=steps + 1, pct_start=0.1)
    ce = nn.CrossEntropyLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(BLIND_POS_WEIGHT, device=device))
    yt = {k: torch.from_numpy(v) for k, v in y.items()}
    cam_t = torch.from_numpy(camera)
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        total = 0.0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            b = perm[i : i + args.batch]
            o = model(augment(torch.from_numpy(images[b]).to(device)), cam_t[b].to(device))
            is_top = (cam_t[b] == 0).float().to(device)
            is_wrist = 1 - is_top
            loss = (
                (ce(o["visibility"], yt["visibility"][b].to(device)) * is_top).sum() / is_top.sum().clamp(min=1)
                + ce(o["target"], yt["target"][b].to(device)).mean()
                + ce(o["phase"], yt["phase"][b].to(device)).mean()
                + (bce(o["blind"], yt["blind"][b].float().to(device)) * is_wrist).sum() / is_wrist.sum().clamp(min=1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        pred = predict_all(model, device, images[ho_idx], camera[ho_idx])
        r = score(pred, {k: v[ho_idx] for k, v in y.items()}, camera[ho_idx])
        print(f"epoch {epoch + 1} loss {total / (len(perm) // args.batch):.3f} hold-out " +
              " ".join(f"{k}={v['acc']:.3f}" for k, v in r.items()), flush=True)

    report = {"holdout_episodes": L.HOLDOUT, "train_frames": int(len(tr_idx)), "holdout_frames": int(len(ho_idx)),
              "classes": {"visibility": L.VISIBILITY, "target": L.TARGET, "phase": L.PHASE},
              "holdout": r,
              "holdout_majority": {k: float(np.bincount(y[k][tr_idx]).max() / len(tr_idx)) for k in y}}
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.backbone.state_dict().items()}, args.out / "resnet18_backbone.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["acc"] for k, v in r.items()}))


if __name__ == "__main__":
    main()
