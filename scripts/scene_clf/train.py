"""Train the clear-or-pick scene classifier (ResNet18 + state, two heads) and score it on the hold-out episodes.

    /path/to/python -I scripts/scene_clf/train.py --cache <dir with frames.npz> --out outputs/scene_clf
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

PRE_TARGET_WINDOW_S = 1.5
MINT_HUE_DEG = (50, 160)
BOWL_ROI = (slice(12, 72), slice(62, 142))  # rows, columns of the bowl in the fixed top camera, at 160x120
MINT_MIN_SAT = 0.30
MINT_MIN_VAL = 0.35
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SceneClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        net.fc = nn.Identity()
        self.backbone = net
        self.state_mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(0.2))
        self.visibility = nn.Linear(128, len(L.VISIBILITY))
        self.next_action = nn.Linear(128, len(L.NEXT_ACTION))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image_uint8: torch.Tensor, state_norm: torch.Tensor):
        x = (image_uint8.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        h = self.trunk(torch.cat([self.backbone(x), self.state_mlp(state_norm)], 1))
        return self.visibility(h), self.next_action(h)


def mint_fraction(frames: np.ndarray) -> np.ndarray:
    """Share of bowl pixels that are yellow-green to mint, for frames [N,H,W,3] uint8."""
    x = frames[:, BOWL_ROI[0], BOWL_ROI[1]].astype(np.float32) / 255
    mx, mn = x.max(-1), x.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    mask = (hue >= MINT_HUE_DEG[0]) & (hue <= MINT_HUE_DEG[1]) & (mx > MINT_MIN_VAL) & (d / np.maximum(mx, 1e-6) > MINT_MIN_SAT)
    return mask.mean((1, 2))


def decision_samples(cache: dict, labels: dict) -> dict:
    """Frames the supervisor decides on: the 1.5 s before each *_target event, plus t=0."""
    rows, vis, nxt = [], [], []
    for e, rec in labels.items():
        idx = np.where(cache["ep"] == e)[0]
        times = cache["t"][idx]
        target_times = [te for te, code in L.events(rec) if code.endswith("_target")]
        keep = times < 1e-6
        for te in target_times:
            keep |= (times >= te - PRE_TARGET_WINDOW_S - 1e-6) & (times <= te + 1e-6)
        for i in idx[keep]:
            action = L.next_action_at(rec, float(cache["t"][i]))
            if action is None:
                continue
            rows.append(i)
            vis.append(L.visibility_at(rec, float(cache["t"][i])))
            nxt.append(L.NEXT_ACTION.index(action))
    rows = np.array(rows)
    return {"rows": rows, "ep": cache["ep"][rows], "vis": np.array(vis), "next": np.array(nxt)}


def confusion(true: np.ndarray, pred: np.ndarray, n: int) -> list:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m.tolist()


def accuracy(true, pred) -> float:
    return float((np.asarray(true) == np.asarray(pred)).mean())


def tune_mint_baseline(frac: np.ndarray, vis: np.ndarray) -> tuple[float, float]:
    """Two thresholds on mint fraction (buried | partly | visible), best train accuracy."""
    grid = np.quantile(frac, np.linspace(0.02, 0.98, 60))
    best = (-1.0, grid[0], grid[-1])
    for a in grid:
        for b in grid[grid > a]:
            acc = accuracy(vis, np.where(frac < a, 0, np.where(frac < b, 1, 2)))
            if acc > best[0]:
                best = (acc, a, b)
    return float(best[1]), float(best[2])


def baseline_report(frac_tr, tr, frac_ho, ho) -> dict:
    a, b = tune_mint_baseline(frac_tr, tr["vis"])
    pred_vis = lambda f: np.where(f < a, 0, np.where(f < b, 1, 2))
    # Mint count cannot tell red from blue: it says pick_striped when the ball is visible enough, otherwise
    # the commonest clearing colour of the training set.
    clear_default = int(np.bincount(tr["next"][tr["next"] < 2], minlength=2).argmax())
    pick_th = tune_pick_threshold(frac_tr, tr["next"], clear_default)
    pred_next = lambda f: np.where(f >= pick_th, 2, clear_default)
    return {
        "thresholds": [a, b], "pick_threshold": pick_th,
        "visibility_acc": accuracy(ho["vis"], pred_vis(frac_ho)),
        "visibility_confusion": confusion(ho["vis"], pred_vis(frac_ho), 3),
        "next_action_acc": accuracy(ho["next"], pred_next(frac_ho)),
        "next_action_confusion": confusion(ho["next"], pred_next(frac_ho), 3),
    }


def tune_pick_threshold(frac, nxt, clear_default) -> float:
    grid = np.quantile(frac, np.linspace(0.02, 0.98, 80))
    scores = [accuracy(nxt, np.where(frac >= th, 2, clear_default)) for th in grid]
    return float(grid[int(np.argmax(scores))])


def augment(x: torch.Tensor) -> torch.Tensor:
    gain = 1 + 0.2 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 2
    shift = 0.1 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 2 * 255
    x = (x.float() * gain + shift).clamp(0, 255)
    dx, dy = np.random.randint(-6, 7), np.random.randint(-6, 7)
    return torch.roll(x, shifts=(dy, dx), dims=(1, 2))


def evaluate(model, device, images, states, batch=128) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    out_v, out_n = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            v, n = model(torch.from_numpy(images[i : i + batch]).to(device), torch.from_numpy(states[i : i + batch]).to(device))
            out_v.append(v.softmax(-1).cpu().numpy())
            out_n.append(n.softmax(-1).cpu().numpy())
    return np.concatenate(out_v), np.concatenate(out_n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/scene_clf"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    cache = L.load_cache(args.cache)
    labels = L.load_labels()
    s = decision_samples(cache, labels)
    holdout = np.isin(s["ep"], L.HOLDOUT)
    pick = lambda m: {k: v[m] for k, v in s.items()}
    tr, ho = pick(~holdout), pick(holdout)
    print(f"train {len(tr['rows'])} frames, hold-out {len(ho['rows'])} frames; device {device}")
    print("train visibility counts", np.bincount(tr["vis"], minlength=3), "next", np.bincount(tr["next"], minlength=3))

    mean, std = cache["state"][tr["rows"]].mean(0), cache["state"][tr["rows"]].std(0) + 1e-6
    norm = lambda st: ((st - mean) / std).astype(np.float32)
    img_tr, st_tr = cache["top"][tr["rows"]], norm(cache["state"][tr["rows"]])
    img_ho, st_ho = cache["top"][ho["rows"]], norm(cache["state"][ho["rows"]])

    model = SceneClassifier().to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW(
        [{"params": model.backbone.layer4.parameters(), "lr": 1e-4}, {"params": head_params, "lr": 5e-4}], weight_decay=1e-2
    )
    ce = nn.CrossEntropyLoss()
    y_v, y_n = torch.from_numpy(tr["vis"]), torch.from_numpy(tr["next"])
    for epoch in range(args.epochs):
        model.train()
        model.backbone.layer1.eval(), model.backbone.layer2.eval(), model.backbone.layer3.eval()
        model.backbone.bn1.eval()
        perm = np.random.permutation(len(img_tr))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i : i + args.batch]
            x = augment(torch.from_numpy(img_tr[b]).to(device))
            lv, ln = model(x, torch.from_numpy(st_tr[b]).to(device))
            loss = ce(lv, y_v[b].to(device)) + ce(ln, y_n[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        pv, pn = evaluate(model, device, img_ho, st_ho)
        print(f"epoch {epoch + 1} loss {total / len(perm):.3f} hold-out vis {accuracy(ho['vis'], pv.argmax(1)):.3f} "
              f"next {accuracy(ho['next'], pn.argmax(1)):.3f}", flush=True)

    pv, pn = evaluate(model, device, img_ho, st_ho)
    frac_tr, frac_ho = mint_fraction(img_tr), mint_fraction(img_ho)
    report = {
        "train_frames": int(len(tr["rows"])), "holdout_frames": int(len(ho["rows"])), "holdout_episodes": L.HOLDOUT,
        "visibility_classes": L.VISIBILITY, "next_action_classes": L.NEXT_ACTION,
        "model": {
            "visibility_acc": accuracy(ho["vis"], pv.argmax(1)), "visibility_confusion": confusion(ho["vis"], pv.argmax(1), 3),
            "next_action_acc": accuracy(ho["next"], pn.argmax(1)), "next_action_confusion": confusion(ho["next"], pn.argmax(1), 3),
        },
        "mint_baseline": baseline_report(frac_tr, tr, frac_ho, ho),
        "majority_baseline": {
            "visibility_acc": accuracy(ho["vis"], np.full(len(ho["vis"]), np.bincount(tr["vis"]).argmax())),
            "next_action_acc": accuracy(ho["next"], np.full(len(ho["next"]), np.bincount(tr["next"]).argmax())),
        },
        "holdout_label_counts": {"visibility": np.bincount(ho["vis"], minlength=3).tolist(), "next_action": np.bincount(ho["next"], minlength=3).tolist()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu().half() if v.is_floating_point() else v.cpu() for k, v in model.state_dict().items()},
                "state_mean": mean.tolist(), "state_std": std.tolist()}, args.out / "scene_clf.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
