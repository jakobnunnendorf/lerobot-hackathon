"""Classify one top-camera frame: prints {"visibility", "next_action", "probs"} as JSON.

    python -I scripts/scene_clf/predict.py frame.png --state 0 -20 30 60 0 10
Also importable: predict(frame_rgb_uint8, state) with a frame of any size.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import label_data as L
from train import SceneClassifier

WEIGHTS = Path(__file__).resolve().parents[2] / "outputs/scene_clf/scene_clf.pt"


def load_model(weights: Path = WEIGHTS):
    ckpt = torch.load(weights, map_location="cpu")
    model = SceneClassifier(pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.eval(), np.array(ckpt["state_mean"], dtype=np.float32), np.array(ckpt["state_std"], dtype=np.float32)


def predict(frame: np.ndarray, state: np.ndarray | None = None, loaded=None) -> dict:
    model, mean, std = loaded or load_model()
    small = np.asarray(Image.fromarray(frame).resize(L.SIZE, Image.BILINEAR))
    # No state given: the training mean, which the model treats as "arm parked in a typical pose".
    state = mean if state is None else np.asarray(state, dtype=np.float32)
    with torch.no_grad():
        v, n = model(torch.from_numpy(small[None].copy()), torch.from_numpy(((state - mean) / std)[None].astype(np.float32)))
    pv, pn = v.softmax(-1)[0].numpy(), n.softmax(-1)[0].numpy()
    return {
        "visibility": L.VISIBILITY[int(pv.argmax())],
        "next_action": L.NEXT_ACTION[int(pn.argmax())],
        "probs": {
            "visibility": dict(zip(L.VISIBILITY, map(float, pv))),
            "next_action": dict(zip(L.NEXT_ACTION, map(float, pn))),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="image file (png/jpg) or .npy array [H,W,3] uint8 RGB")
    ap.add_argument("--state", type=float, nargs=6, default=None)
    args = ap.parse_args()
    frame = np.load(args.image) if args.image.suffix == ".npy" else np.asarray(Image.open(args.image).convert("RGB"))
    print(json.dumps(predict(frame, args.state)))


if __name__ == "__main__":
    main()
