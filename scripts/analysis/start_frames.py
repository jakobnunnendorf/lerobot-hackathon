"""Contact sheet of the top-camera frame the SmolVLA sees at episode start (512x512 after resize_with_pad),
with SmolVLM's 8x8 visual-token grid drawn on it and Jakob's start state + first target as caption.
Also prints, per episode, how many striped-ball (yellow) pixels that frame shows inside the bowl.

    python scripts/analysis/start_frames.py --dataset-root <rb_mixed snapshot> --labels jakob.jsonl \
        --episodes 0 4 9 19 23 30 31 --out start_frames.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot.policies.common.vla_utils import resize_with_pad
from smolvla_first_chunk import Episode

TOKEN_GRID = 8  # SigLIP 512px / patch 16 = 32x32 patches, pixel-shuffled by scale_factor 4 -> 8x8 tokens
BOWL = (222, 128, 448, 384)  # left edge clears the orange arm at its start pose
YELLOW_HUE = (25, 45)  # PIL's 0-255 hue scale; red, blue and the orange arm fall outside


def striped_pixels(image: Image.Image) -> int:
    hsv = np.asarray(image.crop(BOWL).convert("HSV")).astype(int)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return int(((hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1]) & (sat > 90) & (val > 90)).sum())


def network_input(ep: Episode, t: float) -> Image.Image:
    padded = resize_with_pad(ep.image("top", t)[None], 512, 512, pad_value=0)[0]
    return Image.fromarray((padded.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def with_token_grid(image: Image.Image) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    step = 512 // TOKEN_GRID
    for k in range(1, TOKEN_GRID):
        draw.line([(k * step, 0), (k * step, 511)], fill=(255, 255, 255), width=1)
        draw.line([(0, k * step), (511, k * step)], fill=(255, 255, 255), width=1)
    return image


def caption(ep: Episode) -> str:
    first = next((c for _, c in ep.events if c.endswith("_target")), "-")
    cleared = any(c.endswith("_hole") for _, c in ep.events)
    return f"ep{ep.index} start={ep.start_state} first={first[0].upper()} {'clears' if cleared else 'direct'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, default=(192, 128, 448, 384), help="left top right bottom")
    args = parser.parse_args()

    labels = {}
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        labels[row["episode"]] = row

    tiles = []
    print("episode,start_state,cleared_first,striped_px")
    for index in args.episodes:
        ep = Episode(args.dataset_root, index, labels[index])
        image = network_input(ep, 0.0)
        cleared = any(c.endswith("_hole") for _, c in ep.events)
        print(f"{index},{ep.start_state},{int(cleared)},{striped_pixels(image)}", flush=True)
        tile = with_token_grid(image).crop(args.crop)
        ImageDraw.Draw(tile).text((4, 4), caption(ep), fill=(255, 255, 0))
        tiles.append(tile)
    width, height = tiles[0].size
    columns = 4
    sheet = Image.new("RGB", (width * columns, height * -(-len(tiles) // columns)))
    for n, tile in enumerate(tiles):
        sheet.paste(tile, ((n % columns) * width, (n // columns) * height))
    sheet.save(args.out)


if __name__ == "__main__":
    main()
