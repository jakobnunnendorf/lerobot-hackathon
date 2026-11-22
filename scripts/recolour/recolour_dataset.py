import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from target_mask import TargetMasker

COLOURS = {
    "red": (205, 30, 35),
    "blue": (30, 70, 200),
    "purple": (120, 40, 160),
    "orange": (235, 110, 20),
}
ENCODERS = {"av1": "libsvtav1"}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
TO_LINEAR = ((np.arange(256, dtype=np.float32) / 255) ** 2.2).astype(np.float32)


def parse_colour(text: str) -> np.ndarray:
    if text in COLOURS:
        return np.array(COLOURS[text], np.float32)
    if text.startswith("#") and len(text) == 7:
        return np.array([int(text[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
    return np.array([float(v) for v in text.split(",")], np.float32)


def _box(mask: np.ndarray, pad: int) -> tuple[slice, slice]:
    x, y, w, h = cv2.boundingRect(mask)
    return (slice(max(y - pad, 0), min(y + h + pad, mask.shape[0])),
            slice(max(x - pad, 0), min(x + w + pad, mask.shape[1])))


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def _local_mean(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    return _blur(values * weights, sigma) / np.maximum(_blur(weights, sigma), 1e-4)


def _green_stripe_weight(crop_rgb: np.ndarray) -> np.ndarray:
    hue, saturation, _ = cv2.split(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV).astype(np.float32))
    return np.clip((hue - 38) / 10, 0, 1) * np.clip((90 - hue) / 5, 0, 1) * np.clip((saturation - 60) / 40, 0, 1)


def _unstriped_shading(crop: np.ndarray, inside: np.ndarray) -> np.ndarray:
    luma = TO_LINEAR[crop] @ LUMA
    green = _green_stripe_weight(crop) * inside
    cream = inside - green
    sigma = max(2.0, 0.12 * float(np.sqrt(inside.sum())))
    ratio = np.clip(_local_mean(luma, cream, sigma) / np.maximum(_local_mean(luma, green, sigma), 1e-4), 1, 6)
    shading = luma * (1 + green * (ratio - 1))
    # Seen from afar the stripes blend into a darker mixed albedo than the cream alone.
    resolved = np.clip(green.sum() / max(inside.sum(), 1) / 0.2, 0, 1)
    return np.clip(shading / (0.5 + 0.2 * resolved), 0, 1.6)


# Not a hue shift, which leaves a two-tone striped ball: see decisions/other-ball-colours.md.
def paint(rgb: np.ndarray, mask: np.ndarray, colour: np.ndarray, edge_sigma: float = 1.2) -> np.ndarray:
    if not mask.any():
        return rgb
    box = _box(mask, pad=4)
    crop = rgb[box]
    inside = (mask[box] > 0).astype(np.float32)
    shading = _unstriped_shading(crop, inside)[..., None]
    painted = 255 * np.clip(shading * TO_LINEAR[colour.astype(np.uint8)], 0, 1) ** (1 / 2.2)
    alpha = _blur(inside, edge_sigma)[..., None]
    out = rgb.copy()
    out[box] = np.clip(alpha * painted + (1 - alpha) * crop, 0, 255).astype(np.uint8)
    return out


def read_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def video_keys(info: dict) -> list[str]:
    return [key for key, feature in info["features"].items() if feature["dtype"] == "video"]


def camera_of(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def read_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    episodes = pd.concat(pd.read_parquet(f) for f in files)
    episodes = episodes[[c for c in episodes.columns if not c.startswith("stats/")]]
    return episodes.set_index("episode_index").sort_index()


def video_path(root: Path, info: dict, key: str, chunk: int, file: int) -> Path:
    return root / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)


def episode_video(root: Path, info: dict, episodes: pd.DataFrame, episode: int, key: str) -> tuple[Path, float, float]:
    row = episodes.loc[episode]
    path = video_path(root, info, key, int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"]))
    return path, float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"])


def iter_episode_frames(path: Path, start: float, end: float, fps: int):
    half_frame = 0.5 / fps
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(start - 1.0, 0) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start - half_frame:
                continue
            if t >= end - half_frame:
                break
            yield t, frame.to_ndarray(format="rgb24")


def episode_starts(episodes: pd.DataFrame, key: str) -> dict[tuple[int, int], list[float]]:
    starts = {}
    for _, row in episodes.iterrows():
        file = (int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"]))
        starts.setdefault(file, []).append(float(row[f"videos/{key}/from_timestamp"]))
    return {file: sorted(times) for file, times in starts.items()}


def encoder_settings(feature: dict) -> tuple[str, dict, str]:
    info = feature["info"]
    options = {"g": str(info["video.g"]), "crf": str(info["video.crf"]), "preset": str(info["video.preset"])}
    return ENCODERS[info["video.codec"]], options, info["video.pix_fmt"]


# The tracker is restarted at every episode start so a file gives the same masks as the
# episodes processed one by one, and as the live loop would.
def recolour_file(src: Path, dst: Path, camera: str, colour: np.ndarray, starts: list[float],
                  codec: str, options: dict, pix_fmt: str) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    pending, masker, frames, masked = list(starts), TargetMasker(camera), 0, 0
    with av.open(str(src)) as source_file, av.open(str(dst), "w") as target_file:
        source = source_file.streams.video[0]
        fps = round(float(source.average_rate))
        target = target_file.add_stream(codec, rate=fps, options=options)
        target.width, target.height, target.pix_fmt = source.width, source.height, pix_fmt
        for frame in source_file.decode(source):
            time = float(frame.pts * source.time_base)
            while pending and time >= pending[0] - 0.5 / fps:
                masker, pending = TargetMasker(camera), pending[1:]
            rgb = frame.to_ndarray(format="rgb24")
            mask = masker(rgb)
            masked += bool(mask.any())
            painted = av.VideoFrame.from_ndarray(paint(rgb, mask, colour), format="rgb24")
            painted.pts, painted.time_base = round(time * fps), Fraction(1, fps)
            target_file.mux(target.encode(painted))
            frames += 1
        target_file.mux(target.encode())
    return {"file": str(dst), "frames": frames, "masked": masked}


# meta/ is copied unchanged, so its image statistics still describe the striped ball: see
# decisions/other-ball-colours.md.
def recolour_dataset(source: Path, output: Path, colour: np.ndarray, workers: int) -> list[dict]:
    info, episodes = read_info(source), read_episodes(source)
    for folder in ["meta", "data"]:
        shutil.copytree(source / folder, output / folder)
    jobs = []
    for key in video_keys(info):
        codec, options, pix_fmt = encoder_settings(info["features"][key])
        for (chunk, file), starts in episode_starts(episodes, key).items():
            src = video_path(source, info, key, chunk, file)
            dst = video_path(output, info, key, chunk, file)
            jobs.append((src, dst, camera_of(key), colour, starts, codec, options, pix_fmt))
    # Largest files first: one file is one job, and the biggest one sets the wall time.
    jobs.sort(key=lambda job: job[0].stat().st_size, reverse=True)
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(recolour_file, *zip(*jobs)))


def _frame_times(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [float(p.pts * stream.time_base) for p in container.demux(stream) if p.pts is not None]


def _decoded_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_at(path: Path, time: float, fps: int) -> np.ndarray:
    return next(iter_episode_frames(path, time, time + 1.0 / fps, fps))[1]


def _psnr(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> float:
    error = np.mean((a[keep].astype(np.float32) - b[keep].astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(error, 1e-6)))


def _check_video(src: Path, dst: Path) -> dict:
    src_times, dst_times = _frame_times(src), _frame_times(dst)
    return {
        "file": str(dst), "frames": (len(src_times), len(dst_times)), "decoded": _decoded_frames(dst),
        "same_timestamps": len(src_times) == len(dst_times) and np.allclose(sorted(src_times), sorted(dst_times), atol=1e-6),
    }


def _check_table(src: Path, dst: Path) -> dict:
    return {"file": str(dst), "identical": _sha256(src) == _sha256(dst)}


# Mid-episode, because the arm is parked at both ends and neighbouring frames there look alike:
# outside the ball the output must match its own source frame better than either neighbour.
def _check_alignment(source: Path, output: Path, info: dict, episodes: pd.DataFrame, episode: int, key: str) -> dict:
    src, start, end = episode_video(source, info, episodes, episode, key)
    dst = output / src.relative_to(source)
    fps = info["fps"]
    time = start + round((end - start) * fps / 2) / fps
    recoloured, original = _frame_at(dst, time, fps), _frame_at(src, time, fps)
    ball = cv2.dilate(TargetMasker(camera_of(key))(original), np.ones((9, 9), np.uint8)) > 0
    psnr = {offset: round(_psnr(_frame_at(src, time + offset / fps, fps), recoloured, ~ball), 2) for offset in (-1, 0, 1)}
    return {
        "episode": episode, "key": key, "time": round(time, 4), "ball_pixels": int(ball.sum()),
        "psnr_previous_same_next": [psnr[-1], psnr[0], psnr[1]], "aligned": psnr[0] > max(psnr[-1], psnr[1]),
    }


def verify(source: Path, output: Path, check_episodes: list[int]) -> dict:
    info, episodes = read_info(source), read_episodes(source)
    tables = [src for folder in ["meta", "data"] for src in sorted((source / folder).rglob("*")) if src.is_file()]
    return {
        "videos": [_check_video(src, output / src.relative_to(source)) for src in sorted((source / "videos").rglob("*.mp4"))],
        "tables": [_check_table(src, output / src.relative_to(source)) for src in tables],
        "alignment": [_check_alignment(source, output, info, episodes, episode, key)
                      for episode in check_episodes for key in video_keys(info)],
    }


def write_previews(source: Path, output: Path, colour: np.ndarray, episodes_wanted: list[int]) -> None:
    info, episodes = read_info(source), read_episodes(source)
    fps = info["fps"]
    output.mkdir(parents=True, exist_ok=True)
    for episode in episodes_wanted:
        for key in video_keys(info):
            masker = TargetMasker(camera_of(key))
            path, start, end = episode_video(source, info, episodes, episode, key)
            with av.open(str(output / f"preview_e{episode:03d}_{camera_of(key)}.mp4"), "w") as preview:
                stream = preview.add_stream("libx264", rate=fps, options={"crf": "20"})
                stream.width, stream.height, stream.pix_fmt = 1280, 480, "yuv420p"
                for index, (_, rgb) in enumerate(iter_episode_frames(path, start, end, fps)):
                    frame = av.VideoFrame.from_ndarray(np.hstack([rgb, paint(rgb, masker(rgb), colour)]), format="rgb24")
                    frame.pts, frame.time_base = index, Fraction(1, fps)
                    preview.mux(stream.encode(frame))
                preview.mux(stream.encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colour", default="red", help="name, #rrggbb or r,g,b")
    parser.add_argument("--episodes", help="comma-separated episodes: write side-by-side previews only")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    colour = parse_colour(args.colour)
    if args.episodes:
        write_previews(args.source, args.output, colour, [int(e) for e in args.episodes.split(",")])
        return
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; refusing to overwrite")
    for result in recolour_dataset(args.source, args.output, colour, args.workers):
        print(json.dumps(result))
    episodes = read_episodes(args.source).index
    report = verify(args.source, args.output, [int(episodes[0]), int(episodes[len(episodes) // 2]), int(episodes[-1])])
    args.output.with_name(args.output.name + "_verification.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({
        "videos_ok": all(v["same_timestamps"] and v["decoded"] == v["frames"][1] for v in report["videos"]),
        "tables_identical": all(t["identical"] for t in report["tables"]),
        "aligned": all(a["aligned"] for a in report["alignment"]),
    }))


if __name__ == "__main__":
    main()
