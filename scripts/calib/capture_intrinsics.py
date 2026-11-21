import argparse
import time
from pathlib import Path

import cv2

from board import Board, detect_grid, grid_motion
from common import (CONFIG_PATH, QUIT_KEYS, WORK_DIR, Coverage, LatestFrame, Window, camera_settings, draw_text,
                    load_config, open_camera)

INTRINSICS_FRAMES_DIR = WORK_DIR / "intrinsics"
STEADY_PX = 1.0


def saved_frames(folder):
    return sorted(folder.glob("board_*.png"))


def next_frame_path(folder):
    numbers = [int(path.stem.split("_")[1]) for path in saved_frames(folder)]
    return folder / f"board_{max(numbers, default=-1) + 1:03d}.png"


def coverage_of(paths, board, settings):
    coverage = Coverage(settings["width"], settings["height"])
    for path in paths:
        grid = detect_grid(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), board)
        if grid is not None:
            coverage.add(grid)
    return coverage


def draw_view(frame, grid, steady, coverage, board, count, target, message, detect_ms):
    view = coverage.overlay(frame)
    if grid is not None:
        cv2.drawChessboardCorners(view, (board.long_corners, board.short_corners),
                                  grid.reshape(-1, 1, 2).astype("float32"), True)
    state = "board found, holding still: SPACE saves" if steady else ("board found, hold it still" if grid is not None else "no board")
    return draw_text(view, [f"saved {count}/{target}   {state}   ({detect_ms:.0f} ms)",
                            f"image cells never covered (red): {coverage.empty_cells()}   q quits",
                            message])


def main():
    parser = argparse.ArgumentParser(description="Live top-camera view for intrinsic calibration frames. "
                                                 "SPACE saves the frame when the whole board is found and still, q quits.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--camera", help="index or device path, instead of camera.index_or_path in the config")
    parser.add_argument("--out-dir", type=Path, default=INTRINSICS_FRAMES_DIR)
    parser.add_argument("--target", type=int, default=25)
    parser.add_argument("--scale", type=float, default=1.0, help="window size factor")
    args = parser.parse_args()

    config = load_config(args.config)
    board = Board.from_config(config)
    settings = camera_settings(config, args.camera)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = len(saved_frames(args.out_dir))
    coverage = coverage_of(saved_frames(args.out_dir), board, settings)
    print(f"{count} frames already in {args.out_dir}; new ones are added")

    camera = LatestFrame(open_camera(settings))
    window = Window("capture intrinsics", args.scale)
    previous, number, message = None, 0, "move the board: centre, all four corners and edges, near, far, tilted 30-45 deg"
    try:
        while True:
            frame, number = camera.latest(after=number)
            started = time.perf_counter()
            grid = detect_grid(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board)
            detect_ms = 1000 * (time.perf_counter() - started)
            steady = grid is not None and previous is not None and grid_motion(grid, previous) < STEADY_PX
            previous = grid
            key = window.show(draw_view(frame, grid, steady, coverage, board, count, args.target, message, detect_ms))
            if key in QUIT_KEYS:
                break
            if key == ord(" "):
                if steady:
                    path = next_frame_path(args.out_dir)
                    cv2.imwrite(str(path), frame)
                    coverage.add(grid)
                    count += 1
                    message = f"saved {path.name}"
                else:
                    message = "NOT saved: the whole board must be found and held still"
    finally:
        camera.close()
        cv2.destroyAllWindows()
    print(f"{count} frames in {args.out_dir}, {coverage.empty_cells()} image cells never covered by the board")


if __name__ == "__main__":
    main()
