import os
import sys
import time

import cv2
import numpy as np

PATTERN = (7, 5)
SECONDS_BETWEEN_SAVES = 2.0
MIN_SHIFT_PX = 25
GRID = (4, 3)

source, folder, wanted = sys.argv[1], sys.argv[2], int(sys.argv[3])
if source.isdigit() or source.startswith("/dev/"):
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_V4L2 if sys.platform == "linux" else cv2.CAP_ANY)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    print("autofocus off accepted by the driver:", capture.set(cv2.CAP_PROP_AUTOFOCUS, 0), flush=True)
else:
    capture = cv2.VideoCapture(source)

os.makedirs(folder, exist_ok=True)
covered = np.zeros(GRID[::-1], bool)
saved, last_corners = 0, None
while saved < wanted:
    deadline = time.time() + SECONDS_BETWEEN_SAVES
    frame = None
    while frame is None or time.time() < deadline:
        ok, frame = capture.read()
        if not ok:
            sys.exit(f"no frame from {source}; saved {saved}")
        if not (source.isdigit() or source.startswith("/dev/")):
            break
    height, width = frame.shape[:2]
    found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), PATTERN)
    if not found:
        print(f"{width}x{height}  board not found: keep all of it in view, hold it still", flush=True)
        continue
    corners = corners.reshape(-1, 2)
    if last_corners is not None and np.abs(corners - last_corners).mean() < MIN_SHIFT_PX:
        print(f"{width}x{height}  same pose as the last saved frame: move or tilt the board", flush=True)
        continue
    cv2.imwrite(os.path.join(folder, f"board_{saved:02d}.png"), frame)
    saved, last_corners = saved + 1, corners
    cells = np.minimum((corners / [width, height] * GRID).astype(int), np.array(GRID) - 1)
    covered[cells[:, 1], cells[:, 0]] = True
    rows = " / ".join("".join("#" if c else "." for c in row) for row in covered)
    print(f"\a{width}x{height}  saved {saved}/{wanted}   image areas covered so far (top row first): {rows}", flush=True)
capture.release()
