import os
import sys
import time

import cv2

camera, folder, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
capture = cv2.VideoCapture(int(camera) if camera.isdigit() else camera)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FPS, 30)
print("autofocus off accepted:", capture.set(cv2.CAP_PROP_AUTOFOCUS, 0))
os.makedirs(folder, exist_ok=True)
saved = 0
while saved < count:
    deadline = time.time() + 1.5
    while time.time() < deadline:
        ok, frame = capture.read()
        if not ok:
            sys.exit("no frame from the camera")
    found, _ = cv2.findChessboardCornersSB(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 6))
    if found:
        cv2.imwrite(os.path.join(folder, f"board_{saved:02d}.png"), frame)
        saved += 1
    print(f"{frame.shape[1]}x{frame.shape[0]}", f"saved {saved}/{count}" if found else "board not found, keep it fully in view", flush=True)
capture.release()
