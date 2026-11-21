import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

CALIB_DIR = Path(__file__).resolve().parent
REPO_DIR = CALIB_DIR.parents[1]
CONFIG_PATH = CALIB_DIR / "config.yaml"
WORK_DIR = REPO_DIR / "data" / "calib"
BACKENDS = {"v4l2": cv2.CAP_V4L2, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
QUIT_KEYS = (ord("q"), 27)


def load_config(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


def camera_settings(config, camera_override=None):
    settings = dict(config["camera"])
    if camera_override is not None:
        settings["index_or_path"] = int(camera_override) if camera_override.isdigit() else camera_override
    return settings


def backend_id(name):
    if name == "auto":
        return cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    return BACKENDS[name]


def open_camera(settings):
    source = settings["index_or_path"]
    capture = cv2.VideoCapture(source, backend_id(settings["backend"]))
    if not capture.isOpened():
        sys.exit(f"Cannot open the camera {source!r}. Stop the robot program first: only one program can use the camera.")
    # Same order as LeRobot 0.7.0's OpenCVCamera: on Windows a size change can reset the pixel format, so it goes last.
    fourcc_last = sys.platform == "win32"
    if not fourcc_last:
        set_fourcc(capture, settings["fourcc"])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings["width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["height"])
    capture.set(cv2.CAP_PROP_FPS, settings["fps"])
    if fourcc_last:
        set_fourcc(capture, settings["fourcc"])
    fix_focus_and_exposure(capture, settings)
    check_frame_size(capture, settings)
    return capture


def set_fourcc(capture, fourcc):
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    actual = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    if actual != fourcc:
        print(f"WARNING: pixel format is {actual!r}, not {fourcc!r} as in the robot program; the field of view may differ.")


def fix_focus_and_exposure(capture, settings):
    # Autofocus changes the focal length, so the calibration holds only with it off. V4L2 keeps the setting after
    # this program exits, which the robot program needs too; replugging the camera can switch it back on.
    if capture.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("autofocus: switched off")
    else:
        print("WARNING: autofocus could not be switched off. A fixed-focus camera has no such control; check with\n"
              "  v4l2-ctl -d <device> --list-ctrls   (look for focus_auto or focus_automatic_continuous)")
    if not settings.get("manual_exposure"):
        print("exposure: left as it is (manual_exposure: false in config.yaml)")
    elif capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1):
        print("exposure: switched to manual")
    else:
        print("WARNING: auto exposure could not be switched off with this backend")


def check_frame_size(capture, settings):
    for _ in range(30):
        ok, frame = capture.read()
        if ok:
            break
    else:
        sys.exit("The camera opened but delivers no frames (in WSL: re-attach it with scripts/so101/attach_usb.sh).")
    size, wanted = (frame.shape[1], frame.shape[0]), (settings["width"], settings["height"])
    if size != wanted:
        sys.exit(f"The camera delivers {size[0]}x{size[1]}, the config asks for {wanted[0]}x{wanted[1]}. "
                 "Calibrate only at the size the robot program uses.")
    print(f"camera: {size[0]}x{size[1]} at {capture.get(cv2.CAP_PROP_FPS):.0f} fps, backend {capture.getBackendName()}")


class LatestFrame:
    # Reads in its own thread so the window always shows the newest frame, even when the corner detection is slower
    # than the camera; otherwise the driver's frame queue makes the view lag by several frames.
    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.frame, self.number, self.error = None, 0, None
        self.running = True
        self.thread = threading.Thread(target=self._read_forever, daemon=True)
        self.thread.start()

    def _read_forever(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                self.error = "The camera stopped delivering frames (in WSL: re-attach it with scripts/so101/attach_usb.sh)."
                return
            with self.lock:
                self.frame, self.number = frame, self.number + 1

    def latest(self, after=0, timeout_s=3.0):
        deadline = time.monotonic() + timeout_s
        while True:
            if self.error:
                sys.exit(self.error)
            with self.lock:
                if self.frame is not None and self.number > after:
                    return self.frame.copy(), self.number
            if time.monotonic() > deadline:
                sys.exit("No new frame from the camera for 3 s.")
            time.sleep(0.005)

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


class Window:
    def __init__(self, name, scale=1.0, on_click=None):
        self.name, self.scale, self.on_click = name, scale, on_click
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        if on_click is not None:
            cv2.setMouseCallback(name, self._mouse)

    def _mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.on_click(x / self.scale, y / self.scale)

    def show(self, image, wait_ms=1):
        if self.scale != 1.0:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.name, image)
        return cv2.waitKey(wait_ms) & 0xFF

    def wait_for(self, image, keys):
        while (key := self.show(image, 50)) not in keys:
            pass
        return key


def draw_text(image, lines, origin=(8, 8), scale=0.5):
    x, y = origin
    for line in lines:
        (w, h), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(image, (x - 4, y - 2), (x + w + 4, y + h + base + 2), (0, 0, 0), -1)
        cv2.putText(image, line, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += h + base + 6
    return image


def to_point(pixel):
    return tuple(int(round(v)) for v in pixel)


class Coverage:
    def __init__(self, width, height, cells=(8, 6)):
        self.cells = cells
        self.cell_size = np.array([width / cells[0], height / cells[1]])
        self.views = np.zeros((cells[1], cells[0]), int)

    def add(self, grid):
        indices = np.floor(grid.reshape(-1, 2) / self.cell_size).astype(int)
        indices = np.clip(indices, 0, np.array(self.cells) - 1)
        for column, row in set(map(tuple, indices)):
            self.views[row, column] += 1

    def empty_cells(self):
        return int((self.views == 0).sum())

    def overlay(self, image, alpha=0.3):
        tint = image.copy()
        colours = {0: (0, 0, 255), 1: (0, 220, 255)}
        for row, column in np.ndindex(self.views.shape):
            top_left = (self.cell_size * (column, row)).astype(int)
            bottom_right = (self.cell_size * (column + 1, row + 1)).astype(int)
            cv2.rectangle(tint, to_point(top_left), to_point(bottom_right), colours.get(self.views[row, column], (0, 200, 0)), -1)
        return cv2.addWeighted(tint, alpha, image, 1 - alpha, 0)
