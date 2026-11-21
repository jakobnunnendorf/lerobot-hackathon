import argparse
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

EXPECTED_LEROBOT = "0.7.0"
FOLLOWER_PORT = "/dev/so101_follower"
FOLLOWER_ID = "mi_follower"
CAMERA_PATHS = {
    "top": "/dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
}
FRAME_SIZE = (640, 480)
POLICY = "molnrb/act_pick_ball_combined"
POLICY_CAMERAS = ("top", "wrist")
TRACKER_PORT = 8766
VOICE_PORT = 8765
MOTORS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}
DISK_WARN_GB, DISK_FAIL_GB = 20, 5


@dataclass
class Result:
    status: str
    message: str
    fix: str = ""
    details: tuple[str, ...] = ()


def check_python_and_lerobot(args) -> Result:
    python = ".".join(map(str, sys.version_info[:3]))
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        return Result(FAIL, f"python {python}, lerobot not installed", "run from the repo env: source .venv/bin/activate")
    if version != EXPECTED_LEROBOT:
        return Result(FAIL, f"python {python}, lerobot {version}", f"install lerobot=={EXPECTED_LEROBOT} (uv sync)")
    if sys.version_info < (3, 12):
        return Result(FAIL, f"python {python}, lerobot {version}", "use python 3.12 or newer")
    return Result(PASS, f"python {python}, lerobot {version}")


def check_cuda(args) -> Result:
    try:
        import torch
    except ImportError:
        return Result(FAIL, "torch is not installed", "install the CUDA 12.8 torch wheel (uv sync)")
    if not torch.cuda.is_available():
        return Result(FAIL, f"torch {torch.__version__}: CUDA not available", "in WSL check nvidia-smi; reinstall the CUDA wheel")
    properties = torch.cuda.get_device_properties(0)
    return Result(PASS, f"{properties.name}, {properties.total_memory / 1e9:.1f} GB, torch {torch.__version__}")


def follower_port(args) -> str:
    return args.port or os.environ.get("SO101_FOLLOWER_PORT") or FOLLOWER_PORT


def port_exists(port: str) -> bool:
    if os.path.exists(port):
        return True
    if port.upper().startswith("COM"):
        try:
            from serial.tools import list_ports
        except ImportError:
            return False
        return port.upper() in {p.device.upper() for p in list_ports.comports()}
    return False


def check_port(args) -> Result:
    port = follower_port(args)
    if port_exists(port):
        return Result(PASS, f"arm port {port} exists")
    visible = sorted(str(p) for p in Path("/dev").glob("ttyACM*")) if Path("/dev").exists() else []
    seen = f" (serial devices seen: {', '.join(visible)})" if visible else ""
    return Result(FAIL, f"arm port {port} not found{seen}", "re-attach USB to WSL: bash scripts/so101/attach_usb.sh")


def calibration_file(args) -> Path:
    if args.calibration_file:
        return Path(args.calibration_file).expanduser()
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

    # SO-101 followers use the "so_follower" folder in 0.7.0
    return HF_LEROBOT_CALIBRATION / ROBOTS / "so_follower" / f"{args.robot_id}.json"


def check_calibration(args) -> Result:
    path = calibration_file(args)
    fix = f"copy {args.robot_id}.json from the laptop that recorded; never recalibrate"
    if not path.is_file():
        return Result(FAIL, f"{path} not found", fix)
    try:
        data = json.loads(path.read_text())
        ranges = {name: (int(data[name]["range_min"]), int(data[name]["range_max"])) for name in MOTORS}
    except (ValueError, KeyError, TypeError) as error:
        return Result(FAIL, f"{path} is not a valid calibration ({error!r})", fix)
    lines = tuple(
        f"{name:14s} raw {low:4d}..{high:4d}  ({(high - low) * 360 / 4095:5.1f} deg)" for name, (low, high) in ranges.items()
    )
    broken = [name for name, (low, high) in ranges.items() if not 0 <= low < high <= 4095]
    if broken:
        return Result(FAIL, f"{path.name}: bad range for {', '.join(broken)}", fix, lines)
    return Result(PASS, f"{path}", details=lines)


def camera_source(args, name: str) -> str | int:
    given = getattr(args, f"{name}_camera") or os.environ.get(f"SO101_CAM_{name.upper()}") or CAMERA_PATHS[name]
    return int(given) if str(given).isdigit() else given


def check_camera(args, name: str) -> Result:
    import cv2

    source = camera_source(args, name)
    if isinstance(source, str) and not os.path.exists(source):
        return Result(FAIL, f"{name} camera {source} not found", "re-attach USB to WSL: bash scripts/so101/attach_usb.sh")
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    try:
        if not capture.isOpened():
            return Result(FAIL, f"{name} camera {source} does not open", "close other programs using it, then re-plug")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        frame = None
        for _ in range(10):
            ok, frame = capture.read()
            if ok and frame is not None:
                break
        if frame is None or not ok:
            return Result(FAIL, f"{name} camera opens but returns no frame", "re-plug the camera, then attach_usb.sh")
        height, width = frame.shape[:2]
        if (width, height) != FRAME_SIZE:
            return Result(FAIL, f"{name} camera gives {width}x{height}", "the policy needs 640x480; try another port or cable")
        return Result(PASS, f"{name} camera returns a {width}x{height} frame")
    finally:
        capture.release()


def check_camera_top(args) -> Result:
    return check_camera(args, "top")


def check_camera_wrist(args) -> Result:
    return check_camera(args, "wrist")


def check_policy(args) -> Result:
    import lerobot.policies  # registers the policy types, otherwise from_pretrained finds none
    from lerobot.configs.policies import PreTrainedConfig

    try:
        config = PreTrainedConfig.from_pretrained(args.policy)
    except Exception as error:
        return Result(FAIL, f"cannot read config of {args.policy} ({type(error).__name__})", "check the name, network and hf login")
    cameras = {key.split(".")[-1]: feature.shape for key, feature in config.image_features.items()}
    missing = [name for name in POLICY_CAMERAS if name not in cameras]
    if missing:
        return Result(FAIL, f"{args.policy} expects cameras {sorted(cameras)}, missing {missing}", "use a checkpoint trained with top and wrist")
    wrong = {name: shape for name, shape in cameras.items() if tuple(shape[1:]) != FRAME_SIZE[::-1]}
    if wrong:
        return Result(FAIL, f"{args.policy} expects image shapes {wrong}", "cameras must give 640x480")
    if not args.load_policy:
        return Result(PASS, f"{args.policy}: {config.type} config ok (weights not loaded; --load-policy loads them on the GPU)")
    import torch

    if not torch.cuda.is_available():
        return Result(FAIL, f"{args.policy}: config ok but no CUDA to load the weights", "fix the cuda check first")
    from lerobot.policies.factory import get_policy_class

    policy = get_policy_class(config.type).from_pretrained(args.policy).to("cuda")
    memory = torch.cuda.memory_allocated() / 1e9
    del policy
    return Result(PASS, f"{args.policy}: {config.type} weights loaded on the GPU ({memory:.2f} GB)")


def check_port_open(host: str, port: int, label: str, fix: str, strict: bool) -> Result:
    try:
        socket.create_connection((host, port), timeout=2).close()
    except OSError:
        # Preflight runs before the programs start, so a closed port is normal until they are up
        return Result(FAIL if strict else WARN, f"nothing accepting on {host}:{port}", fix)
    return Result(PASS, f"accepting on {host}:{port}")


def check_tracker(args) -> Result:
    return check_port_open(
        "127.0.0.1", TRACKER_PORT, "tracker link", "start the robot program (it listens, the tracker connects)", args.strict_services
    )


def check_voice(args) -> Result:
    return check_port_open(
        args.voice_host, VOICE_PORT, "voice link", "start scripts/voice/link.py (in WSL bind 0.0.0.0, see voice-runbook)", args.strict_services
    )


def check_disk(args) -> Result:
    folder = Path(args.data_dir).expanduser()
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    free = shutil.disk_usage(folder).free / 1e9
    if free < DISK_FAIL_GB:
        return Result(FAIL, f"{free:.1f} GB free at {folder}", "free space or move the dataset cache (HF_LEROBOT_HOME)")
    if free < DISK_WARN_GB:
        return Result(WARN, f"{free:.1f} GB free at {folder}", f"under {DISK_WARN_GB} GB: delete old local datasets")
    return Result(PASS, f"{free:.1f} GB free at {folder}")


def check_hf(args) -> Result:
    hf = shutil.which("hf", path=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    if hf is None:
        return Result(FAIL, "hf CLI not found", "use the repo env (uv sync)")
    done = subprocess.run([hf, "auth", "whoami"], capture_output=True, text=True, timeout=args.timeout - 2)
    output = (done.stdout + done.stderr).strip()
    if done.returncode == 0:
        return Result(PASS, f"logged in to Hugging Face: {output.splitlines()[0]}")
    if "not logged in" in output.lower():
        return Result(FAIL, "not logged in to Hugging Face", "hf auth login")
    return Result(WARN, f"hf auth whoami failed: {output[:80]}", "check the network, then hf auth login")


def check_motors(args) -> Result:
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    port = follower_port(args)
    if not port_exists(port):
        return Result(FAIL, f"arm port {port} not found", "fix the arm port check first")
    bus = SerialMotorsBus(port=port, motors={name: Motor(id, "sts3215", MotorNormMode.DEGREES) for name, id in MOTORS.items()})
    try:
        bus.connect()
        raw = bus.sync_read("Present_Position", normalize=False)
    except Exception as error:
        return Result(FAIL, f"motors do not answer on {port} ({type(error).__name__})", "check power, the USB cable, the other program holding the port")
    finally:
        # disable_torque=False: the default disconnect() writes torque off, and this script never writes to the motors
        bus.disconnect(disable_torque=False)
    lines = tuple(f"{name:14s} raw position {value}" for name, value in raw.items())
    return Result(PASS, f"all 6 motors answer on {port} (read only)", details=lines)


CHECKS = {
    "python": (check_python_and_lerobot, "python and lerobot"),
    "cuda": (check_cuda, "CUDA"),
    "port": (check_port, "arm port"),
    "calibration": (check_calibration, "follower calibration"),
    "camera_top": (check_camera_top, "top camera"),
    "camera_wrist": (check_camera_wrist, "wrist camera"),
    "policy": (check_policy, "policy checkpoint"),
    "tracker": (check_tracker, "tracker link"),
    "voice": (check_voice, "voice link"),
    "disk": (check_disk, "disk space"),
    "hf": (check_hf, "HF login"),
    "motors": (check_motors, "motors read"),
}
CHECK_TIMEOUTS = {"policy": 4.0}


def is_skipped(name: str, skipped: set[str]) -> bool:
    return any(name == token or name.startswith(token + "_") for token in skipped)


def run_with_timeout(function, args, timeout: float) -> Result:
    outcome: list[Result] = []

    def target():
        try:
            outcome.append(function(args))
        except Exception as error:
            outcome.append(Result(FAIL, f"check crashed: {type(error).__name__}: {error}"[:200]))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    return outcome[0] if outcome else Result(FAIL, f"no answer after {timeout:.0f} s", "rerun it alone with a longer --timeout")


def run_checks(args) -> list[tuple[str, Result]]:
    skipped = {token.strip() for token in args.skip.split(",") if token.strip()}
    selected = [name for name in CHECKS if name != "motors" or args.read_motors]
    results = []
    for name in selected:
        function, label = CHECKS[name]
        if is_skipped(name, skipped):
            result = Result(SKIP, "skipped by --skip")
        else:
            started = time.monotonic()
            result = run_with_timeout(function, args, args.timeout * CHECK_TIMEOUTS.get(name, 1.0))
            result.message += f" [{time.monotonic() - started:.1f}s]"
        results.append((label, result))
    if not args.read_motors:
        results.append((CHECKS["motors"][1], Result(SKIP, "off; add --read-motors to read the present positions")))
    return results


def print_report(results: list[tuple[str, Result]]) -> None:
    for label, result in results:
        print(f"{result.status:4s}  {label:22s} {result.message}")
        for line in result.details:
            print(f"        {line}")
        if result.fix and result.status in (WARN, FAIL):
            print(f"        fix: {result.fix}")
    counts = {status: sum(result.status == status for _, result in results) for status in (PASS, WARN, FAIL, SKIP)}
    print(f"\n{counts[PASS]} PASS, {counts[WARN]} WARN, {counts[FAIL]} FAIL, {counts[SKIP]} SKIP")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Read-only setup check for the demo laptop; nothing moves.")
    parser.add_argument("--port", help=f"follower serial port (default: $SO101_FOLLOWER_PORT or {FOLLOWER_PORT})")
    parser.add_argument("--robot-id", default=FOLLOWER_ID)
    parser.add_argument("--calibration-file")
    parser.add_argument("--top-camera", help="path or index (default: $SO101_CAM_TOP or the by-id path)")
    parser.add_argument("--wrist-camera", help="path or index (default: $SO101_CAM_WRIST or the by-id path)")
    parser.add_argument("--policy", default=POLICY, help="Hub id or local pretrained_model folder")
    parser.add_argument("--load-policy", action="store_true", help="load the weights on the GPU, not only the config")
    parser.add_argument("--read-motors", action="store_true", help="open the port and read present positions (no writes)")
    parser.add_argument("--voice-host", default="127.0.0.1")
    parser.add_argument("--strict-services", action="store_true", help="a closed 8766/8765 is FAIL instead of WARN")
    parser.add_argument("--data-dir", default=os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot"))
    parser.add_argument("--skip", default="", help=f"comma list of: {','.join(CHECKS)} (camera skips both)")
    parser.add_argument("--timeout", type=float, default=20.0, help="seconds per check (policy gets 4x)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    results = run_checks(parse_args(argv))
    print_report(results)
    return 1 if any(result.status == FAIL for _, result in results) else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # os._exit: a check stuck in a driver call must not keep the process alive
    os._exit(code)
