import json
import re
import socket
import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preflight  # noqa: E402

CALIBRATION = {
    name: {"id": i + 1, "drive_mode": 0, "homing_offset": 10, "range_min": 800, "range_max": 3200}
    for i, name in enumerate(preflight.MOTORS)
}


def args(**overrides):
    values = preflight.parse_args([])
    for key, value in overrides.items():
        setattr(values, key, value)
    return values


def fake_torch(cuda: bool):
    torch = types.SimpleNamespace()
    torch.__version__ = "2.11.0+cu128"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_properties=lambda index: types.SimpleNamespace(name="NVIDIA GeForce RTX 4050 Laptop GPU", total_memory=6.4e9),
        memory_allocated=lambda: 0.2e9,
    )
    return torch


def listening_port():
    server = socket.create_server(("127.0.0.1", 0))
    threading.Thread(target=lambda: [server.accept()[0].close() for _ in range(3)], daemon=True).start()
    return server, server.getsockname()[1]


def test_versions_and_cuda():
    with mock.patch.object(preflight.importlib.metadata, "version", return_value="0.7.0"):
        assert preflight.check_python_and_lerobot(args()).status == preflight.PASS
    with mock.patch.object(preflight.importlib.metadata, "version", return_value="0.6.1"):
        assert preflight.check_python_and_lerobot(args()).status == preflight.FAIL
    with mock.patch.dict(sys.modules, {"torch": fake_torch(True)}):
        result = preflight.check_cuda(args())
        assert result.status == preflight.PASS and "RTX 4050" in result.message
    with mock.patch.dict(sys.modules, {"torch": fake_torch(False)}):
        assert preflight.check_cuda(args()).status == preflight.FAIL


def test_port_and_calibration():
    with tempfile.TemporaryDirectory() as folder:
        port = Path(folder) / "so101_follower"
        port.touch()
        assert preflight.check_port(args(port=str(port))).status == preflight.PASS
        assert preflight.check_port(args(port=str(port) + "x")).status == preflight.FAIL

        good = Path(folder) / "mi_follower.json"
        good.write_text(__import__("json").dumps(CALIBRATION))
        result = preflight.check_calibration(args(calibration_file=str(good)))
        assert result.status == preflight.PASS and len(result.details) == 6, result
        assert "shoulder_pan" in result.details[0]

        bad = Path(folder) / "bad.json"
        bad.write_text("{}")
        assert preflight.check_calibration(args(calibration_file=str(bad))).status == preflight.FAIL
        assert preflight.check_calibration(args(calibration_file=str(good) + "x")).status == preflight.FAIL


class FakeCapture:
    shape = (480, 640, 3)

    def __init__(self, *_):
        pass

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        return True, np.zeros(self.shape, dtype=np.uint8)

    def release(self):
        pass


def test_cameras():
    with mock.patch("cv2.VideoCapture", FakeCapture), mock.patch.object(preflight.os.path, "exists", return_value=True):
        assert preflight.check_camera_top(args()).status == preflight.PASS
        assert "640x480" in preflight.check_camera_wrist(args()).message
        with mock.patch.object(FakeCapture, "shape", (720, 1280, 3)):
            assert preflight.check_camera_top(args()).status == preflight.FAIL
    assert preflight.check_camera_top(args(top_camera="/nonexistent/camera")).status == preflight.FAIL


NEW_NAMES = """\
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=300 flags=inactive
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
        white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4600 value=4200 flags=inactive
"""
OLD_NAMES = """\
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
              exposure_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=250 flags=inactive
 white_balance_temperature_auto 0x0098090c (bool)   : default=1 value=1
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4600 value=5000 flags=inactive
"""


class FakeV4l2:
    def __init__(self, listing: str):
        self.listing, self.calls = listing, []

    def __call__(self, device, *arguments):
        self.calls.append(arguments)
        if arguments == ("--list-ctrls",):
            return self.listing
        for argument in arguments[1::2]:
            name, value = argument.split("=")
            self.listing = re.sub(rf"({name} .*value=)-?\d+", rf"\g<1>{value}", self.listing)
        return ""


def test_exposure_controls():
    for listing, names in ((NEW_NAMES, ("auto_exposure=1", "exposure_time_absolute=300", "white_balance_automatic=0", "white_balance_temperature=4200")),
                           (OLD_NAMES, ("exposure_auto=1", "exposure_absolute=250", "white_balance_temperature_auto=0", "white_balance_temperature=5000"))):
        fake = FakeV4l2(listing)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(preflight, "v4l2_ctl", fake), mock.patch.object(
            preflight.shutil, "which", return_value="/usr/bin/v4l2-ctl"
        ), mock.patch.object(preflight.sys, "platform", "linux"), mock.patch.object(preflight.os.path, "exists", return_value=True):
            controls_file = str(Path(folder) / "controls.json")
            result = preflight.check_exposure(args(controls_file=controls_file))
            assert result.status == preflight.WARN and "top auto exposure" in result.message, result
            assert any("value=" not in line and "=" in line for line in result.details)

            result = preflight.check_exposure(args(controls_file=controls_file, lock_exposure=True))
            assert result.status == preflight.PASS, result
            setting = [call for call in fake.calls if call[:1] == ("-c",)][0]
            assert [setting[i] for i in range(1, len(setting), 2)] == list(names), setting
            saved = json.loads(Path(controls_file).read_text())
            assert set(saved) == {"top", "wrist"}

            result = preflight.check_exposure(args(controls_file=controls_file, unlock_exposure=True))
            assert result.status == preflight.WARN, result
            restored = [call for call in fake.calls if call[:1] == ("-c",)][2]  # top camera; the fake shares one listing, so wrist saw locked values
            assert restored[1] == names[0].split("=")[0] + "=3" and restored[-1].endswith("=" + names[3].split("=")[1]), restored

    with mock.patch.object(preflight.shutil, "which", return_value=None), mock.patch.object(preflight.sys, "platform", "linux"):
        assert preflight.check_exposure(args()).status == preflight.WARN
    with mock.patch.object(preflight.sys, "platform", "win32"):
        assert preflight.check_exposure(args()).status == preflight.WARN


def test_policy():
    feature = types.SimpleNamespace(shape=(3, 480, 640))
    config = types.SimpleNamespace(
        type="act", image_features={"observation.images.top": feature, "observation.images.wrist": feature}
    )
    policy_module = types.ModuleType("lerobot.policies")
    configs_module = types.ModuleType("lerobot.configs.policies")
    configs_module.PreTrainedConfig = types.SimpleNamespace(from_pretrained=lambda name: config)
    policy = mock.Mock()
    factory_module = types.ModuleType("lerobot.policies.factory")
    factory_module.get_policy_class = lambda name: types.SimpleNamespace(from_pretrained=lambda path: policy)
    modules = {
        "lerobot.policies": policy_module,
        "lerobot.configs.policies": configs_module,
        "lerobot.policies.factory": factory_module,
        "torch": fake_torch(True),
    }
    with mock.patch.dict(sys.modules, modules), mock.patch.object(preflight, "policy_path", lambda spec: spec):
        result = preflight.check_policy(args())
        assert result.status == preflight.PASS and "weights not loaded" in result.message
        result = preflight.check_policy(args(load_policy=True))
        assert result.status == preflight.PASS and "on the GPU" in result.message
        policy.to.assert_called_with("cuda")
        config.image_features = {"observation.images.top": feature}
        assert preflight.check_policy(args()).status == preflight.FAIL


def test_services_disk_hf():
    server, port = listening_port()
    with mock.patch.object(preflight, "TRACKER_PORT", port), mock.patch.object(preflight, "VOICE_PORT", port):
        assert preflight.check_tracker(args()).status == preflight.PASS
        assert preflight.check_voice(args()).status == preflight.PASS
    server.close()
    with mock.patch.object(preflight, "TRACKER_PORT", port):
        assert preflight.check_tracker(args()).status == preflight.WARN
        assert preflight.check_tracker(args(strict_services=True)).status == preflight.FAIL

    assert preflight.check_disk(args(data_dir=tempfile.gettempdir() + "/does/not/exist")).status in (preflight.PASS, preflight.WARN)
    with mock.patch.object(preflight.shutil, "disk_usage", return_value=types.SimpleNamespace(free=2e9)):
        assert preflight.check_disk(args()).status == preflight.FAIL

    ok = types.SimpleNamespace(returncode=0, stdout="jakob\n", stderr="")
    out = types.SimpleNamespace(returncode=1, stdout="", stderr="Not logged in")
    with mock.patch.object(preflight.shutil, "which", return_value="hf"):
        with mock.patch.object(preflight.subprocess, "run", return_value=ok):
            assert preflight.check_hf(args()).status == preflight.PASS
        with mock.patch.object(preflight.subprocess, "run", return_value=out):
            assert preflight.check_hf(args()).status == preflight.FAIL


def test_motors_are_only_read():
    bus = mock.Mock()
    bus.sync_read.return_value = {name: 2048 for name in preflight.MOTORS}
    motors_module = types.ModuleType("lerobot.motors")
    motors_module.Motor = lambda *a, **k: None
    motors_module.MotorNormMode = types.SimpleNamespace(DEGREES=1)
    motors_module.SerialMotorsBus = lambda **kwargs: bus
    with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, {"lerobot.motors": motors_module}):
        port = Path(folder) / "port"
        port.touch()
        result = preflight.check_motors(args(port=str(port)))
        assert result.status == preflight.PASS and len(result.details) == 6
    bus.disconnect.assert_called_once_with(disable_torque=False)
    called = {call[0] for call in bus.method_calls}
    assert called <= {"connect", "sync_read", "disconnect"}, called


def test_runner_skip_timeout_exit_code():
    def pass_check(_):
        return preflight.Result(preflight.PASS, "ok")

    def hang(_):
        threading.Event().wait(5)

    def crash(_):
        raise RuntimeError("boom")

    checks = {"camera_top": (pass_check, "t"), "camera_wrist": (pass_check, "w"), "cuda": (crash, "c"), "disk": (hang, "d"), "motors": (hang, "m")}
    with mock.patch.object(preflight, "CHECKS", checks):
        results = dict(preflight.run_checks(args(skip="camera", timeout=0.3)))
        assert results["t"].status == preflight.SKIP and results["w"].status == preflight.SKIP
        assert results["c"].status == preflight.FAIL and "boom" in results["c"].message
        assert results["d"].status == preflight.FAIL and "no answer" in results["d"].message
        assert results["m"].status == preflight.SKIP  # motors stay off without --read-motors
        assert preflight.main(["--skip", "cuda,disk,camera", "--timeout", "0.3"]) == 0
        assert preflight.main(["--skip", "camera,disk", "--timeout", "0.3"]) == 1


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
    print("PASS")
