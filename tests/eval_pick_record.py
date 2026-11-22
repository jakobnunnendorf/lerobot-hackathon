# Written by Claude
# Offline test of scripts/eval/eval_pick.py with a fake robot: real ACT and SmolVLA checkpoints on the CPU, about 10
# control ticks each, two short episodes recorded to a temp dir. Usage: uv run python -I tests/eval_pick_record.py  (-I: tests/cuda.py would shadow torch's cuda)
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_pick", root / "scripts/eval/eval_pick.py")
ep = importlib.util.module_from_spec(spec)
sys.modules["eval_pick"] = ep
spec.loader.exec_module(ep)

TICKS = 10
POLICIES = {"act": "Space0code/act_redblue_from_foil_v2@20000", "vla": "jakobnunnendorf/smolvla_redblue_v2@003000"}


class FakeBus:
    motors = dict.fromkeys(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"], 0)
    ranges = {"Present_Current": 40, "Present_Load": 120, "Present_Temperature": 36, "Present_Voltage": 74}

    def __init__(self):
        self.calls = 0
        self.fail_on_call = 5  # one failing read, to prove the NaN path

    def sync_read(self, register, normalize=True):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise ConnectionError("no status packet")
        return {m: self.ranges[register] + i for i, m in enumerate(self.motors)}


class FakeRobot:
    bus = FakeBus()

    robot_type = "so101_follower"
    action_features = {k: float for k in ep.HOME}
    observation_features = {**{k: float for k in ep.HOME}, "top": (480, 640, 3), "wrist": (480, 640, 3)}

    def __init__(self):
        self.state = dict(ep.HOME)
        self.sent = []
        self.rng = np.random.default_rng(0)

    def get_observation(self):
        img = lambda: self.rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        return {**self.state, "top": img(), "wrist": img()}

    def send_action(self, action):
        self.sent.append(action)
        self.state = {**self.state, **action}


def run_policy(name: str, dataset, robot, features, calls, telemetry=None):
    path, label = ep.resolve_checkpoint(POLICIES[name])
    policy = ep.Policy(path, None)
    print(f"{name}: {label} type={policy.type} n_action_steps={policy.n_action_steps}", flush=True)
    ticks = iter(range(TICKS))
    ep.key_pressed = lambda: None if next(ticks, None) is not None else " "
    buffer = ep.EpisodeBuffer()
    robot.sent.clear()
    outcome = ep.run_once(robot, policy, features, 30.0, ep.TASK, buffer, telemetry)
    assert outcome["end"] == "operator" and len(robot.sent) == TICKS, (outcome, len(robot.sent))
    assert all(np.isfinite(list(a.values())).all() for a in robot.sent)
    index = ep.save_episode(dataset, buffer, ep.TASK)
    calls.append((name, index, TICKS, outcome["late_ticks_pct"]))
    return telemetry.stats() if telemetry else None


def main():
    robot = FakeRobot()
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ds"
        record_features = {**features, **ep.telemetry_features(list(robot.bus.motors))}
        dataset = ep.open_dataset(out, record_features, robot.robot_type, "local/eval_pick_test")
        calls: list = []
        stats = [run_policy(name, dataset, robot, features, calls, ep.Telemetry(robot.bus)) for name in POLICIES]
        dataset.finalize()
        reopened = LeRobotDataset("local/eval_pick_test", root=out)
        assert reopened.num_episodes == 2 and reopened.num_frames == 2 * TICKS, (reopened.num_episodes, reopened.num_frames)
        assert reopened.meta.tasks.index[0] == ep.TASK
        item = reopened[0]
        assert item["observation.images.top"].shape[0] == 3 and item["action"].shape[0] == 6
        data = reopened.hf_dataset.with_format("numpy")[:]
        for key, width in [("current", 6), ("load", 6), ("temperature", 6), ("voltage", 6), ("wall_time", 1), ("tick_ms", 1)]:
            col = data[f"observation.motor_{key}" if width == 6 else f"observation.{key}"]
            assert col.reshape(2 * TICKS, -1).shape[1] == width, (key, col.shape)
            assert list(reopened.meta.features[f"observation.motor_{key}" if width == 6 else f"observation.{key}"]["shape"]) == [width]
        current = data["observation.motor_current"]
        assert np.isnan(current).all(axis=1).sum() == 1, "exactly the one failed read is NaN"
        assert np.isfinite(data["observation.motor_temperature"][1:TICKS]).all()
        wall = data["observation.wall_time"][:TICKS].reshape(-1)
        assert (np.diff(wall) > 0).all() and wall[0] >= 0
        assert np.isnan(data["observation.tick_ms"][0]).all() and np.isfinite(data["observation.tick_ms"][1]).all()
        assert stats[0]["tick_ms_max"] >= stats[0]["tick_ms_mean"] > 0 and stats[0]["temperature_start"]["gripper"] == 41
        resumed = ep.open_dataset(out, record_features, robot.robot_type, "local/eval_pick_test")
        assert resumed.num_episodes == 2
        try:
            ep.open_dataset(out, features, robot.robot_type, "local/eval_pick_test")
            raise AssertionError("old feature set must not append to a telemetry dataset")
        except SystemExit:
            pass
        plain = ep.open_dataset(Path(tmp) / "plain", features, robot.robot_type, "local/eval_pick_plain")
        assert not any("motor_" in k or k.endswith(("wall_time", "tick_ms")) for k in plain.meta.features)
        import time

        bus = FakeBus()
        bus.fail_on_call = -1
        t0 = time.perf_counter()
        for _ in range(1000):
            bus.sync_read("Present_Current", normalize=False)
        print(f"fake sync_read: {(time.perf_counter() - t0) * 1000:.4f} ms per call (Python only, no serial)")
        print("PASS", calls)


if __name__ == "__main__":
    main()
