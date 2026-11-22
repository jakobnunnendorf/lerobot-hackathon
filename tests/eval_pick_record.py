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


class FakeRobot:
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


def run_policy(name: str, dataset, robot, features, calls):
    path, label = ep.resolve_checkpoint(POLICIES[name])
    policy = ep.Policy(path, None)
    print(f"{name}: {label} type={policy.type} n_action_steps={policy.n_action_steps}", flush=True)
    ticks = iter(range(TICKS))
    ep.key_pressed = lambda: None if next(ticks, None) is not None else " "
    buffer = ep.EpisodeBuffer()
    robot.sent.clear()
    outcome = ep.run_once(robot, policy, features, 30.0, ep.TASK, buffer)
    assert outcome["end"] == "operator" and len(robot.sent) == TICKS, (outcome, len(robot.sent))
    assert all(np.isfinite(list(a.values())).all() for a in robot.sent)
    index = ep.save_episode(dataset, buffer, ep.TASK)
    calls.append((name, index, TICKS, outcome["late_ticks_pct"]))


def main():
    robot = FakeRobot()
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ds"
        dataset = ep.open_dataset(out, features, robot.robot_type, "local/eval_pick_test")
        calls: list = []
        for name in POLICIES:
            run_policy(name, dataset, robot, features, calls)
        dataset.finalize()
        reopened = LeRobotDataset("local/eval_pick_test", root=out)
        assert reopened.num_episodes == 2 and reopened.num_frames == 2 * TICKS, (reopened.num_episodes, reopened.num_frames)
        assert reopened.meta.tasks.index[0] == ep.TASK
        item = reopened[0]
        assert item["observation.images.top"].shape[0] == 3 and item["action"].shape[0] == 6
        resumed = ep.open_dataset(out, features, robot.robot_type, "local/eval_pick_test")
        assert resumed.num_episodes == 2
        print("PASS", calls)


if __name__ == "__main__":
    main()
