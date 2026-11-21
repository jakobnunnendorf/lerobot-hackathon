import sys
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = tuple(f"{motor}.pos" for motor in MOTORS)
SCRIPTS = Path(__file__).resolve().parents[1]


def pose_of(values: dict) -> np.ndarray:
    return np.array([float(values[key]) for key in KEYS])


def action_of(pose: np.ndarray) -> dict:
    return dict(zip(KEYS, (float(v) for v in pose)))


def step_towards(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    delta = goal - current
    largest = np.abs(delta).max(initial=0.0)
    return goal.copy() if largest <= max_step else current + delta * max_step / largest


def import_from(directory: Path, module: str):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return __import__(module)


class Skill(Protocol):
    name: str

    def start(self, obs: dict, last_sent: dict, now: float) -> None: ...
    def step(self, obs: dict, now: float) -> dict | None: ...
    def done(self, obs: dict, now: float) -> bool: ...
    def result(self) -> str: ...


class MoveTo:
    def __init__(self, name: str, target: np.ndarray, max_step_deg: float = 1.5, gripper_step: float = 2.0):
        self.name, self.target = name, np.asarray(target, dtype=float)
        self.max_step_deg, self.gripper_step = max_step_deg, gripper_step

    def start(self, obs, last_sent, now):
        self.commanded = pose_of(last_sent)

    def step(self, obs, now):
        arm = step_towards(self.commanded[:5], self.target[:5], self.max_step_deg)
        gripper = step_towards(self.commanded[5:], self.target[5:], self.gripper_step)
        self.commanded = np.concatenate([arm, gripper])
        return action_of(self.commanded)

    def done(self, obs, now):
        return bool(np.allclose(self.commanded, self.target))

    def result(self):
        return "arrived"


class Sequence:
    def __init__(self, name: str, parts: list):
        self.name, self.parts = name, parts

    def start(self, obs, last_sent, now):
        self.index, self.last_sent = 0, last_sent
        self.parts[0].start(obs, last_sent, now)

    def step(self, obs, now):
        action = self.parts[self.index].step(obs, now)
        self.last_sent = action or self.last_sent
        if self.parts[self.index].done(obs, now) and self.index + 1 < len(self.parts):
            self.index += 1
            self.parts[self.index].start(obs, self.last_sent, now)
        return action

    def done(self, obs, now):
        return self.index == len(self.parts) - 1 and self.parts[-1].done(obs, now)

    def result(self):
        return self.parts[-1].result()


def put_down_at(pose_holding: np.ndarray, gripper_open: float) -> Sequence:
    opened = np.concatenate([pose_holding[:5], [gripper_open]])
    return Sequence("put_down", [MoveTo("put_down_approach", pose_holding), MoveTo("put_down_open", opened)])


class Replay:
    def __init__(self, name: str, actions: np.ndarray, max_step_deg: float = 1.5):
        self.name, self.actions, self.max_step_deg = name, np.asarray(actions, dtype=float), max_step_deg

    def start(self, obs, last_sent, now):
        self.commanded, self.frame = pose_of(last_sent), -1

    # The recorded first frame may be away from where the arm is: approach it speed-limited, gripper held.
    def step(self, obs, now):
        if self.frame < 0:
            first = self.actions[0]
            arm = step_towards(self.commanded[:5], first[:5], self.max_step_deg)
            self.commanded = np.concatenate([arm, self.commanded[5:]])
            if np.allclose(arm, first[:5]):
                self.frame = 0
            return action_of(self.commanded)
        self.commanded = self.actions[self.frame]
        self.frame += 1
        return action_of(self.commanded)

    def done(self, obs, now):
        return self.frame >= len(self.actions)

    def result(self):
        return "replayed"


def load_episode_actions(repo_id: str, episode: int, root: str | None = None) -> np.ndarray:
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import ACTION

    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode])
    names = dataset.features[ACTION]["names"]
    order = [names.index(key) for key in KEYS]
    actions = dataset.select_columns(ACTION)
    return np.array([np.asarray(actions[i][ACTION])[order] for i in range(dataset.num_frames)])


class LoadedPolicy:
    def __init__(self, path: str, device: str | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(path)
        if device:
            config.device = device
        # Same loading as lerobot-rollout's _load_pretrained_policy (0.7.0) without the PEFT branch.
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config)
        # predict_action reads device.type, so it needs a torch.device, not the config's str.
        self.device = torch.device(self.policy.config.device)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": self.device.type}},
        )

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()


def dataset_features(robot) -> dict:
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    return {**hw_to_dataset_features(robot.action_features, ACTION),
            **hw_to_dataset_features(robot.observation_features, OBS_STR)}


# A policy never reports that it is done (CLAUDE.md), so it runs for its time budget unless `finished`
# says the arm has settled in the end pose.
class PolicySkill:
    def __init__(self, name: str, loaded: LoadedPolicy, features: dict, robot_type: str, task: str = "",
                 finished=None):
        self.name, self.loaded, self.features, self.robot_type, self.task = name, loaded, features, robot_type, task
        self.finished = finished

    def start(self, obs, last_sent, now):
        self.loaded.reset()
        self.settled = False

    def step(self, obs, now):
        from lerobot.common.control_utils import predict_action
        from lerobot.policies import make_robot_action
        from lerobot.utils.constants import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        frame = build_dataset_frame(self.features, obs, prefix=OBS_STR)
        policy = self.loaded.policy
        action = predict_action(frame, policy, self.loaded.device, self.loaded.pre, self.loaded.post,
                                policy.config.use_amp, task=self.task, robot_type=self.robot_type)
        return make_robot_action(action, self.features)

    def done(self, obs, now):
        self.settled = self.finished is not None and self.finished(obs, now)
        return self.settled

    def result(self):
        return "settled"


class Settled:
    def __init__(self, pose: np.ndarray, gripper_range: tuple[float, float], joint_tol_deg: float, hold_s: float,
                 min_runtime_s: float):
        self.pose, self.gripper_range, self.tol, self.hold_s, self.min_runtime_s = pose, gripper_range, joint_tol_deg, hold_s, min_runtime_s
        self.started, self.since = None, None

    def __call__(self, obs: dict, now: float) -> bool:
        self.started = now if self.started is None else self.started
        values = pose_of(obs)
        low, high = self.gripper_range
        there = np.abs(values[:5] - self.pose[:5]).max() <= self.tol and low <= values[5] <= high
        self.since = (now if self.since is None else self.since) if there else None
        return now - self.started >= self.min_runtime_s and self.since is not None and now - self.since >= self.hold_s


class TopCameraVisibility:
    def __init__(self, min_area_px: int = 150):
        self.min_area_px = min_area_px
        self.target_mask = import_from(SCRIPTS / "recolour", "target_mask").target_mask

    def __call__(self, rgb: np.ndarray) -> int:
        return int(np.count_nonzero(self.target_mask(rgb, "top")))


class _GiveStopped(Exception):
    def __init__(self, result: str):
        self.result = result


class _HandoverArm:
    def __init__(self, arm):
        self.arm = arm

    def read(self):
        values = pose_of(self.arm.observe())
        return values[:5], values[5]

    def top_frame(self):
        return self.arm.last_obs["top"]

    def send(self, joints_deg, gripper):
        self.arm.send(action_of(np.concatenate([joints_deg, [gripper]])))


# GIVE runs the #13 hand-following loop in this process on the supervisor's robot connection, so its
# safety rules (gripper hold every tick, workspace clamp, guarded steps) apply unchanged. Why not a
# subprocess: decisions/supervisor.md, "Robot ownership".
class GiveSkill:
    name = "give"

    def __init__(self, arm, handover_dir: Path, receiver, abort: threading.Event, timeout_s: float,
                 gripper_open: float, frame_writer=None, fps: int = 30):
        self.arm, self.receiver, self.abort, self.timeout_s = arm, receiver, abort, timeout_s
        self.gripper_open, self.frame_writer, self.fps = gripper_open, frame_writer, fps
        self.loop = import_from(handover_dir, "handover_loop")
        self.following = import_from(handover_dir, "following")
        self._result = "timeout"
        self._done = False

    def start(self, obs, last_sent, now):
        self.gripper_hold = float(last_sent["gripper.pos"])
        self.home = pose_of(last_sent)[:5]
        self._result, self._done = "timeout", False

    # The loop's HOME is the pose GIVE starts from, which give_sequence makes GIVE_START, not the rest pose.
    def _on_tick(self, t, decision, commanded, gripper, tip, goal_tip):
        Mode = self.following.Mode
        if self.abort.is_set():
            raise _GiveStopped("aborted")
        if decision.mode == Mode.DONE and abs(gripper - self.gripper_open) < 0.5:
            raise _GiveStopped("dropped")
        if decision.mode == Mode.HOME:
            raise _GiveStopped("hand_lost")

    def step(self, obs, now):
        follower = self.following.Follower(self.following.FollowSettings())
        try:
            self.loop.run(_HandoverArm(self.arm), self.receiver, follower, self.home, self.gripper_hold,
                          self.gripper_open, self.loop.Limits(), self.fps, self.timeout_s, self._on_tick,
                          self.frame_writer)
        except _GiveStopped as stop:
            self._result = stop.result
        self._done = True

    def done(self, obs, now):
        return self._done

    def result(self):
        return self._result


# The #13 loop moves the tip in clamped Cartesian steps (0.12-0.38 m from the pan axis, 0.08-0.30 m high), so
# from the folded rest pose (tip 3.4 cm behind the axis) it never gets going: tried in the sim, it held for the
# whole timeout. GIVE therefore first moves in joint space to a pose inside that workspace; board 7061460387
# gives this one as tip (0.15, 0, 0.20).
GIVE_START = (0.0, -55.3, 5.6, 91.8, 0.0)


def give_sequence(arm, give_start=GIVE_START, **give_kwargs) -> Sequence:
    held = np.concatenate([give_start, [arm.last_sent["gripper.pos"]]])
    return Sequence("give", [MoveTo("to_give_start", held), GiveSkill(arm, **give_kwargs)])
