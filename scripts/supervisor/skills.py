import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from status import STATUS
from ticks import TICKS

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


def capped(action: dict, last_sent: dict, max_step_deg: float | None) -> dict:
    if max_step_deg is None:
        return action
    goal, current = pose_of(action), pose_of(last_sent)
    arm = step_towards(current[:5], goal[:5], max_step_deg)
    return action_of(np.concatenate([arm, goal[5:]]))


class Skill(Protocol):
    name: str

    def start(self, obs: dict, last_sent: dict, now: float) -> None: ...
    def step(self, obs: dict, now: float) -> dict | None: ...
    def done(self, obs: dict, now: float) -> bool: ...
    def result(self) -> str: ...


# Done only when the measured arm joints are there too: a lagging or blocked arm must not start the next skill.
# The gripper is left out, since it reads the ball's width while holding, not its goal.
class MoveTo:
    def __init__(self, name: str, target: np.ndarray, max_step_deg: float = 1.5, gripper_step: float = 2.0,
                 arrive_tol_deg: float = 4.0):
        self.name, self.target = name, np.asarray(target, dtype=float)
        self.max_step_deg, self.gripper_step, self.arrive_tol_deg = max_step_deg, gripper_step, arrive_tol_deg

    def start(self, obs, last_sent, now):
        self.commanded = pose_of(last_sent)

    def step(self, obs, now):
        arm = step_towards(self.commanded[:5], self.target[:5], self.max_step_deg)
        gripper = step_towards(self.commanded[5:], self.target[5:], self.gripper_step)
        self.commanded = np.concatenate([arm, gripper])
        return action_of(self.commanded)

    def done(self, obs, now):
        measured = pose_of(obs)[:5]
        return bool(np.allclose(self.commanded, self.target)
                    and np.abs(measured - self.target[:5]).max() <= self.arrive_tol_deg)

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
    def __init__(self, path: str, device: str | None = None, n_action_steps: int | None = None,
                 camera_rename: dict[str, str] | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(path)
        if device:
            config.device = device
        if n_action_steps:
            config.n_action_steps = n_action_steps
        # Same loading as lerobot-rollout's _load_pretrained_policy (0.7.0) without the PEFT branch.
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config)
        # predict_action reads device.type, so it needs a torch.device, not the config's str.
        self.device = torch.device(self.policy.config.device)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=path,
            preprocessor_overrides={
                "device_processor": {"device": self.device.type},
                # SmolVLA checkpoints trained with Kamel's map expect camera1/camera2 for top/wrist.
                "rename_observations_processor": {"rename_map": {
                    f"observation.images.{ours}": f"observation.images.{theirs}"
                    for ours, theirs in (camera_rename or {}).items()}},
            },
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
        action = make_robot_action(action, self.features)
        TICKS.note(policy=[round(float(action[key]), 2) for key in KEYS])
        return action

    def done(self, obs, now):
        self.settled = self.finished is not None and self.finished(obs, now)
        return self.settled

    def result(self):
        return "settled"


# Same rule as scripts/eval/eval_pick.py: only lift, elbow and wrist_flex are tight at the end of every recorded episode
# (pan and roll vary), and the arm must leave rest first, because the training rest gripper (17.8) is in the holding band.
SETTLE_JOINTS = [1, 2, 3]
LEFT_REST_DEG = 25.0


class Settled:
    def __init__(self, pose: np.ndarray, gripper_range: tuple[float, float], joint_tol_deg: float, hold_s: float,
                 min_runtime_s: float):
        self.pose, self.gripper_range, self.tol, self.hold_s, self.min_runtime_s = pose, gripper_range, joint_tol_deg, hold_s, min_runtime_s
        self.started, self.since, self.left = None, None, False

    def __call__(self, obs: dict, now: float) -> bool:
        self.started = now if self.started is None else self.started
        values = pose_of(obs)
        off = np.abs(values[SETTLE_JOINTS] - self.pose[SETTLE_JOINTS]).max()
        self.left = self.left or off > LEFT_REST_DEG
        low, high = self.gripper_range
        there = self.left and off <= self.tol and low <= values[5] <= high
        self.since = (now if self.since is None else self.since) if there else None
        return now - self.started >= self.min_runtime_s and self.since is not None and now - self.since >= self.hold_s


# GIVE without camera calibration or the #13 loop: hold the present pose over a taped spot, open once two
# palms have been seen for `dwell_s`. Without palms for `lost_s` it gives up still holding.
class PresentAndDrop:
    name = "give_present"

    def __init__(self, palms_seen, gripper_open: float, dwell_s: float = 1.0, lost_s: float = 5.0,
                 gripper_step: float = 4.0):
        self.palms_seen, self.gripper_open = palms_seen, gripper_open
        self.dwell_s, self.lost_s, self.gripper_step = dwell_s, lost_s, gripper_step

    def start(self, obs, last_sent, now):
        self.commanded = pose_of(last_sent)
        self.seen_since, self.last_seen, self.opening, self.lost = None, now, False, False

    def step(self, obs, now):
        if self.palms_seen():
            self.last_seen = now
            self.seen_since = now if self.seen_since is None else self.seen_since
        else:
            self.seen_since = None
        self.opening |= self.seen_since is not None and now - self.seen_since >= self.dwell_s
        self.lost = not self.opening and now - self.last_seen >= self.lost_s
        if self.opening:
            self.commanded[5:] = step_towards(self.commanded[5:], np.array([self.gripper_open]), self.gripper_step)
        return action_of(self.commanded)

    def done(self, obs, now):
        return self.lost or (self.opening and self.commanded[5] == self.gripper_open)

    def result(self):
        return "hand_lost" if self.lost else "dropped"


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


# The values of the command Balazs ran on the arm with Tian's branch (docs/hand-tracking-test.md section 2b on
# TianK003/hand-tracking-test; board 7064755928, "works fine"): hover 28 cm, every target at least 20 cm up, half speed.
# celebrate: Tian's 3 close-open cycles after the drop (3bb344d, dry run only so far).
@dataclass
class GiveSettings:
    hover_height: float = 0.28
    min_height: float = 0.20
    max_step_deg: float = 1.0
    max_step_m: float = 0.004
    celebrate: int = 3


# GIVE runs the #13 hand-following loop in this process on the supervisor's robot connection, so its
# safety rules (gripper hold every tick, workspace clamp, guarded steps) apply unchanged. Why not a
# subprocess: decisions/supervisor.md, "Robot ownership".
class GiveSkill:
    name = "give"

    def __init__(self, arm, handover_dir: Path, receiver, abort: threading.Event, timeout_s: float,
                 gripper_open: float, frame_writer=None, fps: int = 30, settings: GiveSettings | None = None):
        self.arm, self.receiver, self.abort, self.timeout_s = arm, receiver, abort, timeout_s
        self.gripper_open, self.frame_writer, self.fps = gripper_open, frame_writer, fps
        self.settings = settings or GiveSettings()
        self.loop = import_from(handover_dir, "handover_loop")
        self.following = import_from(handover_dir, "following")
        if self.settings.celebrate and not hasattr(self.loop, "celebration_gripper"):
            raise RuntimeError(f"{handover_dir}/handover_loop.py has no --celebrate (needs #13 at 47d437c or later)")
        self._result = "timeout"
        self._done = False

    def start(self, obs, last_sent, now):
        self.gripper_hold = float(last_sent["gripper.pos"])
        self.home = pose_of(last_sent)[:5]
        self._result, self._done, self._done_at = "timeout", False, None

    # The loop's HOME is the pose GIVE starts from: the present pose, not the rest pose (give_sequence).
    def _on_tick(self, t, decision, commanded, gripper, tip, goal_tip):
        Mode = self.following.Mode
        STATUS.update(give_mode=decision.mode.value, give_s=round(t, 1), tip=np.round(tip, 3).tolist())
        if self.abort.is_set():
            raise _GiveStopped("aborted")
        if decision.mode == Mode.DONE:
            self._done_at = t if self._done_at is None else self._done_at
            # Tian's celebration (close-open cycles, arm still) runs inside the loop; GIVE ends once it is over and the
            # gripper is open. The fold to rest is the supervisor's REST, so the loop gets no rest pose of its own.
            celebrating = t - self._done_at < 2 * self.settings.celebrate * getattr(self.loop, "CELEBRATE_PHASE_S", 0.0)
            if not celebrating and abs(gripper - self.gripper_open) < 0.5:
                raise _GiveStopped("dropped")
        # A lost hand does not end GIVE: the loop goes back to the present pose and follows the palm again when it
        # reappears, as on the arm in T2. GIVE ends on the drop or its time budget. See decisions/supervisor.md.

    def step(self, obs, now):
        g = self.settings
        follower = self.following.Follower(self.following.FollowSettings(hover_height=g.hover_height))
        limits = self.loop.Limits(max_step_deg=g.max_step_deg, max_step_m=g.max_step_m, min_height=g.min_height)
        extra = {"rest": None, "celebrate_cycles": g.celebrate} if g.celebrate else {}
        try:
            self.loop.run(_HandoverArm(self.arm), self.receiver, follower, self.home, self.gripper_hold,
                          self.gripper_open, limits, self.fps, self.timeout_s, self._on_tick,
                          self.frame_writer, **extra)
        except _GiveStopped as stop:
            self._result = stop.result
        finally:
            STATUS.update(give_mode=None)
        self._done = True

    def done(self, obs, now):
        return self._done

    def result(self):
        return self._result


# The #13 loop moves the tip in clamped Cartesian steps (0.12-0.38 m from the pan axis, 0.08-0.30 m high), so
# from the folded rest pose (tip 3.4 cm behind the axis) it never gets going: tried in the sim, it held for the
# whole timeout. GIVE therefore starts from a present pose inside that workspace; board 7061460387 gives this
# one as tip (0.15, 0, 0.20). With Tian's fitted offsets (#13 at 47d437c) the model puts it at (0.151, -0.010, 0.243),
# above the loop's 0.20 m floor, and the joint move from Balazs's folded rest pose to it stays at 0.243-0.374 m.
PRESENT_POSE = (0.0, -55.3, 5.6, 91.8, 0.0)


def holding_move(name: str, arm, joints) -> MoveTo:
    return MoveTo(name, np.concatenate([joints, [arm.last_sent["gripper.pos"]]]))


def give_sequence(arm, present_pose=PRESENT_POSE, **give_kwargs) -> Sequence:
    return Sequence("give", [holding_move("to_present", arm, present_pose), GiveSkill(arm, **give_kwargs)])
