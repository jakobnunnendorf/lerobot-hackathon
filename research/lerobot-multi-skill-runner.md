# LeRobot 0.7.0: running several ACT skills one after another on one SO-101

Researched 2026-11-20 by reading the `huggingface/lerobot` source at tag `v0.7.0` (commit `410c095`, `pyproject.toml` says `version = "0.7.0"`). Nothing was installed or executed; see "Unverified" at the end. The docs quoted here are the `docs/source/*.mdx` files at the same tag, which are the source of `https://huggingface.co/docs/lerobot/v0.7.0/en/...`; the website itself was not opened.

All paths below are relative to the repo root at the tag. To open one: `https://github.com/huggingface/lerobot/blob/v0.7.0/<path>#L<line>`.

## TL;DR

1. Do not use `lerobot-rollout` / `RolloutController` to switch skills. One rollout context owns exactly one policy and one robot connection, there is no policy-swap API, and two contexts cannot share a robot. Write a plain 30 Hz loop: one connected `SO101Follower`, all ACT checkpoints loaded up front, and per tick `get_observation -> predict_action -> send_action` with the policy of the active skill.
2. Switching skills in that loop sends nothing to the robot. Torque stays on and every servo keeps its last `Goal_Position`, so the arm keeps its pose and the gripper stays closed. What can still drop the ball is the first action of the next policy, so record GIVE and PUT DOWN demos that start with the ball already held.
3. `RolloutController.reset()` and the default shutdown both move all six joints, gripper included, back to the pose captured at connect time. Any `disconnect()` turns torque off by default, and every `connect()` briefly turns torque off too. Never reset, disconnect or reconnect while holding the ball.
4. ACT has no "done" output. A rollout segment ends only on `--duration`, a stop/reset request, or an external planner replying `done`. The state machine must decide that a skill is finished itself: a time budget plus a check on joints, gripper or camera.
5. Replay is about ten lines on an already open robot (load the episode's `action` column, `robot.send_action` at the dataset fps). The `lerobot-replay` CLI opens and closes its own connection, so use the loop, not the CLI, for a scripted stir or handshake.
6. ACT ignores the task text. `set_task`, `/subtask`, autosteer and the planner only matter for language-conditioned policies (SmolVLA, pi0, multi_task_dit and similar), none of which train on an 8 GB laptop GPU.

---

## 1. Several ACT checkpoints, one process, one connected robot

### Can a RolloutContext or inference engine swap its policy? No.

- `build_rollout_context` loads one policy and builds one engine from it (`src/lerobot/rollout/context.py:520`, `:652-666`). The result is stored in `PolicyContext(policy, preprocessor, postprocessor, inference)` (`context.py:226-238`).
- `SyncInferenceEngine.__init__` stores `self._policy`, `self._preprocessor`, `self._postprocessor` as private attributes (`src/lerobot/inference/sync.py:51-70`). The `InferenceEngine` base class has `start/stop/reset/get_action/pause/resume/set_task` and no method to replace the policy (`src/lerobot/inference/base.py:130-889`).
- The strategy caches the engine once in `setup()`: `self._engine = ctx.policy.inference` (`src/lerobot/rollout/strategies/core.py:105`). The controller also binds to it at construction: `self._initial_task = ctx.policy.inference.task` and `ctx.policy.inference.set_answer_observer(...)` (`src/lerobot/rollout/controller.py:173`, `:191`).
- `PolicyContext` is a plain dataclass, so reassigning `ctx.policy.inference` and the private `strategy._engine` between segments is technically possible, but it is not a supported path and nothing in the source does it.

### Can several controllers share one robot? No.

- `build_rollout_context` always builds its own robot: `robot = make_robot_from_config(cfg.robot)` (`context.py:348`) and connects it: `robot.connect()` (`context.py:381`). There is no parameter to pass in an existing robot or wrapper (signature at `context.py:478-484`).
- A second context would open the same serial port and the same cameras again.
- `RolloutController` is one-shot: "once `serve()` returns the controller is terminally stopped ... and a second `serve()` call raises" (`controller.py:140-141`, `:364-368`).
- Every strategy's `teardown()` disconnects the robot (`core.py:283-335`), so hand-assembled contexts around one shared robot would fight over it.

### Tearing down and rebuilding a context per skill also fails

`teardown` calls `robot.disconnect()` and the next `connect()` runs `configure()`, which wraps all motor configuration in `with self.bus.torque_disabled():` (`src/lerobot/robots/so_follower/so_follower.py:145-157`). Torque goes off for all six motors during that block, so a held ball is dropped even with `disable_torque_on_disconnect=False`. It also reloads the policy each time.

### The supported simple way: a plain loop with the public API

The repo ships this loop in three examples. All three say in a comment that `lerobot-rollout` is the production path and the script is "a self-contained example for educational purposes":

- `examples/tutorial/act/act_using_example.py:14-58` (shortest: `ACTPolicy.from_pretrained`, `make_pre_post_processors`, `build_inference_frame`, `select_action`, `make_robot_action`, `robot.send_action`; no pacing).
- `examples/lekiwi/evaluate.py:42-174` and `examples/phone_to_so100/evaluate.py:57-233` (30 fps loop with `predict_action` and `precise_sleep`).

The building blocks, with signatures:

| Step | API | Source |
| --- | --- | --- |
| Load policy from a local checkpoint dir or Hub repo id | `ACTPolicy.from_pretrained(pretrained_name_or_path, *, config=None, ...)`; returns the policy on `config.device`, in `eval()` mode | `src/lerobot/policies/pretrained.py:159-220` |
| Load the saved pre/post-processors (normalization stats live in the checkpoint) | `make_pre_post_processors(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs)`; with `pretrained_path` it loads `policy_preprocessor.json` / `policy_postprocessor.json` | `src/lerobot/policies/factory.py:158-224`, `src/lerobot/processor/factory.py:205-235` |
| ACT pipeline contents | pre: Rename -> Batch -> Device -> Normalize; post: Unnormalize -> Device(cpu) | `src/lerobot/processor/factory.py:167-184`, `src/lerobot/policies/act/processor_act.py:29-50` |
| Dataset-style feature dict from the robot | `hw_to_dataset_features(hw_features, prefix, use_video=True)` | `src/lerobot/utils/feature_utils.py:48` |
| Raw observation -> numpy frame | `build_dataset_frame(ds_features, values, prefix)` | `src/lerobot/utils/feature_utils.py:111-137` |
| Frame -> action tensor (tensor conversion, preprocessor, `select_action`, postprocessor) | `predict_action(observation, policy, device, preprocessor, postprocessor, use_amp, task=None, robot_type=None)` | `src/lerobot/common/control_utils.py:48-96` |
| Action tensor -> `{"shoulder_pan.pos": ..., "gripper.pos": ...}` | `make_robot_action(action_tensor, ds_features)` | `src/lerobot/policies/utils.py:192-218` |
| Send | `robot.send_action(action) -> action actually sent` | `src/lerobot/robots/so_follower/so_follower.py:187-213` |
| Pace the loop | `precise_sleep(seconds)` | `src/lerobot/utils/robot_utils.py:19` |
| Clear a policy's queued chunk when a skill starts | `policy.reset()`; for ACT this recreates the action queue | `src/lerobot/policies/act/modeling_act.py:96-101` (`reset`), `:104-127` (`select_action`) |

A trained checkpoint lives at `outputs/train/<job>/checkpoints/<step or last>/pretrained_model` (`src/lerobot/utils/constants.py:67-69`; the docs use `outputs/pretrain/checkpoints/last/pretrained_model` as a `--policy.path`, `docs/source/inference.mdx:128`).

### Minimal sketch - UNTESTED

Written from the source only. It has never been run.

```python
# UNTESTED SKETCH - derived from reading lerobot v0.7.0, never executed.
import threading
import time

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.policies import make_pre_post_processors, make_robot_action
from lerobot.policies.act import ACTPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
CHECKPOINTS = {  # local pretrained_model directory or Hub repo id
    "pick": "outputs/train/act_pick/checkpoints/last/pretrained_model",
    "give": "outputs/train/act_give/checkpoints/last/pretrained_model",
    "put_down": "outputs/train/act_put_down/checkpoints/last/pretrained_model",
}


class Skill:
    def __init__(self, path: str):
        self.policy = ACTPolicy.from_pretrained(path)  # eval mode, on policy.config.device
        # predict_action reads device.type, so it needs a torch.device, not the str in the config
        self.device = torch.device(self.policy.config.device)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": self.device.type}},
        )

    def reset(self) -> None:  # call when the skill becomes active
        self.policy.reset()
        self.pre.reset()
        self.post.reset()


# Camera names, resolution, robot id and use_degrees must match the recording setup.
robot = SO101Follower(
    SO101FollowerConfig(
        port="COM5",
        id="follower",
        cameras={
            "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
            "wrist": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
        },
    )
)
features = {
    **hw_to_dataset_features(robot.action_features, ACTION),
    **hw_to_dataset_features(robot.observation_features, OBS_STR),
}
skills = {name: Skill(path) for name, path in CHECKPOINTS.items()}  # load before connecting

lock = threading.Lock()
shared = {"obs": None, "requested": None}  # the CV thread reads "obs" and writes "requested"
stop = threading.Event()

robot.connect()
try:
    active, started = None, 0.0
    while not stop.is_set():
        t0 = time.perf_counter()
        obs = robot.get_observation()  # only this thread talks to the motor bus
        with lock:
            shared["obs"] = obs
            requested, shared["requested"] = shared["requested"], None

        if requested is not None and requested != active:
            # The switch itself sends nothing: the servos keep their last Goal_Position.
            active, started = requested, t0
            if active in skills:
                skills[active].reset()

        if active in skills:
            s = skills[active]
            frame = build_dataset_frame(features, obs, prefix=OBS_STR)
            action = predict_action(
                observation=frame,
                policy=s.policy,
                device=s.device,
                preprocessor=s.pre,
                postprocessor=s.post,
                use_amp=s.policy.config.use_amp,
                task="",  # ACT ignores it
                robot_type=robot.robot_type,
            )
            sent = robot.send_action(make_robot_action(action, features))
            if skill_done(active, obs, sent, t0 - started):  # your own rule, see section 3
                active = None
        # active is None: send nothing, the arm holds its last commanded pose

        precise_sleep(max(1 / FPS - (time.perf_counter() - t0), 0.0))
finally:
    robot.disconnect()  # default config: torque off on all six motors
```

Notes on the sketch:

- `PreTrainedConfig.device` is a `str` (`src/lerobot/configs/policies.py:63`), but `predict_action` evaluates `device.type` (`control_utils.py:84`). `examples/lekiwi/evaluate.py:118-121` passes `policy.config.device` and `policy.config.device.type`, which by reading would raise `AttributeError` on a `str`. The sketch wraps it in `torch.device(...)`.
- The examples pass `policy_cfg=policy` (the model, not its config) to `make_pre_post_processors`. The sketch passes `policy.config`, which is what `build_rollout_context` does (`context.py:625-634`).
- To change `n_action_steps` at inference without retraining: `PreTrainedConfig.from_pretrained(path, cli_overrides=["--n_action_steps=30"])` and pass it as `config=` to `ACTPolicy.from_pretrained` (`src/lerobot/configs/policies.py:238`), or set `policy.config.n_action_steps` and call `policy.reset()`, which rebuilds the queue with the new `maxlen`.
- Alternative with the same effect: build one `SyncInferenceEngine` per skill and call `lerobot.inference.dispatch_tick` each tick. Its docstring says "a control loop outside the rollout package calls `dispatch_tick` directly" (`src/lerobot/inference/dispatch.py:24-26`, `:127-148`; engine constructor at `sync.py:51-61`). This adds interpolation and latency stats, which the demo does not need.

---

## 2. Reset, hold and torque: what happens to a held object

### `reset()` moves the gripper too

- At connect time the context stores every observation key ending in `.pos`: `initial_position = {key: value for key, value in initial_obs.items() if key.endswith(".pos")}` (`context.py:383-384`). On the SO follower that is all six motors including `gripper.pos` (`so_follower.py:71-78`, `:84-85`).
- `RolloutController.reset()` -> `_reset_robot()` -> `strategy.return_to_initial_position(hw)` (`controller.py:246-267`, `:463-474`).
- `return_to_initial_position(hw, duration_s=3.0, fps=50)` reads the current positions and linearly interpolates every key in that dict to the stored value over 3 s, calling `robot.send_action(interp)` 150 times (`core.py:337-371`).
- Result: the arm goes to its connect-time pose and the gripper goes to its connect-time opening. If the robot was connected with the gripper open, a held ball is released. `reset()` also restores the launch task and stops autosteer.
- The same move runs at shutdown by default: `return_to_initial_position: bool = True` in `RolloutConfig` (`src/lerobot/rollout/configs.py:346-350`), used by `_teardown_hardware` (`core.py:314-320`).

### `hold_control_state` and the segment-end hold

```python
def hold_control_state(self, hw: HardwareContext) -> None:
    """Apply a segment-end hold on the control thread after background dispatch is revoked."""
    if self._interpolator is not None:
        self._interpolator.reset()
    self._cached_obs_processed = None
    if hw.robot_wrapper.supports_hold and hw.robot_wrapper.hardware_failure is None:
        hw.robot_wrapper.hold()
        self._require_engine().acknowledge_hold()
```

(`core.py:177-184`; called from the controller's `_run_segment` `finally` after `engine.pause()`, `controller.py:432-441`.)

- It clears the interpolation buffer and the cached observation. It sends a command only if `supports_hold` is true.
- `supports_hold` is true only after `configure_hold()` was called on the wrapper (`src/lerobot/rollout/robot_wrapper.py:108-117`). That call happens in two places: the RTC engine constructor (`src/lerobot/inference/rtc.py:183-186`) and remote rollouts (`context.py:367-368`, `require_hold=True` at `:810-812`). The default local `sync` backend never calls it.
- So with `--inference.type=sync` (the ACT case) the segment-end hold sends nothing. The servos keep the last `Goal_Position` written by the last policy action, torque on. That is the hold.
- With RTC or remote, `hold()` re-sends the last applied action for every actuator, or the last measured pose if nothing was sent yet (`robot_wrapper.py:129-149`). `Robot.supports_position_hold` is documented as "retaining the last applied position targets (including gripper)" (`src/lerobot/robots/robot.py:50-53`); `SOFollower` sets it to `True` (`so_follower.py:62`).
- The docs warn: "this software hold does not guarantee retained grip or physical immobility" (`docs/source/inference.mdx:162`).

### Torque on exit and disconnect

- `SOFollowerConfig.disable_torque_on_disconnect: bool = True` (`src/lerobot/robots/so_follower/config_so_follower.py:31`).
- `SOFollower.disconnect()` calls `self.bus.disconnect(self.config.disable_torque_on_disconnect)` (`so_follower.py:215-221`), and the bus does `if disable_torque: self.disable_torque(num_retry=5)` before closing the port (`src/lerobot/motors/motors_bus.py:440-461`). With the default the arm goes limp and the gripper lets go.
- `Robot.__del__` also calls `disconnect()` if the object is garbage-collected while connected (`robot.py:88-97`).
- Rollout teardown logs this explicitly: "a prior hold command does not guarantee pose retention after disconnect" (`core.py:322-330`).
- With `disable_torque_on_disconnect=False` (`--robot.disable_torque_on_disconnect=false`) the port is closed and no torque-off command is sent.
- If the process is killed without running `disconnect()`, the source sends nothing to the motors. Whether the servos then keep holding is servo behaviour, not something the source states (unverified).
- Every `connect()` runs `configure()` inside `with self.bus.torque_disabled():`, which disables torque on all motors and re-enables it on exit (`so_follower.py:145-157`, `motors_bus.py:684-705`). A reconnect therefore drops the arm for a moment regardless of the disconnect setting.
- `connect()` can also block on `input()` if the calibration stored in the motors differs from the calibration file (`so_follower.py:117-122`, `robot.py:186-200`).

---

## 3. How a run segment ends, and how a state machine learns a skill finished

### What ends `BaseStrategy.run()`

Only two conditions (`src/lerobot/rollout/strategies/base.py:63-68`):

```python
while not ctx.runtime.shutdown_event.is_set():
    ...
    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
        logger.info("Duration limit reached (%.0fs)", cfg.duration)
        break
```

There is no per-tick user callback and no success or termination signal from the policy. `duration` "bounds each /start segment, not the whole session" in interactive mode (`configs.py:307-309`).

### Hooks the controller offers

- `RolloutController(strategy, ctx, on_event: Callable[[RolloutEvent, QueryAnswer | None], None] | None = None)` (`controller.py:146-151`).
- Events (`controller.py:89-127`): `SEGMENT_STARTED`, `SEGMENT_ENDED` ("The segment returned on its own (e.g. `--duration` elapsed); the robot is holding"), `RESET_STARTED`, `RESET_DONE`, `RESET_SKIPPED`, `RESET_FAILED`, `QUERY_ANSWERED`, `ENGINE_FAILED`, `STRATEGY_FAILED`, `STOPPED`.
- "All events fire on the thread running `serve()`; callbacks must be quick and must not call back into the controller's blocking methods" (`controller.py:92-93`).
- `controller.running` is a pollable property (`controller.py:207-210`).

### Ways to end a segment early from your own code

| Call | Effect | Problem for this demo |
| --- | --- | --- |
| `controller.reset()` | Ends the segment, then homes all joints including the gripper | Drops the ball (section 2) |
| `controller.stop()` | Ends `serve()` for good; the controller is one-shot | Cannot start another segment |
| `.set()` on the `LinkedEvent` you passed to `build_rollout_context` | Ends the segment with no homing; `SEGMENT_ENDED` is emitted because no stop or reset was requested (`controller.py:458-461`). The controller clears the flag before the next segment (`controller.py:415`) | Works by reading the code. It is the same internal mechanism the controller uses for a planner `done` (`controller.py:483-484`), but it is not documented as a public way to pause |
| External planner replies `done` | `answer.completed` -> the controller sets the segment-stop flag; "the robot holds its pose until `reset` or `start`" (`controller.py:476-484`, `docs/source/inference.mdx:472`) | Needs autosteer and a VLM, or your own callable assigned to `engine.external_text` (`src/lerobot/inference/base.py:195`) |
| Custom strategy | Subclass `RolloutStrategy`, register a config with `supports_interactive = True`, and check your own event inside `run()` (`docs/source/inference.mdx:212-272`) | Still one policy per context |

### How the state machine should know a skill is done

ACT gives no signal, so the runner must decide. Options that need nothing from LeRobot beyond the observation dict:

1. A time budget per skill (the equivalent of `--duration`).
2. A joint-space check: record every demo of a skill so it ends in the same rest pose, then declare done when all `*.pos` values are within a tolerance of that pose for N consecutive ticks.
3. A gripper check (section 6): PICK is done when the gripper is commanded closed but stays open by more than a threshold and the arm has lifted; GIVE and PUT DOWN are done when the gripper has opened again.
4. The computer-vision thread (ball no longer in the bowl, hand present, and so on).

Use 1 as the safety net and one of 2-4 as the real trigger. In the plain loop of section 1 this is the `skill_done(...)` call.

---

## 4. The language-conditioned alternative

### `set_task`

- `RolloutController.set_task(task) -> bool`: "Change the instruction the policy follows, effective from the next inference" (`controller.py:281-297`). It forwards to `InferenceEngine.set_task` (`src/lerobot/inference/base.py:218-230`).
- The sync engine reads the task each tick. On a change it drops precomputed actions: `self._policy.drop_queued_actions()` and `self._postprocessor.reset()`, then puts the string into the batch as `observation["task"]` (`sync.py:110-123`, `src/lerobot/policies/utils.py:152`).
- In the interactive CLI this is `/subtask <text>` (`src/lerobot/rollout/interactive.py:279-293`). The docs: "Only affects policies that condition on language (SmolVLA, pi0/pi0.5, and similar)" (`docs/source/inference.mdx:404`).

### Does ACT use the task text? No.

- The ACT model reads only `observation.state`, `observation.environment_state`, the image list, and during training `action` / `action_is_pad` (`src/lerobot/policies/act/modeling_act.py:135-148`, `:413-483`). No file under `src/lerobot/policies/act/` mentions a task or tokenizer.
- The ACT preprocessor has no tokenizer step (section 1 table).
- Docs: "`--task` can be skipped for ACT" (`docs/source/act.mdx`, section "Evaluating ACT").
- For ACT the only effect of `set_task` is the side effect of dropping the queued chunk, which forces a fresh prediction on the next tick.

### Which policies use it

Processors with a tokenizer step exist for `smolvla`, `pi0`, `pi05`, `pi0_fast`, `xvla`, `multi_task_dit`, `wall_x`, `wall_oss_05`, `molmoact2`, `lingbot_vla_v2`, `longwam`, `fineart_vla`, `dm05`, `cosmos3` (grep for `TokenizerProcessorStep` in `src/lerobot/policies/*/processor_*.py`); `groot`, `eo1` and `evo1` handle the task string in their own processors or models. `act`, `diffusion`, `vqbet` and `tdmpc` do not reference it. Per the hardware guide these need about 8-14 GB (`multi_task_dit`) or 10-16 GB (`smolvla`) to train, against about 2-6 GB for ACT (`docs/source/hardware_guide.mdx:11-18`).

### Autosteer, `/vqa` and the planner

- `/autosteer <goal>` / `controller.autosteer(goal)`: every `autosteer_interval_s` seconds (default 10, `configs.py:317-320`) the engine asks for the next subtask and applies it through `set_task` (`controller.py:330-346`, `src/lerobot/inference/base.py:326-358`, `:484-512`).
- The answer comes from the policy's own text head, or from an external planner. Only `eo1`, `fineart_vla`, `g05`, `wall_oss_05` and `wall_x` implement `supports_text_generation` (grep in `src/lerobot/policies`). ACT inherits `return False` (`src/lerobot/policies/pretrained.py:359-361`), so with ACT `ask` and `autosteer` return `AskResult.UNSUPPORTED` unless a planner is attached (`controller.py:312-314`, `:338-339`).
- `src/lerobot/rollout/planner.py`: `VlmPlanner` sends the camera images plus the last `--planner.history` (default 4) observation/command pairs to an OpenAI-compatible VLM and gets one JSON reply. `--planner.instructions='[...]'` restricts replies to a closed list; without it the list is read from the checkpoint's training dataset `meta/tasks.parquet` (`planner.py:79-97`, `:283-302`; `context.py:713-761`). A reply of `done` ends the segment. It is enabled with `--planner.model_id=...` and needs the `annotations` extra and a running VLM server (`docs/source/inference.mdx:466-498`).
- With ACT a planner could pick instruction strings, but ACT would ignore them. The planner is only useful here as an example of the `external_text` hook.

### `sentry` and `highlight`

- `sentry` (`src/lerobot/rollout/strategies/sentry.py`, `configs.py:99-122`): autonomous rollout that records everything, rotates episodes by estimated video file size, and pushes to the Hub every `upload_every_n_episodes` (default 5). It supports `--interactive` and the controller. Frames are labelled with `engine.dispatched_task` (`sentry.py:152-156`). A new rollout dataset name must start with `rollout_` (`context.py:446-451`).
- `highlight` (`src/lerobot/rollout/strategies/highlight.py`, `configs.py:125-144`): autonomous rollout with a ring buffer of the last `ring_buffer_seconds` (default 10). Pressing `s` flushes the buffer and records live until `s` is pressed again; `h` pushes to the Hub. It binds the keyboard, so it does not support `--interactive` or the controller (`supports_interactive` stays `False`).
- Use for this project: `sentry` to log evaluation runs of a single skill, `highlight` to capture failures for later inspection. Neither helps with skill switching.

---

## 5. Recording several skills, and replay

### Can one dataset hold several task strings? Yes.

- The task is stored per frame. `add_frame` pops `"task"` from each frame (`src/lerobot/datasets/dataset_writer.py:228`); `save_episode` collects the unique strings, registers new ones, and writes a per-frame `task_index` (`dataset_writer.py:283-294`).
- The strings live in `meta/tasks.parquet` (`src/lerobot/datasets/utils.py:174`); `save_episode_tasks` appends new tasks with the next free index (`src/lerobot/datasets/dataset_metadata.py:543-570`). Episode metadata also lists the episode's tasks.

### How `--dataset.single_task` is stored

- `DatasetRecordConfig.single_task: str = ""` (`src/lerobot/configs/dataset.py:29`).
- `lerobot-record` writes it into every frame of the session: `frame = {**observation_frame, **action_frame, "task": single_task}` (`src/lerobot/scripts/lerobot_record.py:372`).
- So one `lerobot-record` session gives one task string. A second session with `--resume=true` and a different `--dataset.single_task` would add a second task to the same dataset (by reading; not run).

### Subtask annotation

- `lerobot-record` has no way to mark subtasks during recording (no `subtask` anywhere in `lerobot_record.py`).
- `lerobot-annotate` adds them afterwards with a VLM: it writes `subtask`, `plan`, `memory` and `task_aug` rows into a `language_persistent` column and events into `language_events` (`docs/source/annotation_pipeline.mdx:1-60`, `docs/source/language_and_recipes.mdx:28-58`).
- These columns feed recipe-based language policies. ACT does not read them.

### Cleanest way to record one dataset per skill

Run `lerobot-record` once per skill with its own repo id and task string:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<port> --robot.id=<same id every time> \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=<port> --teleop.id=<id> \
  --dataset.repo_id=${HF_USER}/balls_pick \
  --dataset.single_task="Pick the blue ball from the bowl" \
  --dataset.num_episodes=40 --dataset.episode_time_s=15 --dataset.reset_time_s=10 \
  --dataset.no_stamp=true --dataset.streaming_encoding=true
```

Points from the source:

- Without `--dataset.no_stamp=true` the name gets a `_YYYYMMDD_HHMMSS` suffix (`lerobot_record.py:476`, `src/lerobot/configs/dataset.py:76-88`, `docs/source/il_robots.mdx:349`).
- To add episodes later: same command with `--resume=true`, the full dataset name, `--dataset.root`, and `--dataset.num_episodes` set to the number of additional episodes (`docs/source/il_robots.mdx:384`).
- Names starting with `eval_` are rejected by `lerobot-record` (`lerobot_record.py:469-475`). A teleoperator is mandatory (`lerobot_record.py:196-202`).
- Defaults: `episode_time_s=60`, `reset_time_s=60`, `num_episodes=50`, `push_to_hub=True` (`src/lerobot/configs/dataset.py:35-43`). Right arrow or `n` ends an episode early, left arrow or `r` re-records, Esc or `q` stops (`docs/source/il_robots.mdx:398-404`).
- Keep camera names, resolution, `--robot.id` (calibration file) and `use_degrees` identical across all skills and at deployment. `use_degrees` defaults to `True` in 0.7.0 (`config_so_follower.py:41-42`).
- If everything ends up in one dataset anyway: train one ACT per skill with `--dataset.episodes=[...]` (`src/lerobot/configs/default.py:44-46`), relabel with `lerobot-edit-dataset --operation.type modify_tasks --operation.episode_tasks '{...}'`, or combine datasets with `--operation.type merge` (`docs/source/using_dataset_tools.mdx:150-218`).

### `lerobot-replay` options

`ReplayConfig` (`src/lerobot/scripts/lerobot_replay.py:85-102`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--robot.*` | required | Same robot flags as elsewhere |
| `--dataset.repo_id` | required | Dataset to read |
| `--dataset.episode` | required | Episode index |
| `--dataset.root` | `None` | Local dataset directory |
| `--dataset.fps` | `30` | Declared, but the loop paces with the dataset's own fps (`CycleTimer(dataset.fps, ...)`, `lerobot_replay.py:123`) |
| `--play_sounds` | `True` | Spoken announcement |

The CLI builds its own robot, connects, replays, and disconnects in `finally` (`lerobot_replay.py:112-152`). It cannot be pointed at an already open connection, and its disconnect turns torque off.

### Replay from your own code on an open robot: yes

The loop is public API and short. Docs example (`docs/source/il_robots.mdx:478-512`), same pattern in `lerobot_replay.py:127-149` and `examples/phone_to_so100/replay.py:69-99`:

```python
# UNTESTED - adapted from docs/source/il_robots.mdx, robot is already connected.
dataset = LeRobotDataset("<user>/<dataset>", root=<local dir or None>, episodes=[episode_idx])
actions = dataset.select_columns("action")
names = dataset.features["action"]["names"]
frames = [  # load once, before the demo
    {name: float(actions[i]["action"][j]) for j, name in enumerate(names)}
    for i in range(dataset.num_frames)
]
for action in frames:
    t0 = time.perf_counter()
    robot.send_action(action)
    precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))
```

- `LeRobotDataset(repo_id, root=None, episodes=None, ..., download_videos=True, ...)` (`src/lerobot/datasets/lerobot_dataset.py:53-78`); `select_columns` returns the underlying HF dataset restricted to those columns (`lerobot_dataset.py:607-613`).
- The recorded action includes `gripper.pos`, so a replayed stir or handshake commands the gripper exactly as recorded.
- Replay is open loop and starts from wherever the arm is. If the first recorded action is far from the current pose the arm jumps; move to the episode's first pose with a short interpolation first.
- For hand-written motions without a recording, `examples/omx/reset_environment.py` shows interpolated waypoint moves with `robot.send_action` and `precise_sleep`.

---

## 6. From a second thread: camera frames and joint positions

### Newest camera frames

- Each `OpenCVCamera` runs its own background read thread that stores `latest_frame` and `latest_timestamp` under `frame_lock` (`src/lerobot/cameras/opencv/camera_opencv.py:460-495`).
- `read_latest(max_age_ms: int = 500) -> NDArray` returns that frame without blocking. It raises `TimeoutError` if the frame is older than `max_age_ms` and `RuntimeError` before the first frame or if the read thread died (`camera_opencv.py:588-620`).
- Access: `robot.cameras["front"].read_latest()` on the plain robot (`so_follower.py:81`), or `ctx.hardware.robot_wrapper.cameras["front"]` in a rollout (`robot_wrapper.py:169-171`).
- Frames are `(height, width, 3)` arrays in RGB by default (`color_mode: ColorMode = ColorMode.RGB`, `src/lerobot/cameras/opencv/configuration_opencv.py:62`). Convert to BGR for OpenCV functions that expect it.
- `get_observation()` uses the same `read_latest()` call (`so_follower.py:175`), so the CV thread and the policy can receive the very same array object. Never draw on it or modify it in place; copy first.

### Joint positions and gripper opening

- `robot.get_observation()` returns `{"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos", <camera names>...}`. Positions come from `self.bus.sync_read("Present_Position", ...)` (`so_follower.py:162-185`).
- The Python bus class has no lock (`src/lerobot/motors/motors_bus.py` contains no `Lock`). Do not call the bus from a second thread while the control loop is using it.
- Safe pattern for the plain loop: the control thread stores each observation in a shared variable and the CV thread reads it (as in the sketch).
- In a rollout, `ctx.hardware.robot_wrapper.get_observation()` is serialized by the wrapper's lock (`robot_wrapper.py:50-60`), so it is safe from another thread, but each call does a full motor read plus camera reads while holding the lock the control loop needs.

### Units

- Body joints: degrees when `use_degrees=True` (the 0.7.0 default), otherwise -100..100 (`so_follower.py:68`).
- Gripper: always `MotorNormMode.RANGE_0_100` (`so_follower.py:77`). The value is a percentage of the calibrated range: `((bounded_val - min_) / (max_ - min_)) * 100`, with `min_`/`max_` the calibration's `range_min`/`range_max` (`motors_bus.py:1027-1029`). It is not degrees and not millimetres. Which end is "closed" depends on the calibration (unverified on hardware; by convention 0 is closed).

### Detecting "the gripper is holding something"

- Position gap: `send_action` returns the goal actually sent (`so_follower.py:213`). While holding a ball the commanded `gripper.pos` is at or near the closed end but the measured `gripper.pos` stalls at the ball's width. A gap above a threshold for several ticks means "holding"; a gap near zero with the gripper at the closed end means "closed on nothing".
- Motor current: `robot.bus.sync_read("Present_Current")` is used in `src/lerobot/rl/joint_observations_processor.py:166`. It returns raw register values (only `goal_position` and `present_position` are normalized, `motors_bus.py:197`). Call it from the control thread only.
- The gripper motor is limited at connect time: `Max_Torque_Limit` 500, `Protection_Current` 250, `Overload_Torque` 25 (`so_follower.py:154-157`).

---

## 7. ACT settings for short skills and small data

### Defaults in the source

| Setting | Default | Source |
| --- | --- | --- |
| `chunk_size` | 100 | `src/lerobot/policies/act/configuration_act.py:85` |
| `n_action_steps` | 100 (must be <= `chunk_size`) | `configuration_act.py:86`, `:144-148` |
| `n_obs_steps` | 1 (other values raise) | `configuration_act.py:84`, `:149-152` |
| `temporal_ensemble_coeff` | `None` (off). "the value used in ACT when temporal ensembling is enabled is 0.01". Requires `n_action_steps == 1` | `configuration_act.py:117-119`, `:139-143` |
| `vision_backbone` | `resnet18`, ImageNet weights | `configuration_act.py:98-99` |
| `dim_model`, `n_heads`, `dim_feedforward` | 512, 8, 3200 | `configuration_act.py:103-105` |
| `n_encoder_layers` / `n_decoder_layers` | 4 / 1 | `configuration_act.py:107-111` |
| `use_vae`, `latent_dim`, `kl_weight` | `True`, 32, 10.0 | `configuration_act.py:113-114`, `:123` |
| `dropout` | 0.1 | `configuration_act.py:122` |
| `optimizer_lr`, `optimizer_lr_backbone`, `optimizer_weight_decay` | 1e-5, 1e-5, 1e-4 | `configuration_act.py:126-128` |
| LR scheduler | none (`get_scheduler_preset` returns `None`) | `configuration_act.py:161-162` |
| Normalization | mean/std for images, state and action | `configuration_act.py:88-94` |
| Training `steps` | 100_000 | `src/lerobot/configs/train.py:160` |
| Training `batch_size` | 8 | `train.py:151` |
| `num_workers` | 4 | `train.py:150` |
| `save_freq` / `log_freq` | 20_000 / 200 | `train.py:172`, `:163` |
| Image augmentation | off: `image_transforms.enable: bool = False` | `src/lerobot/transforms/transforms.py:1452-1453` |
| `use_imagenet_stats` | `True` | `src/lerobot/configs/default.py:49` |

### What the settings do at inference

- `select_action` returns one action per call from a queue and runs the model only when the queue is empty, taking the first `n_action_steps` of the predicted chunk (`src/lerobot/policies/act/modeling_act.py:104-127`).
- At 30 fps the default of 100 means one prediction every 3.3 s, executed open loop. A 3-5 s skill is then one or two predictions, and the policy cannot react to anything in between.
- `n_action_steps` can be lowered at inference without retraining because it only slices the chunk (`--policy.n_action_steps=...` on `lerobot-rollout`, or the config override in section 1).
- With temporal ensembling the model runs every tick and the outputs are averaged with weights `exp(-coeff * i)`, oldest first (`modeling_act.py:170` onward, `ACTTemporalEnsembler`). `ACTPolicy.__init__` creates the ensembler only if the config has the coefficient set, so it must be in the config before the policy is constructed.
- ACT does not declare RTC support (`supports_rtc` is inherited as `False`, `pretrained.py:280-282`). Use the default `sync` backend.

### Image augmentation

- Enable with `--dataset.image_transforms.enable=true` (flag used in `docs/source/lerobot-dataset-v3.mdx:288`).
- Default set when enabled: brightness (0.8-1.2), contrast (0.8-1.2), saturation (0.5-1.5), hue (+-0.05), sharpness (0.5-1.5), affine (+-5 degrees, 5% translation); up to `max_num_transforms=3` per frame (`transforms.py:1472-1511`).
- `--dataset.image_transforms.backend=gpu` runs them on the policy device instead of the dataloader workers (`transforms.py:1454-1471`).

### What the 0.7.0 docs recommend

- `docs/source/act.mdx`, "Training Tips": "Start with defaults: ACT's default hyperparameters work well for most tasks"; "Expect a few hours for 100k training steps on a single GPU"; "Start with batch size 8 and adjust based on your GPU memory". The page also says ACT "often achieves high success rates with just 50 demonstrations" and has about 80M parameters.
- The ACT page gives no recommended `chunk_size`, `n_action_steps` or temporal-ensembling values.
- `docs/source/hardware_guide.mdx:22-31`: "Robotics imitation learning typically converges in 5-10 epochs over the dataset", with `steps_per_epoch = ceil(total_frames / (num_gpus x batch_size))`. ACT peak VRAM at batch 8 is about 2-6 GB (`hardware_guide.mdx:13`).
- `docs/source/il_robots.mdx:540`: the default 100,000 steps "is usually more than a small dataset needs".
- `hardware_guide.mdx:70-72` says to shorten the LR schedule when shortening training. That does not apply to ACT, which has no scheduler.
- `docs/source/il_robots.mdx:416`: "We suggest recording at least 50 episodes, with 10 episodes per location. Keep the cameras fixed and maintain consistent grasping behavior".

### Suggested starting values (judgement, not from the docs)

- Train with the default `chunk_size=100`, batch size 8, augmentation on if lighting will differ between recording and demo.
- Worked example of the docs' formula: 40 episodes x 10 s x 30 fps = 12,000 frames; at batch 8 that is 1,500 steps per epoch, so 5-10 epochs is 7,500-15,000 steps. Set `--steps` well below 100,000 (for example 20,000-40,000) with `--save_freq=5000` and compare checkpoints on the robot.
- At inference start with `n_action_steps` around 20-50 so the policy re-plans two or three times per second of motion; try `temporal_ensemble_coeff=0.01` with `n_action_steps=1` only if motion is jerky and the loop still holds 30 Hz.

---

## Recommended way to build the skill runner

1. **One process, one robot object, connected once.** Build `SO101Follower(SO101FollowerConfig(...))` with both cameras, call `connect()` at start and `disconnect()` only at the very end, after the gripper is empty and the arm is in a rest pose.
2. **Load every ACT checkpoint and its processors before connecting.** Keep them in a dict of `Skill` objects (section 1 sketch). Use the same `features` dict, built once from `robot.observation_features` and `robot.action_features`.
3. **One control thread owns the robot.** At 30 Hz it reads the observation, publishes it for the CV thread, checks for a requested skill, runs the active skill for one tick, and sleeps the remainder with `precise_sleep`.
4. **Skills are one of three kinds** behind the same "one tick" interface:
   - policy skill (PICK, GIVE, PUT DOWN, possibly MIX): `predict_action` -> `make_robot_action` -> `send_action`;
   - replay skill (SHAKE, possibly MIX): step through a preloaded list of recorded actions;
   - idle: send nothing.
5. **Switching** happens between ticks: set the new active skill and call its `reset()`. Send nothing during the switch. The arm keeps its pose and the gripper keeps its last commanded target because torque stays on.
6. **Protect the grasp across PICK -> GIVE / PUT DOWN.**
   - Record GIVE and PUT DOWN demos starting from the pose and gripper state where PICK ends, with the ball in the gripper, so the first predicted action keeps the gripper closed.
   - Optionally clamp `gripper.pos` in the outgoing action to the held value for the first N ticks of the new skill, or until the arm has moved away from the hand-over pose.
   - For a replay skill that follows PICK, overwrite `gripper.pos` in each replayed action with the current held value.
7. **The CV thread never touches the motor bus.** It reads frames from the shared observation (or `robot.cameras[name].read_latest()`), copies before drawing, and writes the next skill name into the shared variable under the lock.
8. **Skill end** = per-skill timeout plus a specific check (section 3). On timeout go to idle and let the state machine decide; do not home the arm automatically while the gripper is closed on something.
9. **Shutdown and emergency.** On normal exit: finish or abort the skill, open the gripper over the table, move to the rest pose with an interpolated move, then `disconnect()`. Decide deliberately whether to set `disable_torque_on_disconnect=False`: with the default the arm goes limp at disconnect, so it must already be resting on something.
10. **Use `lerobot-rollout --strategy.type=base --duration=...` only to evaluate each skill on its own** while training, with `--return_to_initial_position=false` if the test ends with a held object.

### Main traps

- `controller.reset()`, `/reset` and the default rollout shutdown move the gripper to its connect-time opening.
- `disconnect()` disables torque by default; `connect()` always cycles torque off and on.
- Default `n_action_steps=100` is 3.3 s of open-loop motion at 30 fps.
- `predict_action` needs a `torch.device`; the shipped examples pass the config's string.
- `read_latest()` raises `TimeoutError` when a camera stalls for more than 500 ms; catch it in the CV thread. In the control loop the same exception comes out of `get_observation()`.
- `max_relative_target` must be a `float` or a per-motor `dict`; an `int` raises `TypeError` (`src/lerobot/robots/utils.py:142-149`). When set, every `send_action` also reads the motors first, which slows the loop (`so_follower.py:204-209`), and it caps how far the gripper goal can sit beyond the measured position, which may weaken the grip (inferred from the clamp, not tested).
- Dataset names get a timestamp suffix unless `--dataset.no_stamp=true`; rollout datasets must start with `rollout_`; `lerobot-record` rejects `eval_`.
- `use_degrees` now defaults to `True`. A dataset recorded with one setting cannot be replayed or used for a policy with the other.
- `import lerobot.rollout` requires the `dataset` extra (`src/lerobot/rollout/__init__.py:23-25`); `pyproject.toml` requires Python >= 3.12.

---

## Unverified

Nothing in this file was executed. LeRobot is not installed on the research machine, no robot or camera was connected, and no training or inference was run. Every statement is from reading the v0.7.0 source and docs. In particular:

- The code sketches (skill loop, replay loop) are untested. Import paths and signatures were checked against the source, but not by running them.
- That the arm and gripper physically hold their pose while no commands are sent is inferred from "torque stays enabled and nothing is written". Servo behaviour under load, heating of the gripper motor while squeezing a ball for a long time, and its overload protection (`Overload_Torque` 25, `Protection_Current` 250) were not tested.
- What the servos do if the process is killed without `disconnect()` is not stated in the source.
- That `examples/lekiwi/evaluate.py` and `examples/phone_to_so100/evaluate.py` fail on `policy.config.device.type` is a reading of the code (`device` is typed `str`), not an observed error.
- Setting the `LinkedEvent` yourself to end a controller segment without homing, and swapping `ctx.policy.inference`, are readings of the control flow, not documented or tested usage.
- `--resume=true` with a different `--dataset.single_task` producing a two-task dataset is inferred from the record loop; not run.
- Which end of the gripper's 0-100 range is "closed" depends on calibration and was not checked on hardware. The thresholds for a "holding" detector have to be measured.
- Whether the Rust motor library behind `SerialMotorsBus` (`rustypot`) is safe to call from two threads was not checked; the recommendation to keep one thread on the bus is the conservative reading.
- Register names other than those used in the LeRobot source (`Present_Position`, `Goal_Position`, `Present_Current`, `Max_Torque_Limit`, `Protection_Current`, `Overload_Torque`) are defined in `rustypot`, which was not read.
- GPU memory with five ACT policies loaded at once, and whether inference every tick (temporal ensembling) holds 30 Hz on the 8 GB laptop, were not measured.
- The suggested `--steps`, `n_action_steps` and demo counts in section 7 are judgement, extrapolated from the docs' "5-10 epochs" rule; they are not documented recommendations for ACT.
- The docs website was not opened; the `.mdx` files at the tag were read instead, so any difference between the built site and the tag would be missed.
