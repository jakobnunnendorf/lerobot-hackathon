# Supervised cable cutting (disassembly cell)

**Status: DROPPED** on 2026-11-20. The team chose `ball-picking-and-handover.md` instead. Kept for the record. The parts about a supervising camera, pulling the arm back and checking results still apply to the new idea.

## In one line

An SO-101 with a small cutter cuts the correct cable on a device that is being taken apart, while a second camera supervises it: it picks the cable, pulls the arm back when a hand comes near the tool, and checks that the cut really happened.

Story: battery and e-waste disassembly is still mostly manual, and "which cable is cut first" is a real question there. The same robot work as the placeholder, without the bomb prop and without a blade.

## What the jury sees

1. The supervisor camera looks at the device and marks the target cable on a screen.
2. The arm reaches in and cuts it with spring-loaded snips or blunt scissors. The "cables" are soft: silicone tubing, yarn or paper straws.
3. A judge puts a hand near the tool. The arm stops and pulls back to its start pose.
4. The hand leaves. The supervisor looks at the scene again, because a person was in there and may have changed something, decides again, and the arm goes back in.
5. After the cut the supervisor checks that the cable is really severed. If not, the arm tries again.

## How the two cameras are used

The judges require at least two cameras (`kb/judges-requirements.md`).

- Both cameras feed the learned policy, as in the standard LeRobot setup. One sees the whole arm, one sees the work up close.
- The second camera is also the supervisor. It does three things the policy does not: choose the target, watch the distance between a hand and the tool, verify the result.
- With two viewpoints, "the hand is near the tool" can be required in both views before the arm reacts. That gives a rough 3D check without calibrating the cameras, and one view still works when the other is blocked.

## What is learned and what is coded

- Learned (ACT, from our own demonstrations): reaching the cable and making the cut. One small skill per cable position, about 40 episodes each.
- Coded: target choice, hand and tool tracking, the pull-back, the re-check, the retry.

## Checked in the LeRobot 0.7.0 source (tag v0.7.0), not yet run

- `lerobot.rollout.RolloutController` has thread-safe `start()`, `reset()` and `stop()`. `reset()` stops movement and returns the robot to its initial position while the policy stays loaded, and `start()` runs it again. That is the pull-back and resume. It works with `--strategy.type=base`.
- There is no public "freeze where you are and continue" call. Pulling back to the start pose is what the API offers, and it is the safer behaviour next to a cutter.
- The rollout's robot wrapper exposes its cameras, and the OpenCV camera class has `read_latest()` backed by its own read thread. So a supervisor thread in the same process can read the same cameras the policy uses. A third camera is not needed.

## Not known yet

- Whether the gripper closes the snips through the chosen material 10 times out of 10. First test.
- Whether a hand detector (MediaPipe Hands is the obvious one) installs cleanly next to LeRobot on Python 3.12 and how often it fires falsely on the arm itself. A bright glove and a colour check is the fallback.
- How the policy behaves when it starts again on a cable that is half cut.
- How reliably "cable is severed" can be seen. Mount the cables under slight tension so the ends fall apart when cut.

## Stretch: route around an obstacle

Added 2026-11-20. First thing to drop if time runs short.

- The joint angles say where every part of the arm is. A small hand-written arm model turns them into the positions of elbow, wrist and tool. The camera only has to find the obstacle on the table.
- Level 1, virtual fence: if a part of the arm is about to enter a keep-out zone around the obstacle, the supervisor pulls the arm back. Same mechanism as the hand guard.
- Level 2, route selection: three or four routes are taught from the start pose to a fixed pose just in front of the cables (over, left, right). The arm model checks each route against the obstacle and the arm takes the first one that clears. Then the learned policy does the cut. A judge moves the obstacle and the arm picks another way.
- Not planned: a full motion planner, or changing the policy's commands while it runs.
- **This affects recording.** All cutting demonstrations start from the fixed pose in front of the cables, not from the rest position. Then the policy does not depend on how the arm got there, and routing can be added or dropped without retraining. Decide this before recording starts.
- The obstacle is a fat foam or cardboard tube, not steel. Failed runs will hit it, and the arm model will be off by a centimetre or more.
- LeRobot's own arm model cannot be used on Windows (`kb/kinematics-on-windows.md`), so the model is written by hand and runs on any of our laptops.

## Why it can be built in parallel

The supervisor needs a policy to drive, but any policy will do for testing the pull-back and restart, including the throwaway one from the first smoke test. So it can be built on Saturday while the real data is recorded.
