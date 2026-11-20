# Supervised cable cutting (disassembly cell)

**Status: CANDIDATE.** Grew out of the bomb-defusal placeholder on 2026-11-20. Not locked. The team decides on the message board (issue #1).

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

## Why it can be built in parallel

The supervisor needs a policy to drive, but any policy will do for testing the pull-back and restart, including the throwaway one from the first smoke test. So it can be built on Saturday while the real data is recorded.
