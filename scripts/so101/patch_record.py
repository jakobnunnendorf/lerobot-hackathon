"""SO-101 hackathon patches for LeRobot 0.7 recording (idempotent; keeps .orig backups).

1. SPACE acts like the RIGHT arrow (start / finish an episode).
2. SO101_WAIT_START=1: wait for SPACE before the first episode (teleop live, nothing recorded).
3. Ignore key presses that piled up while an episode was being saved.
4. SO101_BANNERS=1: big on-screen instructions at every phase + voice via espeak-ng
   (non-blocking; replaces the flaky spd-say).
Re-run after reinstalling LeRobot:  python scripts/so101/patch_record.py
"""
import os, shutil, sys
import lerobot

base = os.path.dirname(lerobot.__file__)

def patch(rel, old, new, marker):
    path = os.path.join(base, rel)
    src = open(path).read()
    if marker in src:
        print(f"{rel}: [{marker}] already patched"); return
    if src.count(old) != 1:
        sys.exit(f"{rel}: anchor for [{marker}] found {src.count(old)} times - NOT patching")
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
    open(path, "w").write(src.replace(old, new))
    print(f"{rel}: [{marker}] patched")

# 1. space = next
patch("utils/keyboard_input.py",
      'if key in ("right", "n"):',
      'if key in ("right", "n", "space"):  # SO101 patch: space = next',
      "SO101 patch: space")

# 2. wait before first episode
patch("scripts/lerobot_record.py",
"""            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:""",
"""            recorded_episodes = 0
            # SO101 patch: wait (teleop live, nothing recorded) until SPACE/right before episode 0
            if __import__("os").environ.get("SO101_WAIT_START") == "1":
                log_say("Press space to start", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                )
                events["rerecord_episode"] = False
                events["exit_early"] = False
                timer.restart()
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:""",
"SO101 patch: wait")

# 3. drop stale key presses at the start of each episode / pause, and announce saving
patch("scripts/lerobot_record.py",
"""                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
""",
"""                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
                events["exit_early"] = False  # SO101 patch: stale keys
                events["rerecord_episode"] = False
""",
"SO101 patch: stale keys")
patch("scripts/lerobot_record.py",
"""                    log_say("Reset the environment", cfg.play_sounds)
""",
"""                    log_say("Reset the environment", cfg.play_sounds)
                    events["exit_early"] = False  # SO101 patch: stale keys (pause)
""",
"SO101 patch: stale keys (pause)")
patch("scripts/lerobot_record.py",
"""                dataset.save_episode()
                recorded_episodes += 1""",
"""                log_say("Saving episode", False)  # SO101 patch: saving banner
                dataset.save_episode()
                recorded_episodes += 1""",
"SO101 patch: saving banner")

# 4. banners + reliable voice
patch("utils/utils.py",
"""def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)
""",
"""def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    logging.info(text)
    if os.environ.get("SO101_BANNERS") == "1":  # SO101 patch: banners
        _so101_banner(text)
        if play_sounds:
            import subprocess
            try:
                subprocess.Popen(["espeak-ng", "-s", "160", text],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        return

    if play_sounds:
        say(text, blocking)


_SO101_KEYS = "  KEYS:  SPACE = start / finish episode   LEFT = throw away   ESC = end session"


def _so101_banner(text: str) -> None:
    Y, R, G, C, B, X = "\\033[30;43m", "\\033[97;41m", "\\033[30;42m", "\\033[30;46m", "\\033[1m", "\\033[0m"
    if text.startswith("Press space to start"):
        col, title, lines = Y, "WAITING - NOT RECORDING YET", [
            "Put the arm in REST POSE and reset the scene.",
            "Press SPACE to START recording the first episode.",
            "(Esc = quit without recording)"]
    elif text.startswith("Recording episode"):
        col, title, lines = R, "RECORDING  " + text.split("Recording ")[-1].upper(), [
            "Do the task now (see the task line at the top).",
            "Press SPACE when finished  ->  the episode is KEPT.",
            "Press LEFT if it went wrong ->  the episode is THROWN AWAY."]
    elif text.startswith("Reset the environment"):
        col, title, lines = Y, "PAUSED - NOT RECORDING", [
            "Put the arm back in REST POSE and reset the scene.",
            "Bad last episode? Press LEFT now to throw it away.",
            "Press SPACE to START the next episode.   ESC = finish the session."]
    elif text.startswith("Saving episode"):
        col, title, lines = C, "SAVING... PLEASE WAIT", [
            "Don't press any keys. Recording starts by itself in a moment."]
    elif text.startswith("Re-record episode"):
        col, title, lines = C, "EPISODE THROWN AWAY", ["It will be recorded again."]
    elif text.startswith("Stop recording"):
        col, title, lines = C, "FINISHING - SAVING EVERYTHING", [
            "Please wait. Do NOT press Ctrl+C."]
    elif text.startswith("Exiting"):
        col, title, lines = G, "SESSION ENDED", ["Tell Claude - it will check what was saved."]
    else:
        return
    w = 72
    out = ["", col + B + " " * w + X, col + B + title.center(w) + X, col + B + " " * w + X]
    out += [B + "  " + l + X for l in lines]
    if not text.startswith(("Saving", "Stop", "Exiting", "Re-record")):
        out.append(_SO101_KEYS)
    out.append("")
    print("\\n".join(out), flush=True)
""",
"SO101 patch: banners")

import importlib  # noqa: E402
import lerobot.utils.utils as u, lerobot.scripts.lerobot_record  # noqa: E401,F401
print("import ok; os in utils:", hasattr(u, "os"))
