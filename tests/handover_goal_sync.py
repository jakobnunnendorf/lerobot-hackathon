"""connect_robot writes Goal_Position = Present_Position before LeRobot switches the torque on. No hardware: the bus
and the follower are fakes that model LeRobot 0.7.0's connect() (bus.connect, then configure() turns the torque off
and on again without writing a goal)."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handover"))
import handover_loop  # noqa: E402

PRESENT = {"shoulder_pan": 2100, "shoulder_lift": 900, "elbow_flex": 3000, "wrist_flex": 2600, "wrist_roll": 2048, "gripper": 1500}
STALE_GOAL = {motor: 2048 for motor in PRESENT}


class FakeBus:
    def __init__(self, log):
        self.log = log
        self.goal = dict(STALE_GOAL)
        self.torque = False
        self.connected = False
        self.goal_at_torque_on = None

    def connect(self, handshake=True):
        assert not self.connected, "port already open"
        self.connected = True
        self.log.append("connect")

    def disconnect(self, disable_torque=True):
        assert self.connected
        if disable_torque:
            self.torque = False
        self.connected = False
        self.log.append(f"disconnect(disable_torque={disable_torque})")

    def sync_read(self, data_name, motors=None, *, normalize=True):
        assert self.connected and data_name == "Present_Position" and normalize is False
        self.log.append("read Present_Position")
        return dict(PRESENT)

    def sync_write(self, data_name, values, *, normalize=True):
        assert self.connected and data_name == "Goal_Position" and normalize is False
        self.goal.update(values)
        self.log.append("write Goal_Position")

    def disable_torque(self):
        self.torque = False
        self.log.append("torque off")

    def enable_torque(self):
        self.torque = True
        self.goal_at_torque_on = dict(self.goal)
        self.log.append("torque on")


class FakeFollower:
    instances = []

    def __init__(self, config):
        self.log = []
        self.bus = FakeBus(self.log)
        self.calibration_fpath = Path("unused.json")
        FakeFollower.instances.append(self)

    def connect(self):
        # LeRobot 0.7.0 SOFollower.connect(): bus.connect(), cameras, configure() under torque_disabled().
        self.bus.connect()
        self.bus.disable_torque()
        self.bus.enable_torque()


def install_fake_lerobot():
    names = ["lerobot", "lerobot.cameras", "lerobot.cameras.opencv", "lerobot.robots", "lerobot.robots.so_follower"]
    saved = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["lerobot.cameras.opencv"].OpenCVCameraConfig = lambda **kwargs: kwargs
    sys.modules["lerobot.robots.so_follower"].SO101Follower = FakeFollower
    sys.modules["lerobot.robots.so_follower"].SO101FollowerConfig = lambda **kwargs: kwargs
    return saved


def main():
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {got}" + ("" if ok else f" (want {want})"), flush=True)
        if not ok:
            failures.append(name)

    log = []
    bus = FakeBus(log)
    handover_loop.sync_goal_to_present(bus)
    check("helper: goal is the present position", bus.goal, PRESENT)
    check("helper: torque untouched, port closed", (bus.torque, bus.connected), (False, False))
    check("helper: order", log, ["connect", "read Present_Position", "write Goal_Position", "disconnect(disable_torque=False)"])

    saved = install_fake_lerobot()
    limits = handover_loop.kinematics.joint_limits_from_calibration
    handover_loop.kinematics.joint_limits_from_calibration = lambda path: {}
    try:
        robot = handover_loop.connect_robot("/dev/fake", "mi_follower", "0")
    finally:
        handover_loop.kinematics.joint_limits_from_calibration = limits
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    check("connect_robot: goal at torque on is the present position, not the stale goal", robot.bus.goal_at_torque_on, PRESENT)
    check("connect_robot: goal written before torque on", robot.log.index("write Goal_Position") < robot.log.index("torque on"), True)
    check(
        "connect_robot: sync disconnect keeps torque state, then LeRobot connects",
        robot.log[:5],
        ["connect", "read Present_Position", "write Goal_Position", "disconnect(disable_torque=False)", "connect"],
    )

    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
