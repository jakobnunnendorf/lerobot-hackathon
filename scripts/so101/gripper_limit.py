# Usage: python scripts/so101/gripper_limit.py <follower port>   (or import cap_gripper_torque and call it on a connected bus)
# The follower gripper's torque is capped at 200 of 1000: never raise it. See kb/gripper-torque-limit.md.
# LeRobot 0.7.0's connect() writes Max_Torque_Limit 500 (EEPROM) every time; the live Torque_Limit (RAM) keeps what was
# written last and only reloads from Max_Torque_Limit at power-up. So the cap is written after every connect, and once
# before the lerobot-* CLIs, whose connect then leaves it alone (checked on the arm 2026-11-22).
import sys

GRIPPER_TORQUE_LIMIT = 200


def cap_gripper_torque(bus) -> None:
    bus.write("Torque_Limit", "gripper", GRIPPER_TORQUE_LIMIT, normalize=False)
    applied = bus.read("Torque_Limit", "gripper", normalize=False)
    if applied != GRIPPER_TORQUE_LIMIT:
        raise RuntimeError(f"gripper Torque_Limit reads {applied}, not {GRIPPER_TORQUE_LIMIT}")
    print(f"gripper torque capped at {GRIPPER_TORQUE_LIMIT}/1000", flush=True)


if __name__ == "__main__":
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port=sys.argv[1], motors={"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)})
    bus.connect(handshake=False)
    try:
        cap_gripper_torque(bus)
    finally:
        bus.disconnect(disable_torque=False)
