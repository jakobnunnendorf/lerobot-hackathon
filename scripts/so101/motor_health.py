"""Usage: python motor_health.py              read every motor of both arms (nothing moves)
       python motor_health.py --cool 45    poll the follower gripper until it is at or below 45 C"""
import argparse
import os
import time
import warnings

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

warnings.filterwarnings("ignore")
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Feetech Status register bits
STATUS_BITS = {1: "VOLTAGE", 2: "SENSOR", 4: "OVERHEAT", 8: "CURRENT", 32: "OVERLOAD"}


def bus_for(arm, names=NAMES):
    port = os.environ.get(f"SO101_{arm.upper()}_PORT", f"/dev/so101_{arm}")
    bus = FeetechMotorsBus(port, {n: Motor(NAMES.index(n) + 1, "sts3215", MotorNormMode.DEGREES) for n in names})
    bus.connect(handshake=False)
    return bus


def describe(status):
    return ",".join(name for bit, name in STATUS_BITS.items() if status & bit) or "ok"


def report(arm):
    bus = bus_for(arm)
    print(f"== {arm}")
    for i, name in enumerate(NAMES, 1):
        if bus.ping(i, num_retry=3) is None:
            print(f"   {name:13s} NO REPLY (cable/plug, or motor without power)")
            continue
        read = lambda reg: bus.read(reg, name, normalize=False)
        status = read("Status")
        print(f"   {name:13s} {read('Present_Temperature'):3d}C {read('Present_Voltage') / 10:4.1f}V "
              f"torque={read('Torque_Enable')} status={describe(status)}")
    bus.disconnect(disable_torque=False)


def cool(limit):
    bus = bus_for("follower", ["gripper"])
    start = time.time()
    while (temp := bus.read("Present_Temperature", "gripper", normalize=False)) > limit:
        print(f"{int(time.time() - start):4d}s gripper {temp}C", flush=True)
        time.sleep(20)
    bus.disconnect(disable_torque=False)
    print(f"COOLED: gripper {temp}C", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cool", type=int)
    args = parser.parse_args()
    if args.cool:
        cool(args.cool)
    else:
        report("follower")
        report("leader")
