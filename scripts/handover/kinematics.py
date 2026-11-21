import numpy as np

# PLACEHOLDER until TianK003/hand-following lands: a planar three-link guess with rough link
# lengths, only so the dry run has something to integrate. handover_loop.py refuses to drive
# the robot while this is True. Swap in Tian's URDF kinematics behind the same two functions.
PLACEHOLDER = True

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

SHOULDER_HEIGHT = 0.12
UPPER_ARM = 0.116
FOREARM = 0.135
WRIST_TO_TIP = 0.10


def forward_kinematics(joints_deg: np.ndarray) -> np.ndarray:
    pan, lift, elbow, wrist = np.radians(joints_deg[:4])
    a1, a2, a3 = lift, lift + elbow, lift + elbow + wrist
    reach = UPPER_ARM * np.sin(a1) + FOREARM * np.sin(a2) + WRIST_TO_TIP * np.sin(a3)
    height = SHOULDER_HEIGHT + UPPER_ARM * np.cos(a1) + FOREARM * np.cos(a2) + WRIST_TO_TIP * np.cos(a3)
    return np.array([reach * np.cos(pan), reach * np.sin(pan), height])


def inverse_kinematics(target_xyz: np.ndarray, seed_deg: np.ndarray) -> np.ndarray | None:
    x, y, z = target_xyz
    pan, reach = np.arctan2(y, x), np.hypot(x, y)
    for approach in np.radians(np.arange(180, 85, -5)):
        planar = _two_link(reach - WRIST_TO_TIP * np.sin(approach), z - SHOULDER_HEIGHT - WRIST_TO_TIP * np.cos(approach))
        if planar is not None:
            lift, elbow = planar
            return np.array([*np.degrees([pan, lift, elbow, approach - lift - elbow]), seed_deg[4]])
    return None


def _two_link(reach: float, height: float) -> tuple[float, float] | None:
    dist = np.hypot(reach, height)
    if dist > UPPER_ARM + FOREARM or dist < abs(UPPER_ARM - FOREARM):
        return None
    elbow = np.arccos(np.clip((dist**2 - UPPER_ARM**2 - FOREARM**2) / (2 * UPPER_ARM * FOREARM), -1.0, 1.0))
    lift = np.arctan2(reach, height) - np.arctan2(FOREARM * np.sin(elbow), UPPER_ARM + FOREARM * np.cos(elbow))
    return lift, elbow
