import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
PALM = (0, 5, 9, 13, 17)
STRAIGHT_ABOVE = 0.9


def image_points(landmarks, width, height):
    return np.array([[lm.x * width, lm.y * height] for lm in landmarks])


def world_points(world_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])


def palm_centre_px(points_px):
    return points_px[list(PALM)].mean(axis=0)


def palm_facing_camera(points_px, handedness):
    # Image axes: x right, y down. For a right hand whose palm faces the camera, index knuckle x pinky
    # knuckle (from the wrist) is negative; a left hand is the mirror image. A right palm and a left back
    # look the same in 2D, so this is only as good as MediaPipe's handedness label.
    v_index = points_px[5] - points_px[WRIST]
    v_pinky = points_px[17] - points_px[WRIST]
    cross = v_index[0] * v_pinky[1] - v_index[1] * v_pinky[0]
    return cross < 0 if handedness == "Right" else cross > 0


def finger_straightness(world, chain):
    segments = np.linalg.norm(np.diff(world[list(chain)], axis=0), axis=1).sum()
    return np.linalg.norm(world[chain[-1]] - world[chain[0]]) / segments


def extended_fingers(world, threshold=STRAIGHT_ABOVE, thumb_threshold=0.85):
    extended = [name for name, chain in FINGERS.items() if finger_straightness(world, chain) > threshold]
    if finger_straightness(world, THUMB) > thumb_threshold:
        extended.append("thumb")
    return extended


def index_direction_px(points_px):
    direction = points_px[8] - points_px[5]
    return direction / np.linalg.norm(direction)


def describe(result, index, width, height):
    landmarks = result.hand_landmarks[index]
    world = world_points(result.hand_world_landmarks[index])
    category = result.handedness[index][0]
    points = image_points(landmarks, width, height)
    return {
        "handedness": category.category_name,
        "handedness_score": round(category.score, 3),
        "palm_centre_px": tuple(np.round(palm_centre_px(points), 1)),
        "palm_facing_camera": bool(palm_facing_camera(points, category.category_name)),
        "extended": extended_fingers(world),
        "straightness": {name: round(float(finger_straightness(world, chain)), 3) for name, chain in FINGERS.items()},
        "index_tip_px": tuple(points[8]),
        "index_direction_px": tuple(index_direction_px(points)),
    }


def is_offering(hand, min_extended=3):
    return hand["palm_facing_camera"] and len(hand["extended"]) >= min_extended


def is_pointing(hand, curled_below=0.75):
    straightness = hand["straightness"]
    others_curled = all(straightness[name] < curled_below for name in ("middle", "ring", "pinky"))
    return straightness["index"] > STRAIGHT_ABOVE and others_curled
