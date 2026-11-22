import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calib" / "cross_check.py"
SIZE = (640, 480)
K_TRUE = np.array([[560.0, 0, 322.0], [0, 562.0, 238.0], [0, 0, 1.0]])
DIST_TRUE = np.array([-0.18, 0.06, 0.0005, -0.0004, 0.0])
SQUARE_MM = (28.75, 29.0)
SQUARES = (8, 6)
CAMERA_MM = np.array([300.0, -60.0, 560.0])
THICKNESS_MM = 3.0
SUPERSAMPLE = 2
rng = np.random.default_rng(7)


def look_down(position, target):
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


def pixel_rays():
    w, h = SIZE
    u, v = np.meshgrid((np.arange(w * SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5, (np.arange(h * SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5)
    points = np.column_stack([u.ravel(), v.ravel()]).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    return cv2.undistortPointsIter(points, K_TRUE, DIST_TRUE, None, None, criteria).reshape(-1, 2)


def render(R_world_cam, position, to_board, plane_z, with_tape=False):
    """Pixels -> rays through the distorted camera -> plane -> board coordinates -> checker colour."""
    w, h = SIZE
    normalized = pixel_rays()
    rays = np.column_stack([normalized, np.ones(len(normalized))]) @ R_world_cam.T
    hit = position + ((plane_z - position[2]) / rays[:, 2:3]) * rays
    a, b = to_board(hit[:, :2])
    sx, sy = SQUARE_MM
    on_board = (a >= 0) & (a < SQUARES[0] * sx) & (b >= 0) & (b < SQUARES[1] * sy)
    black = ((np.floor(a / sx) + np.floor(b / sy)) % 2 == 0) & on_board
    paper = (a > -1.3 * sx) & (a < (SQUARES[0] + 1.3) * sx) & (b > -1.3 * sy) & (b < (SQUARES[1] + 1.3) * sy)
    gray = np.where(black, 25.0, np.where(paper, 235.0, 120.0))
    image = np.repeat(gray[:, None], 3, axis=1)
    if with_tape:
        tape = (np.abs(a - (-0.7 * sx)) < 0.45 * sx) & (np.abs(b - (-0.7 * sy)) < 0.45 * sy)
        image[tape] = (30, 30, 220)
    image = image.reshape(h * SUPERSAMPLE, w * SUPERSAMPLE, 3).astype(np.float32)
    image = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.6) + rng.normal(0, 1.0, image.shape), 0, 255).astype(np.uint8)


def intrinsic_frames(folder, count=25):
    folder.mkdir(parents=True)
    saved = 0
    while saved < count:
        distance = rng.uniform(380, 560)
        centre = np.array([rng.uniform(-90, 90), rng.uniform(-60, 60), 0.0])
        yaw, tilt_axis, tilt = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(0, 40))
        rotation = cv2.Rodrigues(np.array([np.cos(tilt_axis), np.sin(tilt_axis), 0.0]) * tilt)[0] @ cv2.Rodrigues(np.array([0, 0, yaw]))[0]
        board_centre = np.array([SQUARES[0] * SQUARE_MM[0] / 2, SQUARES[1] * SQUARE_MM[1] / 2, 0.0])
        # Board frame -> camera frame: rotate about the board's centre, then put the centre at (centre x, y, distance).
        to_cam = lambda p: (np.column_stack([p, np.zeros(len(p))]) - board_centre) @ rotation.T + centre + [0, 0, distance]
        inner = np.mgrid[1:SQUARES[0], 1:SQUARES[1]].reshape(2, -1).T * SQUARE_MM
        projected = cv2.projectPoints(to_cam(inner), np.zeros(3), np.zeros(3), K_TRUE, DIST_TRUE)[0].reshape(-1, 2)
        if (projected < 12).any() or (projected > np.array(SIZE) - 12).any() or to_cam(inner)[:, 2].min() < 150:
            continue

        image = render_tilted(rotation, centre, distance, board_centre)
        cv2.imwrite(str(folder / f"board_{saved:02d}.png"), image)
        saved += 1


def render_tilted(rotation, centre, distance, board_centre):
    """A tilted board in front of an identity camera: intersect each ray with the board's plane."""
    w, h = SIZE
    normalized = pixel_rays()
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    plane_point, plane_normal = centre + [0, 0, distance], rotation[:, 2]
    hit = rays * ((plane_point @ plane_normal) / (rays @ plane_normal))[:, None]
    local = (hit - plane_point) @ rotation + board_centre
    a, b = local[:, 0], local[:, 1]
    sx, sy = SQUARE_MM
    on_board = (a >= 0) & (a < SQUARES[0] * sx) & (b >= 0) & (b < SQUARES[1] * sy)
    black = ((np.floor(a / sx) + np.floor(b / sy)) % 2 == 0) & on_board
    paper = (a > -1.3 * sx) & (a < (SQUARES[0] + 1.3) * sx) & (b > -1.3 * sy) & (b < (SQUARES[1] + 1.3) * sy)
    gray = np.where(black, 25.0, np.where(paper, 235.0, 120.0)).reshape(h * SUPERSAMPLE, w * SUPERSAMPLE).astype(np.float32)
    gray = cv2.GaussianBlur(cv2.resize(gray, SIZE, interpolation=cv2.INTER_AREA), (0, 0), 0.6)
    return np.clip(np.repeat(gray[:, :, None], 3, axis=2) + rng.normal(0, 1.0, (h, w, 1)), 0, 255).astype(np.uint8)


def table_scene(folder, yaw_deg, centre_mm):
    """The board lies flat on the table; returns the true ruler values for corners 1-4 and the true camera rotation."""
    R_cam = look_down(CAMERA_MM, np.array([*centre_mm, 0.0]))
    yaw = np.radians(yaw_deg)
    R2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    sx, sy = SQUARE_MM
    origin_mm = centre_mm - R2 @ [SQUARES[0] * sx / 2, SQUARES[1] * sy / 2]
    to_board = lambda xy: tuple(((xy - origin_mm) @ R2).T)
    image = render(R_cam, CAMERA_MM, to_board, THICKNESS_MM, with_tape=True)
    labelled = np.array([[1, 1], [7, 1], [7, 5], [1, 5]]) * [sx, sy]
    ruler = {str(i + 1): R2 @ p + origin_mm for i, p in enumerate(labelled)}
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "board_00.png"), image)
    return ruler, R_cam


def run(*args):
    result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    print(result.stdout + result.stderr)
    return result


def corner_args(ruler, noise_mm=1.0):
    return [f"{n}: {x + rng.normal(0, noise_mm):.1f}, {y + rng.normal(0, noise_mm):.1f}" for n, (x, y) in ruler.items()]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intrinsic_frames(tmp / "intrinsics")
        yaw_cases = {"board turned 20 deg": 20.0, "board turned 200 deg": 200.0}
        scenes = {name: table_scene(tmp / f"table_{i}", yaw, np.array([340.0, 90.0])) for i, (name, yaw) in enumerate(yaw_cases.items())}

        print("=== case 1: good data, both board orientations")
        for i, (name, (ruler, R_cam)) in enumerate(scenes.items()):
            result = run("--intrinsics-frames", tmp / "intrinsics", "--table-frame", tmp / f"table_{i}", "--corners-mm", *corner_args(ruler),
                         "--thickness-mm", THICKNESS_MM, "--lens-mm", 330, -40, 520, "--out-dir", tmp / f"out_{i}")
            report = yaml.safe_load((tmp / f"out_{i}" / "result.yaml").read_text())
            position = np.array(report["camera_position_mm"])
            assert result.returncode == 0 and report["verdict"] == "GOOD", f"{name}: expected GOOD, got {report['verdict']}"
            assert np.linalg.norm(position - CAMERA_MM) < 5, f"{name}: camera {position} is not {CAMERA_MM}"
            print(f"PASS {name}: camera {position.round(1).tolist()} vs true {CAMERA_MM.tolist()}")

        print("=== case 2: intrinsics recovered")
        result = run("--intrinsics-frames", tmp / "intrinsics", "--out-dir", tmp / "out_k")
        line = next(l for l in result.stdout.splitlines() if l.startswith("fx"))
        fx, fy, cx, cy = (float(line.split()[i]) for i in (1, 3, 5, 7))
        assert abs(fx / K_TRUE[0, 0] - 1) < 0.01 and abs(fy / K_TRUE[1, 1] - 1) < 0.01, line
        assert abs(cx - K_TRUE[0, 2]) < 2.5 and abs(cy - K_TRUE[1, 2]) < 2.5, line
        k1 = float(line.split("dist(k1 k2 p1 p2 k3) [")[1].split(",")[0])
        assert abs(k1 - DIST_TRUE[0]) < 0.03, line
        print(f"PASS intrinsics: {line}")

        print("=== case 3: a mistyped ruler value (30 mm) must not pass")
        ruler = scenes["board turned 20 deg"][0]
        bad = corner_args(ruler)
        bad[2] = f"3: {ruler['3'][0] + 30:.1f}, {ruler['3'][1]:.1f}"
        result = run("--intrinsics-frames", tmp / "intrinsics", "--table-frame", tmp / "table_0", "--corners-mm", *bad, "--thickness-mm", THICKNESS_MM,
                     "--lens-mm", 330, -40, 520, "--out-dir", tmp / "out_bad")
        assert result.returncode == 2 and "VERDICT: BAD" in result.stdout
        print("PASS mistyped corner gives BAD")

        print("=== case 4: wrong picture size is BAD")
        small = tmp / "small"
        small.mkdir()
        for path in sorted((tmp / "intrinsics").glob("*.png"))[:12]:
            cv2.imwrite(str(small / path.name), cv2.resize(cv2.imread(str(path)), (320, 240)))
        result = run("--intrinsics-frames", small, "--out-dir", tmp / "out_small")
        assert result.returncode == 2 and "must be 640x480" in result.stdout
        print("PASS 320x240 frames give BAD")

        print("=== case 5: Tian's YAML (true values) agrees, and a different one is caught")
        K_list, dist_list = K_TRUE.tolist(), DIST_TRUE.tolist()
        ruler, R_cam = scenes["board turned 20 deg"]
        intrinsics = {"created": "t", "image_width": 640, "image_height": 480, "K": K_list, "dist": dist_list, "rms_px": 0.2,
                      "inner_corners": [7, 5], "square_x_m": SQUARE_MM[0] / 1000, "square_y_m": SQUARE_MM[1] / 1000}
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_cam, CAMERA_MM / 1000
        extrinsics = {"intrinsics_created": "t", "T_base_cam": T.tolist(), "board": {"thickness_m": THICKNESS_MM / 1000}}
        for name, shift in (("matching", 0.0), ("camera 40 mm off", 0.040)):
            extrinsics["T_base_cam"][0][3] = float(CAMERA_MM[0] / 1000 + shift)
            (tmp / "intrinsics.yaml").write_text(yaml.safe_dump(intrinsics))
            (tmp / "extrinsics.yaml").write_text(yaml.safe_dump(extrinsics))
            result = run("--intrinsics-frames", tmp / "intrinsics", "--table-frame", tmp / "table_0", "--corners-mm", *corner_args(ruler),
                         "--lens-mm", 330, -40, 520, "--tian-intrinsics", tmp / "intrinsics.yaml", "--tian-extrinsics", tmp / "extrinsics.yaml",
                         "--out-dir", tmp / "out_tian")
            assert result.returncode == (0 if shift == 0 else 2), f"{name}: exit {result.returncode}"
            print(f"PASS Tian's files {name}: exit {result.returncode}")

        jaw_ready = (SCRIPT.parents[1] / "handover" / "check_corners.py").exists()
        print(f"jaw-tip comparison: {'run in case 6' if jaw_ready else 'skipped, scripts/handover/ (PR #13) is not on this branch'}")
        if jaw_ready:
            sys.path.insert(0, str(SCRIPT.parents[1] / "handover"))
            from kinematics import ARM_JOINTS, inverse_kinematics

            print("=== case 6: jaw-tip lines")
            # Joints for points the arm can reach; they differ from the ruler values on purpose: this only exercises the
            # parsing and the comparison, the kinematics are checked by tests/handover_kinematics.py.
            seed = np.array([0.0, 0.0, 90.0, 0.0, 0.0])
            lines = []
            for (n, (x, y)), reachable in zip(ruler.items(), ((0.22, 0.05), (0.22, -0.05), (0.28, -0.05), (0.28, 0.05))):
                joints = inverse_kinematics(np.array([*reachable, THICKNESS_MM / 1000]), seed)
                assert joints is not None, f"no joints for {reachable}"
                lines.append(f"{n}: {x:.1f}, {y:.1f} | " + "  ".join(f"{j}={v:7.2f}" for j, v in zip(ARM_JOINTS, joints)))
            (tmp / "corners.txt").write_text("\n".join(lines))
            result = run("--intrinsics-frames", tmp / "intrinsics", "--table-frame", tmp / "table_0", "--corners-file", tmp / "corners.txt",
                         "--thickness-mm", THICKNESS_MM, "--out-dir", tmp / "out_jaw")
            assert "jaw-tip touches" in result.stdout
    print("\nALL PASS")


if __name__ == "__main__":
    main()
