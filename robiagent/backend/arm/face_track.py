from __future__ import annotations

import cv2
import math
import time
import platform
import numpy as np
import mediapipe as mp
from typing import Any, Optional
from dataclasses import dataclass

from robiagent.utils.misc import suppress_native_stderr

from robiagent.backend.arm.tracking_session import TrackingSession, run_face_tracking_loop


# =========================
# Utility
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ema(prev: float, cur: float, alpha: float) -> float:
    return alpha * cur + (1.0 - alpha) * prev


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def rot_x(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float64)


def rot_y(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def open_camera(camera_id: int, width: int, height: int, fps: int):
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(camera_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    return cap


def import_so101_classes():
    try:
        from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
        return SO101FollowerConfig, SO101Follower
    except Exception:
        from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
        return SO101FollowerConfig, SO101Follower

def estimate_face_3d_from_bbox(
    camera_config,
    face_width_m,
    u: float,
    v: float,
    face_width_px: float,
) -> np.ndarray:
    fx = camera_config.fx
    fy = camera_config.fy
    cx = camera_config.cx
    cy = camera_config.cy

    face_width_px = max(face_width_px, 1.0)
    Z = fx * face_width_m / face_width_px
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)


# =========================
# Configs
# =========================

@dataclass
class InternalParams:
    # joint mapping
    pan_sign: float = 1.0
    lift_sign: float = 1.0

    # 单位：joint_units / radian
    pan_units_per_rad: float = 45.0
    lift_units_per_rad: float = 45.0

    # bias
    pan_bias: float = 0.0
    lift_bias: float = 0.0

    # deadband
    yaw_deadband_rad: float = 0.02
    pitch_deadband_rad: float = 0.02

    # single-step limit
    max_pan_step: float = 5.0
    max_lift_step: float = 5.0

    # safe window relative to HOME
    pan_min_offset: float = -90.0
    pan_max_offset: float = 90.0
    lift_min_offset: float = -90.0
    lift_max_offset: float = 90.0

    # 内部状态初始化常量
    face_x_s: float = -0.05
    face_y_s: float = 0.0
    face_z_s: float = 0.8


# =========================
# Face Tracker
# =========================

class FaceTracker:
    def __init__(self, model_path: str, min_detection_confidence: float = 0.6):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        with suppress_native_stderr():
            self.detector = FaceDetector.create_from_options(
                FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.VIDEO,
                    min_detection_confidence=min_detection_confidence,
                )
            )

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass

    def detect_largest_face(self, frame_bgr, timestamp_ms: int):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if result is None or not result.detections:
            return None

        best = None
        best_area = -1

        for det in result.detections:
            bbox = det.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            bw = int(bbox.width)
            bh = int(bbox.height)

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            area = bw * bh
            if area > best_area:
                best_area = area
                best = (det, x, y, bw, bh)

        if best is None:
            return None

        det, x, y, bw, bh = best

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        eye_center_found = False

        try:
            if hasattr(det, "keypoints") and det.keypoints and len(det.keypoints) >= 2:
                kp0 = det.keypoints[0]
                kp1 = det.keypoints[1]
                lx = kp0.x * w
                ly = kp0.y * h
                rx = kp1.x * w
                ry = kp1.y * h
                cx = 0.5 * (lx + rx)
                cy = 0.5 * (ly + ry)
                eye_center_found = True
        except Exception:
            pass

        return {
            "bbox": (x, y, bw, bh),
            "center": (cx, cy),
            "width_px": float(bw),
            "height_px": float(bh),
            "eye_center_found": eye_center_found,
        }


# =========================
# Robot Controller
# =========================

class SO101Controller:
    def __init__(self, port: str, robot_id: str, disable_calibration: bool = False):
        self.port = port
        self.robot_id = robot_id
        self.disable_calibration = disable_calibration

        self.robot = None
        self.action_keys: list[str] = []
        self.home: dict[str, float] | None = None

    def connect(self):
        SO101FollowerConfig, SO101Follower = import_so101_classes()
        # Keep holding torque even after normal disconnect / program exit.
        # lerobot's SOFollower.disconnect() can disable torque depending on this flag.
        config = SO101FollowerConfig(port=self.port, id=self.robot_id, disable_torque_on_disconnect=False)
        self.robot = SO101Follower(config)
        self.robot.connect(calibrate=not self.disable_calibration)

        self.action_keys = list(self.robot.action_features.keys())
        self.home = self.get_pose()

        required = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        for k in required:
            if k not in self.action_keys:
                raise RuntimeError(f"Missing required action key: {k}")

    def disconnect(self):
        if self.robot is not None:
            # Intentionally do NOT disable torque here.
            # For demo / tracking, we want motors to keep holding their last pose
            # after normal program exit (as long as they remain powered).
            self.robot.disconnect()
            self.robot = None

    def get_pose(self) -> dict[str, float]:
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        obs = self.robot.get_observation()
        return {k: float(obs[k]) for k in self.action_keys}

    def send_pose(self, pose_dict: dict[str, float]):
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        self.robot.send_action(dict(pose_dict))

    def go_home(self):
        if self.home is None:
            raise RuntimeError("Robot home pose not initialized.")
        self.send_pose(self.home)


# =========================
# Simplified Geometry Model
# =========================

class SimplePhoneGeometryModel:
    """
    forward 函数：关节 → 相机系里手机位姿
    face_to_desired_pose 函数：人脸 3D 位置 + 上次位姿 → 当前让屏幕落在人脸前方 pan/lift 所需的动作步（带约束）
    """
    def __init__(self, home_pose, task_config, internal_config):
        self.home_pose = dict(home_pose)
        self.internal_config = internal_config

        self.pan_axis_cam = np.array(task_config.geometry.pan_axis_cam, dtype=np.float64)
        self.pan_to_lift_home = np.array(task_config.geometry.pan_to_lift_home, dtype=np.float64)
        self.lift_to_phone_home = np.array(task_config.geometry.lift_to_phone_home, dtype=np.float64)
        self.screen_normal_home = normalize(np.array(task_config.geometry.screen_normal_home, dtype=np.float64))

    def pose_to_joint_angles(self, pose: dict[str, float]) -> tuple[float, float]:
        pan_units = pose["shoulder_pan.pos"] - self.home_pose["shoulder_pan.pos"] - self.internal_config.pan_bias
        lift_units = pose["shoulder_lift.pos"] - self.home_pose["shoulder_lift.pos"] - self.internal_config.lift_bias

        pan_rad = self.internal_config.pan_sign * (pan_units / self.internal_config.pan_units_per_rad)
        lift_rad = self.internal_config.lift_sign * (lift_units / self.internal_config.lift_units_per_rad)
        return pan_rad, lift_rad

    def forward(self, pose: dict[str, float]) -> dict[str, Any]:
        pan_rad, lift_rad = self.pose_to_joint_angles(pose)

        R_pan = rot_y(pan_rad)
        R_lift = rot_x(lift_rad)
        R_total = R_pan @ R_lift

        p_pan = self.pan_axis_cam
        p_lift = p_pan + R_pan @ self.pan_to_lift_home
        p_phone = p_lift + R_total @ self.lift_to_phone_home
        n_screen = normalize(R_total @ self.screen_normal_home)

        return {
            "pan_rad": pan_rad,
            "lift_rad": lift_rad,
            "p_pan": p_pan,
            "p_lift": p_lift,
            "p_phone": p_phone,
            "n_screen": n_screen,
        }

    def face_to_desired_pose(
        self,
        face_xyz_cam: np.ndarray,
        prev_target_pose: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        fk = self.forward(prev_target_pose)
        p_phone = fk["p_phone"]
        vec_to_face = face_xyz_cam - p_phone
        desired_normal = normalize(vec_to_face)

        if np.linalg.norm(desired_normal) < 1e-6:
            return dict(prev_target_pose), {
                "yaw_des_rad": 0.0,
                "pitch_des_rad": 0.0,
                "p_phone": p_phone,
                "n_screen": fk["n_screen"],
                "desired_normal": desired_normal,
            }

        yaw_des = math.atan2(desired_normal[0], desired_normal[2])
        n_after_unyaw = rot_y(-yaw_des) @ desired_normal
        pitch_des = math.atan2(n_after_unyaw[1], n_after_unyaw[2])

        if abs(yaw_des) < self.internal_config.yaw_deadband_rad:
            yaw_des = 0.0
        if abs(pitch_des) < self.internal_config.pitch_deadband_rad:
            pitch_des = 0.0

        desired_pan = (
            self.home_pose["shoulder_pan.pos"]
            + self.internal_config.pan_bias
            + self.internal_config.pan_sign * self.internal_config.pan_units_per_rad * yaw_des
        )
        desired_lift = (
            self.home_pose["shoulder_lift.pos"]
            + self.internal_config.lift_bias
            + self.internal_config.lift_sign * self.internal_config.lift_units_per_rad * pitch_des
        )

        desired_pan = clamp(
            desired_pan,
            self.home_pose["shoulder_pan.pos"] + self.internal_config.pan_min_offset,
            self.home_pose["shoulder_pan.pos"] + self.internal_config.pan_max_offset,
        )
        desired_lift = clamp(
            desired_lift,
            self.home_pose["shoulder_lift.pos"] + self.internal_config.lift_min_offset,
            self.home_pose["shoulder_lift.pos"] + self.internal_config.lift_max_offset,
        )

        pan_step = clamp(
            desired_pan - prev_target_pose["shoulder_pan.pos"],
            -self.internal_config.max_pan_step,
            self.internal_config.max_pan_step,
        )
        lift_step = clamp(
            desired_lift - prev_target_pose["shoulder_lift.pos"],
            -self.internal_config.max_lift_step,
            self.internal_config.max_lift_step,
        )

        next_pose = dict(prev_target_pose)
        next_pose["shoulder_pan.pos"] += pan_step
        next_pose["shoulder_lift.pos"] += lift_step

        next_pose["elbow_flex.pos"] = self.home_pose["elbow_flex.pos"]
        next_pose["wrist_flex.pos"] = self.home_pose["wrist_flex.pos"]
        next_pose["wrist_roll.pos"] = self.home_pose["wrist_roll.pos"]
        next_pose["gripper.pos"] = self.home_pose["gripper.pos"]

        diag = {
            "yaw_des_rad": yaw_des,
            "pitch_des_rad": pitch_des,
            "p_phone": p_phone,
            "n_screen": fk["n_screen"],
            "desired_normal": desired_normal,
        }
        return next_pose, diag


# =========================
# Main follower class
# =========================

class FaceTrack:
    def __init__(
        self,
        task_config,
        body_config,
        camera_config,
        internal_config=None,
        disable_calibration: bool = True,
    ):
        self.task_config = task_config
        self.body_config = body_config
        self.camera_config = camera_config
        self.internal_config = internal_config or InternalParams()
        self.disable_calibration = disable_calibration

        self.arm = SO101Controller(
            port=body_config.port,
            robot_id=body_config.id,
            disable_calibration=disable_calibration,
        )
        self.tracker = FaceTracker(
            model_path=task_config.model_path,
            min_detection_confidence=task_config.min_detection_confidence,
        )

        self.geom_model: Optional[SimplePhoneGeometryModel] = None

        self.target_pose: Optional[dict[str, float]] = None
        self.face_x_s = self.internal_config.face_x_s
        self.face_y_s = self.internal_config.face_y_s
        self.face_z_s = self.internal_config.face_z_s

        self.session = TrackingSession(task_config)
        self.last_control_time = 0.0
        self.connected = False

    def connect(self):
        self.arm.connect()
        self.geom_model = SimplePhoneGeometryModel(
            self.arm.home,
            self.task_config,
            self.internal_config
        )
        self.target_pose = dict(self.arm.home)
        self.connected = True

    def disconnect(self):
        try:
            self.tracker.close()
        finally:
            self.arm.disconnect()
            self.connected = False

    def refresh_home(self):
        self.arm.home = self.arm.get_pose()
        self.geom_model = SimplePhoneGeometryModel(self.arm.home, self.task_config, self.internal_config)
        self.target_pose = dict(self.arm.home)

    def restart_tracking(self):
        self.session.restart()

    def process_frame(
        self,
        frame_bgr,
        timestamp_ms: Optional[int] = None,
        execute: bool = True,
    ) -> dict[str, Any]:
        if not self.connected:
            raise RuntimeError("Follower is not connected. Call connect() first.")
        if self.geom_model is None or self.target_pose is None:
            raise RuntimeError("Geometry model / target pose not initialized.")

        now = time.time()
        self.session.update_timeout(now)

        if timestamp_ms is None:
            timestamp_ms = int(now * 1000)

        result: dict[str, Any] = {
            "ok": False,
            "finished": False,
            "has_face": False,
            "face_xyz": None,
            "p_phone": None,
            "n_screen": None,
            "desired_normal": None,
            "yaw_des_rad": None,
            "pitch_des_rad": None,
            "target_pose": dict(self.target_pose),
            "overlay": {},
        }

        if self.session.timed_out_hold_last():
            fk = self.geom_model.forward(self.target_pose)
            result["ok"] = True
            result["finished"] = True
            result["has_face"] = False
            result["p_phone"] = fk["p_phone"]
            result["n_screen"] = fk["n_screen"]
            result["target_pose"] = dict(self.target_pose)
            result["overlay"] = {
                "status": "HOLD_LAST",
            }
            return result

        face = self.tracker.detect_largest_face(frame_bgr, timestamp_ms)

        if face is None:
            if now - self.session.last_face_time > self.task_config.lost_timeout:
                if execute:
                    self.arm.go_home()
                self.target_pose = dict(self.arm.home)
                result["overlay"]["status"] = "LOST -> HOME"
            else:
                result["overlay"]["status"] = "NO FACE"
            result["ok"] = True
            return result

        self.session.mark_face_seen(now)
        result["has_face"] = True

        x, y, bw, bh = face["bbox"]
        u, v = face["center"]
        face_width_px = face["width_px"]

        face_xyz = estimate_face_3d_from_bbox(self.camera_config, self.task_config.geometry.face_width_m,
                                              u, v, face_width_px)

        self.face_x_s = ema(self.face_x_s, float(face_xyz[0]), self.task_config.alpha_xyz)
        self.face_y_s = ema(self.face_y_s, float(face_xyz[1]), self.task_config.alpha_xyz)
        self.face_z_s = ema(self.face_z_s, float(face_xyz[2]), self.task_config.alpha_xyz)
        face_xyz_s = np.array([self.face_x_s, self.face_y_s, self.face_z_s], dtype=np.float64)

        result["face_xyz"] = face_xyz_s

        if now - self.last_control_time >= 1.0 / self.task_config.control_hz:
            next_pose, diag = self.geom_model.face_to_desired_pose(
                face_xyz_cam=face_xyz_s,
                prev_target_pose=self.target_pose,
            )
            self.target_pose = dict(next_pose)
            if execute:
                self.arm.send_pose(self.target_pose)
            self.last_control_time = now
        else:
            fk = self.geom_model.forward(self.target_pose)
            diag = {
                "yaw_des_rad": None,
                "pitch_des_rad": None,
                "p_phone": fk["p_phone"],
                "n_screen": fk["n_screen"],
                "desired_normal": normalize(face_xyz_s - fk["p_phone"]),
            }

        result["p_phone"] = diag["p_phone"]
        result["n_screen"] = diag["n_screen"]
        result["desired_normal"] = diag["desired_normal"]
        result["yaw_des_rad"] = diag["yaw_des_rad"]
        result["pitch_des_rad"] = diag["pitch_des_rad"]
        result["target_pose"] = dict(self.target_pose)

        result["overlay"] = {
            "bbox": (x, y, bw, bh),
            "center": (u, v),
            "eye_center_found": face.get("eye_center_found", False),
            "status": "FOLLOW",
        }
        result["ok"] = True
        return result

    def draw_debug(self, frame_bgr, result: dict[str, Any]):
        display = frame_bgr.copy()
        overlay = result.get("overlay", {})

        if result.get("has_face", False):
            x, y, bw, bh = overlay["bbox"]
            u, v = overlay["center"]
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(display, (int(u), int(v)), 4, (0, 0, 255), -1)

            if overlay.get("eye_center_found", False):
                cv2.putText(
                    display,
                    "eye-center",
                    (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            face_xyz = result["face_xyz"]
            p_phone = result["p_phone"]
            n_screen = result["n_screen"]
            desired_normal = result["desired_normal"]

            line1 = f"FACE XYZ=({face_xyz[0]:+.2f},{face_xyz[1]:+.2f},{face_xyz[2]:.2f})m"
            line2 = f"PHONE XYZ=({p_phone[0]:+.2f},{p_phone[1]:+.2f},{p_phone[2]:.2f})m"
            line3 = f"screen_n=({n_screen[0]:+.2f},{n_screen[1]:+.2f},{n_screen[2]:+.2f})"
            line4 = f"desired_n=({desired_normal[0]:+.2f},{desired_normal[1]:+.2f},{desired_normal[2]:+.2f})"
            line5 = f"pan={self.target_pose['shoulder_pan.pos']:.2f} lift={self.target_pose['shoulder_lift.pos']:.2f}"

            cv2.putText(display, line1, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, line2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, line3, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.putText(display, line4, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.putText(display, line5, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        else:
            status = overlay.get("status", "NO FACE")
            color = (0, 255, 255)
            if status == "HOLD_LAST":
                color = (255, 255, 0)
            cv2.putText(display, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return display

    def run_forever(self, return_on_finish=True, args=None,
                    execute=True) -> dict[str, Any] | None:
        self.session.begin_after_connect()

        def on_key(key: int) -> None:
            if key == ord("h"):
                self.arm.go_home()
                self.target_pose = dict(self.arm.home)
            elif key == ord("r"):
                self.refresh_home()
                print("HOME refreshed from current pose.")
            elif key == ord("t"):
                self.restart_tracking()
                print("Tracking restarted.")

        return run_face_tracking_loop(
            self,
            self.camera_config,
            window_title="Face Tracking",
            return_on_finish=return_on_finish,
            execute=execute,
            on_key=on_key,
        )
