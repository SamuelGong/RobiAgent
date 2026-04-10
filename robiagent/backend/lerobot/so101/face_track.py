#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标：
- 显式建模手机中心 p_phone 和屏幕法向 n_screen
- 用 face 3D 点 + 简化前向运动学，追求：
    屏幕法向尽量指向人脸
- 代码既能直接运行，也能被其他 Python 脚本调用
- 支持限时跟踪：跟踪 N 秒后保持最后姿态，而不是回 HOME

依赖：
    pip install numpy opencv-python mediapipe
"""
from __future__ import annotations
import math
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np


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


# =========================
# LeRobot import
# =========================

def import_so101_classes():
    try:
        from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
        return SO101FollowerConfig, SO101Follower
    except Exception:
        from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
        return SO101FollowerConfig, SO101Follower


# =========================
# Configs
# =========================

@dataclass
class CameraConfig:
    # intrinsics
    fx: float
    fy: float
    cx: float
    cy: float

    id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 60


@dataclass
class GeometryConfig:
    """
    简化几何模型参数（单位：米）
    相机坐标系：
        X 向右
        Y 向下
        Z 向前
    """
    # monocular face depth approx
    face_width_m: float = 0.18

    # shoulder_pan 轴点在相机坐标系下的位置
    pan_axis_cam: tuple[float, float, float] = (0.055, 0.30, -0.13)

    # 从 pan 轴点到 lift 轴点的连杆偏移（在 pan=0 的局部坐标里）
    pan_to_lift_home: tuple[float, float, float] = (0.00, -0.075, 0.03)

    # 从 lift 轴点到手机中心的偏移（在 lift=0 的局部坐标里）
    lift_to_phone_home: tuple[float, float, float] = (0.00, 0.00, 0.12)

    # 屏幕法向在 home 几何下的方向（局部坐标）
    screen_normal_home: tuple[float, float, float] = (0.00, 0.00, 0.12)


@dataclass
class TrackingParams:
    # tracker / control
    control_hz: float = 50.0
    alpha_xyz: float = 0.18
    lost_timeout: float = 2.0
    min_detection_confidence: float = 0.6

    # timed tracking
    track_duration_s: float | None = None
    hold_last_on_timeout: bool = True


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
    pan_min_offset: float = -45.0
    pan_max_offset: float = 45.0
    lift_min_offset: float = -45.0
    lift_max_offset: float = 45.0

    # 内部状态初始化常量
    face_x_s: float = -0.05
    face_y_s: float = 0.0
    face_z_s: float = 0.8


# =========================
# Face Tracker
# =========================

class FaceTracker:
    """
    MediaPipe Face Detector
    输出：
    - bbox
    - center: 优先用“两眼中心”，否则回退到 bbox center
    """

    def __init__(self, model_path: str, min_detection_confidence: float = 0.6):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = FaceDetector.create_from_options(options)

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
        config = SO101FollowerConfig(port=self.port, id=self.robot_id)
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
    简化 2-DoF 几何模型：
    - 只控制 shoulder_pan 和 shoulder_lift
    - 其他关节固定在 HOME
    - pan 绕相机坐标的 Y 轴转
    - lift 绕 pan 后局部坐标的 X 轴转
    """

    def __init__(self, home_pose, body_config, internal_config):
        self.home_pose = dict(home_pose)
        self.geometry_config = body_config.geometry
        self.internal_config = internal_config

        self.pan_axis_cam = np.array(self.geometry_config.pan_axis_cam, dtype=np.float64)
        self.pan_to_lift_home = np.array(self.geometry_config.pan_to_lift_home, dtype=np.float64)
        self.lift_to_phone_home = np.array(self.geometry_config.lift_to_phone_home, dtype=np.float64)
        self.screen_normal_home = normalize(np.array(self.geometry_config.screen_normal_home, dtype=np.float64))

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
        """
        用当前几何模型估计 p_phone / n_screen，然后追求：
            n_screen 尽量指向 (face - p_phone)
        """
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
        disable_calibration: bool = True,
    ):
        self.task_config = task_config
        self.body_config = body_config
        self.camera_config = camera_config
        self.internal_config = InternalParams()
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

        self.last_face_time = 0.0
        self.last_control_time = 0.0
        self.connected = False

        self.track_start_time: float | None = None
        self.tracking_active: bool = True
        self.holding_last_pose: bool = False

    def connect(self):
        self.arm.connect()
        self.geom_model = SimplePhoneGeometryModel(
            self.arm.home,
            self.body_config,
            self.internal_config
        )
        self.target_pose = dict(self.arm.home)
        self.connected = True

        self.track_start_time = time.time()
        self.tracking_active = True
        self.holding_last_pose = False

    def disconnect(self):
        try:
            self.tracker.close()
        finally:
            self.arm.disconnect()
            self.connected = False

    def refresh_home(self):
        self.arm.home = self.arm.get_pose()
        self.geom_model = SimplePhoneGeometryModel(self.arm.home, self.body_config, self.internal_config)
        self.target_pose = dict(self.arm.home)

    def restart_tracking(self):
        self.track_start_time = time.time()
        self.tracking_active = True
        self.holding_last_pose = False

    def _update_tracking_timeout(self):
        if self.task_config.track_duration_s is None:
            return

        if self.track_start_time is None:
            self.track_start_time = time.time()
            return

        elapsed = time.time() - self.track_start_time
        if elapsed >= self.task_config.track_duration_s:
            self.tracking_active = False
            self.holding_last_pose = self.task_config.hold_last_on_timeout

    def estimate_face_3d_from_bbox(
        self,
        u: float,
        v: float,
        face_width_px: float,
    ) -> np.ndarray:
        fx = self.camera_config.fx
        fy = self.camera_config.fy
        cx = self.camera_config.cx
        cy = self.camera_config.cy

        face_width_px = max(face_width_px, 1.0)
        Z = fx * self.body_config.geometry.face_width_m / face_width_px
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z], dtype=np.float64)

    def process_frame(
        self,
        frame_bgr,
        timestamp_ms: Optional[int] = None,
        execute: bool = True,
    ) -> dict[str, Any]:
        """
        对外暴露的逐帧调用接口。
        外部脚本可以循环调用它。
        """
        if not self.connected:
            raise RuntimeError("Follower is not connected. Call connect() first.")
        if self.geom_model is None or self.target_pose is None:
            raise RuntimeError("Geometry model / target pose not initialized.")

        now = time.time()
        self._update_tracking_timeout()

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

        if not self.tracking_active and self.holding_last_pose:
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
            if now - self.last_face_time > self.task_config.lost_timeout:
                if execute:
                    self.arm.go_home()
                self.target_pose = dict(self.arm.home)
                result["overlay"]["status"] = "LOST -> HOME"
            else:
                result["overlay"]["status"] = "NO FACE"
            result["ok"] = True
            return result

        self.last_face_time = now
        result["has_face"] = True

        x, y, bw, bh = face["bbox"]
        u, v = face["center"]
        face_width_px = face["width_px"]

        face_xyz = self.estimate_face_3d_from_bbox(u, v, face_width_px)

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

            yaw = result["yaw_des_rad"]
            pitch = result["pitch_des_rad"]

            line1 = f"FACE XYZ=({face_xyz[0]:+.2f},{face_xyz[1]:+.2f},{face_xyz[2]:.2f})m"
            line2 = f"PHONE XYZ=({p_phone[0]:+.2f},{p_phone[1]:+.2f},{p_phone[2]:.2f})m"
            line3 = f"screen_n=({n_screen[0]:+.2f},{n_screen[1]:+.2f},{n_screen[2]:+.2f})"
            line4 = f"des_n=({desired_normal[0]:+.2f},{desired_normal[1]:+.2f},{desired_normal[2]:+.2f})"
            line5 = f"yaw={None if yaw is None else round(math.degrees(yaw),1)} pitch={None if pitch is None else round(math.degrees(pitch),1)}"
            line6 = f"pan={self.target_pose['shoulder_pan.pos']:.2f} lift={self.target_pose['shoulder_lift.pos']:.2f}"

            cv2.putText(display, line1, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, line2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, line3, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.putText(display, line4, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.putText(display, line5, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.putText(display, line6, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        else:
            status = overlay.get("status", "NO FACE")
            color = (0, 255, 255)
            if status == "HOLD_LAST":
                color = (255, 255, 0)
            cv2.putText(display, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return display

    def run_forever(self, return_on_finish: bool = True) -> dict[str, Any] | None:
        cap = open_camera(
            camera_id=self.camera_config.id,
            width=self.camera_config.width,
            height=self.camera_config.height,
            fps=self.camera_config.fps,
        )

        last_result = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                result = self.process_frame(frame, execute=True)
                last_result = result

                display = self.draw_debug(frame, result)
                cv2.imshow("so101 phase-b model", display)
                key = cv2.waitKey(1) & 0xFF

                if return_on_finish and result.get("finished", False):
                    break

                if key == 27 or key == ord("q"):
                    break
                elif key == ord("h"):
                    self.arm.go_home()
                    self.target_pose = dict(self.arm.home)
                elif key == ord("r"):
                    self.refresh_home()
                    print("HOME refreshed from current pose.")
                elif key == ord("t"):
                    self.restart_tracking()
                    print("Tracking restarted.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return last_result
