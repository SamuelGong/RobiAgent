from __future__ import annotations

import cv2
import math
import time
import numpy as np
from typing import Any, Optional
from dataclasses import dataclass

from robiagent.backend.arm.face_track import (
    FaceTracker,
    clamp,
    ema,
    estimate_face_3d_from_bbox,
    normalize,
)
from robiagent.backend.arm.tracking_session import (
    TrackingSession,
    run_face_tracking_loop,
)


def parse_axis_label(axis_label: str) -> tuple[int, float]:
    s = axis_label.strip().upper()
    if s not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise ValueError(f"Unsupported axis label: {axis_label}")
    sign = 1.0 if s[0] == "+" else -1.0
    idx = {"X": 0, "Y": 1, "Z": 2}[s[1]]
    return idx, sign


def build_rotmat_with_forward_axis(
    forward_ik: np.ndarray, axis_label: str
) -> np.ndarray:
    f = normalize(forward_ik)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(f, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t = normalize(np.cross(up, f))

    axis_idx, sign = parse_axis_label(axis_label)
    if axis_idx == 0:
        x_axis = sign * f
        y_axis = t
        z_axis = normalize(np.cross(x_axis, y_axis))
        return np.column_stack([x_axis, y_axis, z_axis])
    if axis_idx == 1:
        y_axis = sign * f
        z_axis = t
        x_axis = normalize(np.cross(y_axis, z_axis))
        return np.column_stack([x_axis, y_axis, z_axis])
    z_axis = sign * f
    x_axis = t
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def extract_forward_from_rotmat(rot: np.ndarray, axis_label: str) -> np.ndarray:
    idx, sign = parse_axis_label(axis_label)
    return normalize(sign * rot[:, idx])


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return normalize(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_rotmat(rot_a: np.ndarray, rot_b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = clamp(alpha, 0.0, 1.0)
    qa = rotmat_to_quat(rot_a)
    qb = rotmat_to_quat(rot_b)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        return quat_to_rotmat(normalize((1.0 - alpha) * qa + alpha * qb))
    theta_0 = math.acos(clamp(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_to_rotmat(normalize(s0 * qa + s1 * qb))


@dataclass
class NewInternalParams:
    urdf_path: str

    # EMA state seeds (camera frame), same defaults as face_track.InternalParams.
    face_x_s: float = -0.05
    face_y_s: float = 0.0
    face_z_s: float = 0.8

    distance_tolerance_m: float = 0.03
    screen_tilt_deg: float = 55.0
    tilt_tolerance_deg: float = 5.0
    ik_target_frame_name: str = "gripper_frame_link"
    # Max Cartesian step (m) per control tick toward IK input; <= 0 disables position slew.
    max_ee_step_m: float = 0.005
    # 1.0 = no extra orientation smoothing; (0, 1) = slerp from last sent rot toward vision.
    ee_ori_slerp_alpha: float = 1.0
    orientation_alpha: float = 0.3
    cam_to_ik_rot: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )
    pan_axis_ik: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_forward_axis: str = "+Z"
    target_plane_below_eye_m: float = 0.05
    ik_mode: str = "hard_bisection"
    ik_bisection_max_depth: int = 5
    ik_pos_tol_m: float = 0.01
    ik_rot_tol_rad: float = 0.10
    # vision_pos - actual_pos slow integral in IK frame; requires torque margin.
    outer_loop_pos_enable: bool = True
    outer_loop_ki_pos: float = 0.05
    outer_loop_bias_max_m: float = 0.10
    outer_loop_reset_on_no_face: bool = True


@dataclass
class EEPoseTarget:
    pos: np.ndarray
    normal: np.ndarray
    wx: float
    wy: float
    wz: float
    distance_error_m: float
    tilt_error_deg: float
    degraded: bool = False
    degraded_reason: str = ""


@dataclass
class EEPoseState:
    pos: np.ndarray
    rvec: np.ndarray
    rot: np.ndarray
    normal: np.ndarray


class SO101AdvancedController:
    def __init__(self, port: str, robot_id: str, internal: NewInternalParams):
        self.port = port
        self.robot_id = robot_id
        self.internal = internal
        self.robot = None
        self.ee_to_joints = None
        self.joints_to_ee = None
        self.action_keys: list[str] = []
        self.home: dict[str, float] | None = None

    def connect(self):
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
        from lerobot.model.kinematics import RobotKinematics
        from lerobot.processor import RobotProcessorPipeline
        from lerobot.processor.converters import (
            robot_action_observation_to_transition,
            transition_to_robot_action,
        )
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            ForwardKinematicsJointsToEE,
            InverseKinematicsEEToJoints,
        )

        # Keep holding torque even after normal disconnect / program exit.
        # lerobot's SOFollower.disconnect() can disable torque depending on this flag.
        self.robot = SOFollower(
            SOFollowerRobotConfig(
                id=self.robot_id,
                port=self.port,
                use_degrees=True,
                disable_torque_on_disconnect=False,
            )
        )
        self.robot.connect(calibrate=False)
        self.action_keys = list(self.robot.action_features.keys())
        self.home = self.get_pose()

        motor_names = list(self.robot.bus.motors.keys())
        kinematics_solver = RobotKinematics(
            urdf_path=self.internal.urdf_path,
            target_frame_name=self.internal.ik_target_frame_name,
            joint_names=motor_names,
        )
        self.ee_to_joints = RobotProcessorPipeline(
            [
                InverseKinematicsEEToJoints(
                    kinematics=kinematics_solver,
                    motor_names=motor_names,
                    initial_guess_current_joints=True,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        self.joints_to_ee = RobotProcessorPipeline(
            [
                ForwardKinematicsJointsToEE(
                    kinematics=kinematics_solver, motor_names=motor_names
                )
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

    def disconnect(self):
        if self.robot is None:
            return
        # Intentionally do NOT disable torque here.
        # For demo / tracking, we want motors to keep holding their last pose
        # after normal program exit (as long as they remain powered).
        self.robot.disconnect()
        self.robot = None

    def get_pose(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {k: float(obs[k]) for k in self.action_keys}

    def get_observation(self) -> dict[str, Any]:
        return self.robot.get_observation()

    def go_home(self):
        self.robot.send_action(dict(self.home))

    @staticmethod
    def _rotation_error_rad(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
        delta = rot_a.T @ rot_b
        tr = clamp((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0)
        return float(math.acos(tr))

    def _build_ee_action(self, target: EEPoseTarget) -> dict[str, float]:
        return {
            "ee.x": float(target.pos[0]),
            "ee.y": float(target.pos[1]),
            "ee.z": float(target.pos[2]),
            "ee.wx": float(target.wx),
            "ee.wy": float(target.wy),
            "ee.wz": float(target.wz),
            "ee.gripper_pos": float(self.home.get("gripper.pos", 0.0)),
        }

    def _finalize_joints_action(self, joints_act: dict[str, float]) -> dict[str, float]:
        if "wrist_roll.pos" in joints_act and "wrist_roll.pos" in self.home:
            joints_act["wrist_roll.pos"] = float(self.home["wrist_roll.pos"])
        if "gripper.pos" in joints_act and "gripper.pos" in self.home:
            joints_act["gripper.pos"] = float(self.home["gripper.pos"])
        return joints_act

    def _pose_to_target(
        self, pose: EEPoseState, template: EEPoseTarget
    ) -> EEPoseTarget:
        rvec = pose.rvec.reshape(3)
        return EEPoseTarget(
            pos=pose.pos.copy(),
            normal=pose.normal.copy(),
            wx=float(rvec[0]),
            wy=float(rvec[1]),
            wz=float(rvec[2]),
            distance_error_m=template.distance_error_m,
            tilt_error_deg=template.tilt_error_deg,
            degraded=template.degraded,
            degraded_reason=template.degraded_reason,
        )

    def _interpolate_pose(
        self, left: EEPoseState, right: EEPoseState, alpha: float
    ) -> EEPoseState:
        pos = (1.0 - alpha) * left.pos + alpha * right.pos
        rot = slerp_rotmat(left.rot, right.rot, alpha)
        rvec, _ = cv2.Rodrigues(rot)
        normal = extract_forward_from_rotmat(rot, self.internal.ee_forward_axis)
        return EEPoseState(pos=pos, rvec=rvec.reshape(3), rot=rot, normal=normal)

    def _try_solve_ee_target_hard(
        self, target: EEPoseTarget, robot_obs: dict[str, Any]
    ) -> tuple[bool, Optional[dict[str, float]], dict[str, Any]]:
        ee_act = self._build_ee_action(target)
        try:
            joints_act = self.ee_to_joints((ee_act, robot_obs))
        except Exception as e:
            return (
                False,
                None,
                {"reason": f"ik_exception:{type(e).__name__}", "error": str(e)},
            )
        joints_act = self._finalize_joints_action(joints_act)
        try:
            ee_obs = self.joints_to_ee((joints_act, robot_obs))
            solved_pos = np.array(
                [float(ee_obs["ee.x"]), float(ee_obs["ee.y"]), float(ee_obs["ee.z"])],
                dtype=np.float64,
            )
            solved_rvec = np.array(
                [
                    float(ee_obs["ee.wx"]),
                    float(ee_obs["ee.wy"]),
                    float(ee_obs["ee.wz"]),
                ],
                dtype=np.float64,
            )
            solved_rot, _ = cv2.Rodrigues(solved_rvec.reshape(3, 1))
            target_rot, _ = cv2.Rodrigues(
                np.array([target.wx, target.wy, target.wz], dtype=np.float64).reshape(
                    3, 1
                )
            )
            pos_err = float(np.linalg.norm(solved_pos - target.pos))
            rot_err = self._rotation_error_rad(solved_rot, target_rot)
            feasible = (
                pos_err <= self.internal.ik_pos_tol_m
                and rot_err <= self.internal.ik_rot_tol_rad
            )
            diag = {
                "reason": "ok" if feasible else "hard_residual_too_large",
                "ik_residual_pos": pos_err,
                "ik_residual_rot": rot_err,
            }
            return feasible, joints_act if feasible else None, diag
        except Exception as e:
            return (
                False,
                None,
                {
                    "reason": f"ik_validate_exception:{type(e).__name__}",
                    "error": str(e),
                },
            )

    def _try_solve_ee_target_soft(
        self, target: EEPoseTarget, robot_obs: dict[str, Any]
    ) -> tuple[bool, Optional[dict[str, float]], dict[str, Any]]:
        ee_act = self._build_ee_action(target)
        try:
            joints_act = self.ee_to_joints((ee_act, robot_obs))
        except Exception as e:
            return (
                False,
                None,
                {"reason": f"soft_ik_exception:{type(e).__name__}", "error": str(e)},
            )
        joints_act = self._finalize_joints_action(joints_act)
        return True, joints_act, {"reason": "soft_ok"}

    def get_actual_ee_state(
        self, robot_obs: Optional[dict[str, Any]] = None
    ) -> tuple[Optional[EEPoseState], str]:
        if self.robot is None or self.joints_to_ee is None:
            return None, "robot_or_fk_not_initialized"
        try:
            if robot_obs is None:
                robot_obs = self.robot.get_observation()
            if all(
                k in robot_obs
                for k in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")
            ):
                x, y, z = (
                    float(robot_obs["ee.x"]),
                    float(robot_obs["ee.y"]),
                    float(robot_obs["ee.z"]),
                )
                wx, wy, wz = (
                    float(robot_obs["ee.wx"]),
                    float(robot_obs["ee.wy"]),
                    float(robot_obs["ee.wz"]),
                )
            else:
                action_like = {
                    k: float(robot_obs[k]) for k in self.action_keys if k in robot_obs
                }
                ee_obs = self.joints_to_ee((action_like, robot_obs))
                x, y, z = (
                    float(ee_obs["ee.x"]),
                    float(ee_obs["ee.y"]),
                    float(ee_obs["ee.z"]),
                )
                wx, wy, wz = (
                    float(ee_obs["ee.wx"]),
                    float(ee_obs["ee.wy"]),
                    float(ee_obs["ee.wz"]),
                )
            pos = np.array([x, y, z], dtype=np.float64)
            rvec = np.array([wx, wy, wz], dtype=np.float64)
            rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            normal = extract_forward_from_rotmat(rot, self.internal.ee_forward_axis)
            return EEPoseState(pos=pos, rvec=rvec, rot=rot, normal=normal), ""
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def send_ee_target(
        self,
        target: EEPoseTarget,
        robot_obs: Optional[dict[str, Any]] = None,
        execute: bool = True,
    ) -> dict[str, Any]:
        if robot_obs is None:
            robot_obs = self.robot.get_observation()
        ik_mode = str(self.internal.ik_mode).lower().strip()
        if ik_mode not in {"soft", "hard", "hard_bisection"}:
            ik_mode = "hard_bisection"

        if ik_mode == "soft":
            ok, joints_act, diag = self._try_solve_ee_target_soft(target, robot_obs)
            if ok:
                if execute:
                    self.robot.send_action(joints_act)
                return {
                    "sent": True,
                    "source": "soft",
                    "reason": diag["reason"],
                    "ik_attempts": 1,
                    "ik_depth_used": 0,
                    "ik_selected_alpha": 1.0,
                }
            return {
                "sent": False,
                "source": "hold",
                "reason": diag.get("reason", "soft_failed_hold_current"),
                "ik_attempts": 1,
                "ik_depth_used": 0,
                "ik_selected_alpha": 0.0,
            }

        actual_pose, actual_err = self.get_actual_ee_state(robot_obs)
        if actual_pose is None:
            return {
                "sent": False,
                "source": "hold",
                "reason": f"actual_pose_unavailable:{actual_err}",
                "ik_attempts": 0,
                "ik_depth_used": 0,
                "ik_selected_alpha": 0.0,
            }
        attempts = 0
        ok, joints_act, diag = self._try_solve_ee_target_hard(target, robot_obs)
        attempts += 1
        if ok:
            if execute:
                self.robot.send_action(joints_act)
            return {
                "sent": True,
                "source": "direct",
                "reason": diag["reason"],
                "ik_attempts": attempts,
                "ik_depth_used": 0,
                "ik_selected_alpha": 1.0,
                "ik_residual_pos": diag.get("ik_residual_pos"),
                "ik_residual_rot": diag.get("ik_residual_rot"),
            }

        if ik_mode == "hard":
            return {
                "sent": False,
                "source": "hold",
                "reason": diag.get("reason", "hard_infeasible_hold_current"),
                "ik_attempts": attempts,
                "ik_depth_used": 0,
                "ik_selected_alpha": 0.0,
                "ik_residual_pos": diag.get("ik_residual_pos"),
                "ik_residual_rot": diag.get("ik_residual_rot"),
            }

        target_rot, _ = cv2.Rodrigues(
            np.array([target.wx, target.wy, target.wz], dtype=np.float64).reshape(3, 1)
        )
        right_pose = EEPoseState(
            pos=target.pos.copy(),
            rvec=np.array([target.wx, target.wy, target.wz], dtype=np.float64),
            rot=target_rot,
            normal=target.normal.copy(),
        )
        right_alpha = 1.0
        max_depth = max(0, int(self.internal.ik_bisection_max_depth))
        for depth in range(1, max_depth + 1):
            mid_pose = self._interpolate_pose(actual_pose, right_pose, 0.5)
            mid_alpha = right_alpha * 0.5
            mid_target = self._pose_to_target(mid_pose, target)
            ok, joints_act, diag = self._try_solve_ee_target_hard(mid_target, robot_obs)
            attempts += 1
            if ok:
                if execute:
                    self.robot.send_action(joints_act)
                return {
                    "sent": True,
                    "source": "bisection",
                    "reason": diag["reason"],
                    "ik_attempts": attempts,
                    "ik_depth_used": depth,
                    "ik_selected_alpha": mid_alpha,
                    "ik_residual_pos": diag.get("ik_residual_pos"),
                    "ik_residual_rot": diag.get("ik_residual_rot"),
                }
            right_pose = mid_pose
            right_alpha = mid_alpha

        return {
            "sent": False,
            "source": "hold",
            "reason": "hard_infeasible_hold_current",
            "ik_attempts": attempts,
            "ik_depth_used": max_depth,
            "ik_selected_alpha": 0.0,
        }


class AdvancedScreenTargetModel:
    def __init__(self, body_config, task_config, internal: NewInternalParams):
        self.body_config = body_config
        self.task_config = task_config
        self.internal = internal
        self.R_ik_cam = np.array(internal.cam_to_ik_rot, dtype=np.float64)
        self.t_ik_cam = np.array(
            task_config.geometry.camera_origin_in_ik_m, dtype=np.float64
        )
        self.pan_axis_ik = np.array(internal.pan_axis_ik, dtype=np.float64)
        self.prev_normal: Optional[np.ndarray] = None

    def cam_to_ik_point(self, p_cam: np.ndarray) -> np.ndarray:
        return self.R_ik_cam @ p_cam + self.t_ik_cam

    def solve(self, eye_ik: np.ndarray) -> EEPoseTarget:
        degraded, reason = False, ""
        desired_d = float(self.task_config.desired_eye_distance_m)
        eye, pan = eye_ik, self.pan_axis_ik
        z_offset = float(self.internal.target_plane_below_eye_m)
        pan_vec_xy = np.array([pan[0] - eye[0], pan[1] - eye[1], 0.0], dtype=np.float64)
        pan_dist_xy = float(np.linalg.norm(pan_vec_xy))
        if pan_dist_xy < 1e-8:
            dir_to_pan = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            d_along = 0.0
            degraded, reason = True, "degenerate_eye_pan_line"
        else:
            dir_to_pan = pan_vec_xy / pan_dist_xy
            # 3D distance constraint with fixed vertical drop of 0.1m
            if desired_d <= z_offset:
                d_xy_target = 0.0
                degraded, reason = True, "desired_distance_too_small_for_z_offset"
            else:
                d_xy_target = math.sqrt(
                    max(0.0, desired_d * desired_d - z_offset * z_offset)
                )
            d_along = min(d_xy_target, pan_dist_xy)
            if d_along < d_xy_target:
                degraded, reason = True, "distance_clamped_by_between_constraint"
        target_pos = eye + d_along * dir_to_pan
        target_pos[2] = eye[2] - z_offset

        z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        horiz_to_eye = np.array(
            [eye[0] - target_pos[0], eye[1] - target_pos[1], 0.0], dtype=np.float64
        )
        if np.linalg.norm(horiz_to_eye) < 1e-8:
            horiz_to_eye = -dir_to_pan
        h = normalize(horiz_to_eye)
        tilt = math.radians(self.internal.screen_tilt_deg)
        n_up = normalize(math.cos(tilt) * h + math.sin(tilt) * z_world)
        n_down = normalize(math.cos(tilt) * h - math.sin(tilt) * z_world)
        if self.prev_normal is None:
            n_goal = n_up
        else:
            n_goal = (
                n_up
                if float(np.dot(n_up, self.prev_normal))
                >= float(np.dot(n_down, self.prev_normal))
                else n_down
            )
        to_eye = normalize(eye - target_pos)
        if float(np.dot(n_goal, to_eye)) < 0:
            n_goal = -n_goal
        if self.prev_normal is not None:
            alpha = clamp(self.internal.orientation_alpha, 0.0, 1.0)
            n_goal = normalize((1.0 - alpha) * self.prev_normal + alpha * n_goal)
        self.prev_normal = n_goal

        dist_err = abs(float(np.linalg.norm(eye - target_pos)) - desired_d)
        tilt_deg = math.degrees(
            math.atan2(
                abs(float(n_goal[2])), max(float(np.linalg.norm(n_goal[:2])), 1e-8)
            )
        )
        tilt_err = abs(tilt_deg - self.internal.screen_tilt_deg)
        if dist_err > self.internal.distance_tolerance_m:
            degraded, reason = True, reason or "distance_out_of_tolerance"
        if tilt_err > self.internal.tilt_tolerance_deg:
            degraded, reason = True, reason or "tilt_out_of_tolerance"

        rot = build_rotmat_with_forward_axis(n_goal, self.internal.ee_forward_axis)
        rvec, _ = cv2.Rodrigues(rot.astype(np.float64))
        wx, wy, wz = float(rvec[0]), float(rvec[1]), float(rvec[2])
        return EEPoseTarget(
            target_pos, n_goal, wx, wy, wz, dist_err, tilt_err, degraded, reason
        )


class FaceTrackNew:
    def __init__(
        self,
        task_config,
        body_config,
        camera_config,
        internal_config: Optional[NewInternalParams] = None,
    ):
        self.task_config = task_config
        self.body_config = body_config
        self.camera_config = camera_config
        self.internal_config = internal_config or NewInternalParams()
        self.internal_config.ik_mode = task_config.ik_mode
        self.internal_config.ik_bisection_max_depth = int(
            task_config.ik_bisection_max_depth
        )
        self.internal_config.ik_pos_tol_m = float(task_config.ik_pos_tol_m)
        self.internal_config.ik_rot_tol_rad = float(task_config.ik_rot_tol_rad)
        self.internal_config.urdf_path = body_config.urdf_path

        self.arm = SO101AdvancedController(
            body_config.port, body_config.id, self.internal_config
        )
        self.tracker = FaceTracker(
            task_config.model_path, task_config.min_detection_confidence
        )
        self.model: Optional[AdvancedScreenTargetModel] = None
        self.face_x_s, self.face_y_s, self.face_z_s = (
            self.internal_config.face_x_s,
            self.internal_config.face_y_s,
            self.internal_config.face_z_s,
        )
        self.session = TrackingSession(task_config)
        self.last_control_time = 0.0
        self.connected = False
        self.last_target: Optional[EEPoseTarget] = None
        self.last_actual_screen_pos_ik: Optional[np.ndarray] = None
        self.last_actual_screen_normal_ik: Optional[np.ndarray] = None
        self.last_actual_error: str = ""
        self.last_ik_diag: dict[str, Any] = {"ik_status": "idle"}
        self._outer_bias_pos = np.zeros(3, dtype=np.float64)
        self._cmd_prev_pos: Optional[np.ndarray] = None
        self._cmd_prev_rot: Optional[np.ndarray] = None

    def _reset_control_helpers(self):
        self._outer_bias_pos[:] = 0.0
        self._cmd_prev_pos = None
        self._cmd_prev_rot = None

    def connect(self):
        self.arm.connect()
        self._reset_control_helpers()
        self.model = AdvancedScreenTargetModel(
            self.body_config, self.task_config, self.internal_config
        )
        self.session.begin_after_connect()
        self.connected = True

    def disconnect(self):
        try:
            self.tracker.close()
        finally:
            self.arm.disconnect()
            self.connected = False

    def restart_tracking(self):
        self.session.restart()
        self._reset_control_helpers()

    def process_frame(
        self, frame_bgr, timestamp_ms: Optional[int] = None, execute: bool = True
    ) -> dict[str, Any]:
        if not self.connected or self.model is None:
            raise RuntimeError("Tracker not connected.")
        now = time.time()
        self.session.update_timeout(now)
        if timestamp_ms is None:
            timestamp_ms = int(now * 1000)
        robot_obs = self.arm.get_observation()
        actual_state, actual_st_err = self.arm.get_actual_ee_state(robot_obs)
        if actual_state is not None:
            self.last_actual_screen_pos_ik = actual_state.pos
            self.last_actual_screen_normal_ik = actual_state.normal
            self.last_actual_error = ""
        else:
            self.last_actual_error = actual_st_err

        result: dict[str, Any] = {
            "ok": False,
            "finished": False,
            "has_face": False,
            "screen_target_pos_ik": self.last_target.pos
            if self.last_target is not None
            else None,
            "screen_target_normal_ik": self.last_target.normal
            if self.last_target is not None
            else None,
            "screen_actual_pos_ik": self.last_actual_screen_pos_ik,
            "screen_actual_normal_ik": self.last_actual_screen_normal_ik,
            "screen_actual_error": self.last_actual_error,
            "ik_diag": self.last_ik_diag,
            "overlay": {},
        }

        if self.session.timed_out_hold_last():
            result["ok"], result["finished"], result["overlay"] = (
                True,
                True,
                {"status": "HOLD_LAST"},
            )
            return result

        face = self.tracker.detect_largest_face(frame_bgr, timestamp_ms)
        if face is None:
            if self.internal_config.outer_loop_reset_on_no_face:
                self._reset_control_helpers()
            if now - self.session.last_face_time > self.task_config.lost_timeout:
                if execute:
                    self.arm.go_home()
                result["overlay"]["status"] = "LOST -> HOME"
            else:
                result["overlay"]["status"] = "NO FACE"
            result["ok"] = True
            return result

        self.session.mark_face_seen(now)
        x, y, bw, bh = face["bbox"]
        u, v = face["center"]
        face_xyz = estimate_face_3d_from_bbox(
            self.camera_config,
            self.task_config.geometry.face_width_m,
            u,
            v,
            face["width_px"],
        )
        self.face_x_s = ema(
            self.face_x_s, float(face_xyz[0]), self.task_config.alpha_xyz
        )
        self.face_y_s = ema(
            self.face_y_s, float(face_xyz[1]), self.task_config.alpha_xyz
        )
        self.face_z_s = ema(
            self.face_z_s, float(face_xyz[2]), self.task_config.alpha_xyz
        )
        face_cam = np.array(
            [self.face_x_s, self.face_y_s, self.face_z_s], dtype=np.float64
        )
        face_ik = self.model.cam_to_ik_point(face_cam)
        target = self.model.solve(face_ik)
        self.last_target = target

        if now - self.last_control_time >= 1.0 / self.task_config.control_hz:
            ic = self.internal_config
            if not ic.outer_loop_pos_enable:
                self._outer_bias_pos[:] = 0.0
            elif actual_state is not None:
                dt = (
                    (now - self.last_control_time)
                    if self.last_control_time > 0
                    else 1.0 / self.task_config.control_hz
                )
                e_pos = target.pos - actual_state.pos
                self._outer_bias_pos += ic.outer_loop_ki_pos * e_pos * dt
                bn = float(np.linalg.norm(self._outer_bias_pos))
                if bn > ic.outer_loop_bias_max_m:
                    self._outer_bias_pos *= ic.outer_loop_bias_max_m / max(bn, 1e-12)

            pos_biased = target.pos + self._outer_bias_pos
            vision_rot, _ = cv2.Rodrigues(
                np.array([target.wx, target.wy, target.wz], dtype=np.float64).reshape(
                    3, 1
                )
            )

            prev_p = self._cmd_prev_pos
            prev_r = self._cmd_prev_rot
            if prev_p is None and actual_state is not None:
                prev_p = actual_state.pos.copy()
                prev_r = actual_state.rot.copy()
            elif prev_p is None:
                prev_p = pos_biased.copy()
                prev_r = vision_rot.copy()

            delta = pos_biased - prev_p
            dist = float(np.linalg.norm(delta))
            ms = ic.max_ee_step_m
            if ms > 0.0 and dist > ms:
                pos_cmd = prev_p + delta * (ms / dist)
            else:
                pos_cmd = pos_biased.copy()

            oa = clamp(ic.ee_ori_slerp_alpha, 0.0, 1.0)
            if oa >= 1.0 - 1e-9:
                rot_cmd = vision_rot
            else:
                rot_cmd = slerp_rotmat(prev_r, vision_rot, oa)

            rvec, _ = cv2.Rodrigues(rot_cmd)
            wx_c, wy_c, wz_c = float(rvec[0, 0]), float(rvec[1, 0]), float(rvec[2, 0])
            normal_cmd = extract_forward_from_rotmat(rot_cmd, ic.ee_forward_axis)

            cmd_target = EEPoseTarget(
                pos_cmd,
                normal_cmd,
                wx_c,
                wy_c,
                wz_c,
                target.distance_error_m,
                target.tilt_error_deg,
                target.degraded,
                target.degraded_reason,
            )

            ik_result = self.arm.send_ee_target(
                cmd_target, robot_obs=robot_obs, execute=execute
            )
            if ik_result.get("sent"):
                self._cmd_prev_pos = pos_cmd.copy()
                self._cmd_prev_rot = rot_cmd.copy()

            self.last_ik_diag = {
                "ik_status": f"{ik_result.get('source', 'hold')}_{'ok' if ik_result.get('sent', False) else 'fail'}",
                "ik_attempts": ik_result.get("ik_attempts", 0),
                "ik_depth_used": ik_result.get("ik_depth_used", 0),
                "ik_selected_alpha": ik_result.get("ik_selected_alpha", 0.0),
                "ik_residual_pos": ik_result.get("ik_residual_pos"),
                "ik_residual_rot": ik_result.get("ik_residual_rot"),
                "ik_reason": ik_result.get("reason", ""),
                "ik_mode": self.internal_config.ik_mode,
                "outer_bias_norm_m": float(np.linalg.norm(self._outer_bias_pos)),
                "ee_cmd_pos": pos_cmd.copy(),
            }
            self.last_control_time = now

        result.update(
            {
                "ok": True,
                "has_face": True,
                "face_xyz_ik": face_ik,
                "screen_target_pos_ik": target.pos,
                "screen_target_normal_ik": target.normal,
                "overlay": {
                    "bbox": (x, y, bw, bh),
                    "center": (u, v),
                    "distance_error_m": target.distance_error_m,
                    "tilt_error_deg": target.tilt_error_deg,
                    "degraded": target.degraded,
                    "degraded_reason": target.degraded_reason,
                    "status": "FOLLOW",
                    "ik_status": self.last_ik_diag.get("ik_status", "idle"),
                    "ik_reason": self.last_ik_diag.get("ik_reason", ""),
                    "ik_attempts": self.last_ik_diag.get("ik_attempts", 0),
                    "ik_depth_used": self.last_ik_diag.get("ik_depth_used", 0),
                    "ik_selected_alpha": self.last_ik_diag.get(
                        "ik_selected_alpha", 0.0
                    ),
                    "ik_residual_pos": self.last_ik_diag.get("ik_residual_pos"),
                    "ik_residual_rot": self.last_ik_diag.get("ik_residual_rot"),
                },
            }
        )
        return result

    def draw_debug(self, frame_bgr, result: dict[str, Any]):
        display = frame_bgr.copy()
        overlay = result.get("overlay", {})
        if result.get("has_face", False):
            x, y, bw, bh = overlay["bbox"]
            u, v = overlay["center"]
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(display, (int(u), int(v)), 4, (0, 0, 255), -1)
            face_ik = result.get("face_xyz_ik")
            if face_ik is not None:
                cv2.putText(
                    display,
                    f"face_ik=({face_ik[0]:+.3f},{face_ik[1]:+.3f},{face_ik[2]:+.3f})",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                display,
                f"dist_err={overlay['distance_error_m']:.3f}m",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"tilt_err={overlay['tilt_error_deg']:.2f}deg",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"degraded={overlay['degraded']} {overlay['degraded_reason'] or ''}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"ik={overlay.get('ik_status', 'idle')} depth={overlay.get('ik_depth_used', 0)} alpha={overlay.get('ik_selected_alpha', 0.0):.3f}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                2,
            )
            cv2.putText(
                display,
                f"ik_reason={overlay.get('ik_reason', '')}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                2,
            )
            tp = result.get("screen_target_pos_ik")
            tn = result.get("screen_target_normal_ik")
            ap = result.get("screen_actual_pos_ik")
            an = result.get("screen_actual_normal_ik")
            if tp is not None:
                cv2.putText(
                    display,
                    f"screen_target_pos_ik=({tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f})",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 0),
                    2,
                )
            if tn is not None:
                cv2.putText(
                    display,
                    f"screen_target_normal_ik=({tn[0]:+.3f},{tn[1]:+.3f},{tn[2]:+.3f})",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 0),
                    2,
                )
            if ap is not None:
                cv2.putText(
                    display,
                    f"screen_actual_pos_ik=({ap[0]:+.3f},{ap[1]:+.3f},{ap[2]:+.3f})",
                    (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 200, 0),
                    2,
                )
            if an is not None:
                cv2.putText(
                    display,
                    f"screen_actual_normal_ik=({an[0]:+.3f},{an[1]:+.3f},{an[2]:+.3f})",
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 200, 0),
                    2,
                )
        else:
            cv2.putText(
                display,
                overlay.get("status", "NO FACE"),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        return display

    def run_forever(
        self, return_on_finish: bool = True, execute: bool = True
    ) -> dict[str, Any] | None:
        def on_key(key: int) -> None:
            if key == ord("h"):
                self._reset_control_helpers()
                self.arm.go_home()
            elif key == ord("t"):
                self.restart_tracking()

        return run_face_tracking_loop(
            self,
            self.camera_config,
            window_title="Face Tracking New",
            return_on_finish=return_on_finish,
            execute=execute,
            on_key=on_key,
        )
