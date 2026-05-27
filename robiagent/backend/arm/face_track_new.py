from __future__ import annotations

import cv2
import math
import time
import json
import traceback
import logging
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


def build_rotmat_with_forward_axis_keep_current_roll(
    forward_ik: np.ndarray,
    axis_label: str,
    current_rot: np.ndarray,
) -> np.ndarray:
    """
    Build a full EE rotation whose selected forward axis matches forward_ik,
    while preserving the current roll as much as possible.

    This avoids arbitrary world-up roll completion.
    """
    f = normalize(forward_ik)
    axis_idx, sign = parse_axis_label(axis_label)
    primary = sign * f

    def project_ref(ref: np.ndarray, normal: np.ndarray) -> np.ndarray:
        v = ref - normal * float(np.dot(ref, normal))
        if np.linalg.norm(v) >= 1e-8:
            return normalize(v)

        # fallback only when current ref is degenerate
        for alt in (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ):
            v = alt - normal * float(np.dot(alt, normal))
            if np.linalg.norm(v) >= 1e-8:
                return normalize(v)

        raise RuntimeError("Cannot build rotation: degenerate forward/ref vectors")

    if axis_idx == 0:
        # EE ±X is screen normal. Preserve current Y as roll reference.
        x_axis = primary
        y_axis = project_ref(current_rot[:, 1], x_axis)
        z_axis = normalize(np.cross(x_axis, y_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))
        return np.column_stack([x_axis, y_axis, z_axis])

    if axis_idx == 1:
        # EE ±Y is screen normal. Preserve current Z as roll reference.
        y_axis = primary
        z_axis = project_ref(current_rot[:, 2], y_axis)
        x_axis = normalize(np.cross(y_axis, z_axis))
        z_axis = normalize(np.cross(x_axis, y_axis))
        return np.column_stack([x_axis, y_axis, z_axis])

    # EE ±Z is screen normal. Preserve current X as roll reference.
    z_axis = primary
    x_axis = project_ref(current_rot[:, 0], z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def build_rotmat_with_forward_axis(
    forward_ik: np.ndarray, axis_label: str
) -> np.ndarray:
    """
    Fallback full-rotation builder.

    Normal control path should prefer build_rotmat_with_forward_axis_keep_current_roll()
    because the task mainly cares about screen normal, not arbitrary roll.

    This function is only used when no current EE rotation is available, or as a
    generic fallback. It uses world +Y as the roll reference because this was
    empirically more stable than world +Z for the screen-normal task.
    """
    f = normalize(forward_ik)

    # Experiment A: always use world +Y as roll reference.
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Degenerate only if forward is almost parallel to +Y/-Y.
    if abs(float(np.dot(f, up))) > 0.98:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    t = normalize(np.cross(up, f))
    if np.linalg.norm(t) < 1e-8:
        # Last-resort fallback; should almost never happen.
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
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


def normal_angle_deg(n_a: np.ndarray, n_b: np.ndarray) -> float:
    dot = clamp(float(np.dot(normalize(n_a), normalize(n_b))), -1.0, 1.0)
    return math.degrees(math.acos(dot))


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
    urdf_path: str = ""

    # EMA state seeds (camera frame), same defaults as face_track.InternalParams.
    face_x_s: float = -0.05
    face_y_s: float = 0.0
    face_z_s: float = 0.8

    distance_tolerance_m: float = 0.03
    screen_tilt_deg: float = 30.0
    tilt_tolerance_deg: float = 5.0
    ik_target_frame_name: str = "gripper_frame_link"
    # orientation_alpha: float = 1.0
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
    target_plane_below_eye_m: float = 0.15
    ik_mode: str = "hard_bisection"
    ik_bisection_max_depth: int = 5
    # For ik_mode == "hard_binary_search".
    # Candidate alpha values are 0, step, 2*step, ..., 1.0.
    # Default 0.01 means grid: 0, 0.01, 0.02, ..., 0.99, 1.0.
    ik_binary_search_alpha_step: float = 0.01
    ik_pos_tol_m: float = 0.02
    ik_rot_tol_rad: float = 0.10

    # IK/control debug.
    ik_debug_enabled: bool = True
    ik_debug_print: bool = False
    ik_debug_jsonl_path: str = "ik_debug_log.jsonl"

    # Face stationary hold / anchor gate.
    face_hold_enabled: bool = True
    face_hold_release_m: float = 0.025
    face_hold_candidate_m: float = 0.015
    face_hold_stable_s: float = 0.10

    # Command target slew limiter.
    target_slew_enabled: bool = True
    target_pos_deadband_m: float = 0.004  # 4 mm
    target_normal_deadband_deg: float = 0.5  # 0.5 deg
    target_max_pos_step_m: float = 0.001  # 1 cm / control tick
    target_max_normal_step_deg: float = 1.0  # 1 deg / control tick

    # Reject IK solutions that jump to a far-away joint-space branch.
    # Units are robot joint action units. With use_degrees=True, these are degrees.
    ik_joint_delta_gate_enabled: bool = True
    ik_max_joint_delta_deg: float = 8.0
    ik_joint_delta_gate_keys: tuple[str, ...] = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
    )


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
        try:
            self.robot.connect(calibrate=False)
        except Exception as e:
            logging.info(f"{e}")
            logging.info(f"{traceback.format_exc()}")

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

    @staticmethod
    def _selected_target_fields(target: EEPoseTarget) -> dict[str, Any]:
        return {
            "ik_selected_target_pos": target.pos.copy(),
            "ik_selected_target_normal": target.normal.copy(),
            "ik_selected_target_wxyz": (
                float(target.wx),
                float(target.wy),
                float(target.wz),
            ),
        }

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

    @staticmethod
    def _target_debug_fields(target: EEPoseTarget) -> dict[str, Any]:
        return {
            "ik_selected_target_pos": [
                float(target.pos[0]),
                float(target.pos[1]),
                float(target.pos[2]),
            ],
            "ik_selected_target_normal": [
                float(target.normal[0]),
                float(target.normal[1]),
                float(target.normal[2]),
            ],
            "ik_selected_target_rvec": [
                float(target.wx),
                float(target.wy),
                float(target.wz),
            ],
        }

    def _joints_debug_fields(
        self,
        joints_act: Optional[dict[str, float]],
        robot_obs: dict[str, Any],
    ) -> dict[str, Any]:
        if joints_act is None:
            return {
                "ik_command_joints": None,
                "ik_current_joints": None,
                "ik_joint_delta": None,
                "ik_joint_delta_abs_max": None,
                "ik_joint_delta_abs_max_key": "",
            }

        command: dict[str, float] = {}
        current: dict[str, float] = {}
        delta: dict[str, float] = {}

        for k in self.action_keys:
            if k not in joints_act:
                continue

            cmd = float(joints_act[k])
            command[k] = cmd

            if k in robot_obs:
                cur = float(robot_obs[k])
                current[k] = cur
                delta[k] = cmd - cur

        if delta:
            max_key = max(delta.keys(), key=lambda key: abs(delta[key]))
            max_abs = abs(delta[max_key])
        else:
            max_key = ""
            max_abs = None

        return {
            "ik_command_joints": command,
            "ik_current_joints": current,
            "ik_joint_delta": delta,
            "ik_joint_delta_abs_max": max_abs,
            "ik_joint_delta_abs_max_key": max_key,
        }

    @staticmethod
    def _trace_entry(
        *,
        depth: int,
        alpha: float,
        ok: bool,
        diag: dict[str, Any],
        mode: str = "",
        iter_idx: Optional[int] = None,
    ) -> dict[str, Any]:
        entry = {
            "depth": int(depth),
            "alpha": float(alpha),
            "ok": bool(ok),
            "reason": diag.get("reason", ""),
            "ik_residual_pos": diag.get("ik_residual_pos"),
            "ik_residual_rot": diag.get("ik_residual_rot"),
            "ik_joint_delta_abs_max": diag.get("ik_joint_delta_abs_max"),
            "ik_joint_delta_abs_max_key": diag.get("ik_joint_delta_abs_max_key", ""),
        }
        if mode:
            entry["mode"] = mode
        if iter_idx is not None:
            entry["iter"] = int(iter_idx)
        return entry

    def _finalize_joints_action(self, joints_act: dict[str, float]) -> dict[str, float]:
        if "wrist_roll.pos" in joints_act and "wrist_roll.pos" in self.home:
            joints_act["wrist_roll.pos"] = float(self.home["wrist_roll.pos"])
        if "gripper.pos" in joints_act and "gripper.pos" in self.home:
            joints_act["gripper.pos"] = float(self.home["gripper.pos"])
        return joints_act

    def _joint_delta_gate_diag(
        self,
        joints_act: dict[str, float],
        robot_obs: dict[str, Any],
    ) -> dict[str, Any]:
        enabled = bool(getattr(self.internal, "ik_joint_delta_gate_enabled", True))
        max_allowed = float(getattr(self.internal, "ik_max_joint_delta_deg", 8.0))
        gate_keys = tuple(
            getattr(
                self.internal,
                "ik_joint_delta_gate_keys",
                (
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                ),
            )
        )

        deltas: dict[str, float] = {}

        for key in gate_keys:
            if key not in joints_act or key not in robot_obs:
                continue
            deltas[key] = float(joints_act[key]) - float(robot_obs[key])

        if deltas:
            max_key = max(deltas.keys(), key=lambda k: abs(deltas[k]))
            max_abs = abs(deltas[max_key])
        else:
            max_key = ""
            max_abs = 0.0

        passed = (not enabled) or (max_abs <= max_allowed)

        return {
            "ik_joint_delta_gate_enabled": enabled,
            "ik_max_joint_delta_deg": max_allowed,
            "ik_joint_delta_gate_keys": list(gate_keys),
            "ik_joint_delta": deltas,
            "ik_joint_delta_abs_max": max_abs,
            "ik_joint_delta_abs_max_key": max_key,
            "ik_joint_delta_pass": passed,
        }

    def _rvec_to_ee_state(self, pos: np.ndarray, rvec: np.ndarray) -> EEPoseState:
        rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        normal = extract_forward_from_rotmat(rot, self.internal.ee_forward_axis)
        return EEPoseState(
            pos=pos.astype(np.float64),
            rvec=rvec.reshape(3).astype(np.float64),
            rot=rot,
            normal=normal,
        )

    def _ee_state_from_joint_fk(
        self, robot_obs: dict[str, Any]
    ) -> tuple[Optional[EEPoseState], str]:
        if self.joints_to_ee is None:
            return None, "joints_to_ee_not_initialized"
        try:
            action_like = {
                k: float(robot_obs[k]) for k in self.action_keys if k in robot_obs
            }
            ee_obs = self.joints_to_ee((action_like, robot_obs))
            pos = np.array(
                [
                    float(ee_obs["ee.x"]),
                    float(ee_obs["ee.y"]),
                    float(ee_obs["ee.z"]),
                ],
                dtype=np.float64,
            )
            rvec = np.array(
                [
                    float(ee_obs["ee.wx"]),
                    float(ee_obs["ee.wy"]),
                    float(ee_obs["ee.wz"]),
                ],
                dtype=np.float64,
            )
            return self._rvec_to_ee_state(pos, rvec), ""
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

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
        joint_gate_diag = self._joint_delta_gate_diag(joints_act, robot_obs)
        joint_dbg = self._joints_debug_fields(joints_act, robot_obs)

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

            pos_pass = pos_err <= self.internal.ik_pos_tol_m
            rot_pass = rot_err <= self.internal.ik_rot_tol_rad
            joint_pass = bool(joint_gate_diag["ik_joint_delta_pass"])

            feasible = pos_pass and rot_pass and joint_pass

            if feasible:
                reason = "ok"
            elif not joint_pass:
                reason = "joint_delta_too_large"
            else:
                reason = "hard_residual_too_large"

            diag = {
                "reason": reason,
                "ik_residual_pos": pos_err,
                "ik_residual_rot": rot_err,
                "ik_pos_tol_m": float(self.internal.ik_pos_tol_m),
                "ik_rot_tol_rad": float(self.internal.ik_rot_tol_rad),
                "ik_pos_pass": bool(pos_pass),
                "ik_rot_pass": bool(rot_pass),
                **joint_gate_diag,
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
        return True, joints_act, {
            "reason": "soft_ok",
            **self._joints_debug_fields(joints_act, robot_obs),
        }

    def get_actual_ee_state(
        self, robot_obs: Optional[dict[str, Any]] = None
    ) -> tuple[Optional[EEPoseState], str]:
        """Actual EE from joint FK (same path as IK hard validation)."""
        if self.robot is None:
            return None, "robot_not_connected"
        if robot_obs is None:
            robot_obs = self.robot.get_observation()
        return self._ee_state_from_joint_fk(robot_obs)

    def send_ee_target(
        self,
        target: EEPoseTarget,
        robot_obs: Optional[dict[str, Any]] = None,
        execute: bool = True,
    ) -> dict[str, Any]:
        if robot_obs is None:
            robot_obs = self.robot.get_observation()
        ik_mode = str(self.internal.ik_mode).lower().strip()
        if ik_mode not in {"soft", "hard", "hard_bisection", "hard_binary_search"}:
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
        ik_trace: list[dict[str, Any]] = []

        ok, joints_act, diag = self._try_solve_ee_target_hard(target, robot_obs)
        attempts += 1
        ik_trace.append(
            self._trace_entry(depth=0, alpha=1.0, ok=ok, diag=diag, mode="direct")
        )

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
                "ik_trace": ik_trace,
                "ik_joint_delta_abs_max": diag.get("ik_joint_delta_abs_max"),
                "ik_joint_delta_abs_max_key": diag.get("ik_joint_delta_abs_max_key"),
                "ik_joint_delta": diag.get("ik_joint_delta"),
                **self._target_debug_fields(target),
                **self._joints_debug_fields(joints_act, robot_obs),
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

        if ik_mode == "hard_binary_search":
            # Search the largest feasible alpha on a discrete grid:
            # 0, step, 2*step, ..., 1.0.
            #
            # We already tried alpha=1.0 above and it failed, so this branch
            # binary-searches between alpha=0.0, treated as feasible hold/current,
            # and alpha=1.0, known infeasible in this frame.
            best_target: Optional[EEPoseTarget] = None
            alpha_step = float(
                getattr(self.internal, "ik_binary_search_alpha_step", 0.01)
            )
            if not math.isfinite(alpha_step) or alpha_step <= 0.0:
                alpha_step = 0.01
            alpha_step = clamp(alpha_step, 1e-6, 1.0)

            max_idx = max(1, int(math.ceil(1.0 / alpha_step)))
            low_idx = 0          # alpha=0.0, feasible by definition: current/hold
            high_idx = max_idx   # alpha=1.0, already failed above

            best_joints_act: Optional[dict[str, float]] = None
            best_diag: dict[str, Any] = diag
            best_alpha = 0.0
            search_iters = 0

            last_diag = diag

            while high_idx - low_idx > 1:
                mid_idx = (low_idx + high_idx) // 2
                mid_alpha = min(1.0, mid_idx * alpha_step)

                mid_pose = self._interpolate_pose(actual_pose, right_pose, mid_alpha)
                mid_target = self._pose_to_target(mid_pose, target)

                ok, joints_act, mid_diag = self._try_solve_ee_target_hard(
                    mid_target, robot_obs
                )
                attempts += 1
                search_iters += 1
                last_diag = mid_diag

                ik_trace.append(
                    self._trace_entry(
                        depth=0,
                        alpha=mid_alpha,
                        ok=ok,
                        diag=mid_diag,
                        mode="binary_search",
                        iter_idx=search_iters,
                    )
                )

                if ok:
                    low_idx = mid_idx
                    best_alpha = mid_alpha
                    best_joints_act = joints_act
                    best_diag = mid_diag
                    best_target = mid_target
                else:
                    high_idx = mid_idx

            if best_joints_act is not None and best_alpha > 0.0:
                if execute:
                    self.robot.send_action(best_joints_act)
                return {
                    "sent": True,
                    "source": "binary_search",
                    "reason": best_diag["reason"],
                    "ik_attempts": attempts,
                    "ik_depth_used": search_iters,
                    "ik_selected_alpha": best_alpha,
                    "ik_residual_pos": best_diag.get("ik_residual_pos"),
                    "ik_residual_rot": best_diag.get("ik_residual_rot"),
                    "ik_binary_search_alpha_step": alpha_step,
                    "ik_trace": ik_trace,
                    "ik_joint_delta_abs_max": diag.get("ik_joint_delta_abs_max"),
                    "ik_joint_delta_abs_max_key": diag.get("ik_joint_delta_abs_max_key"),
                    "ik_joint_delta": diag.get("ik_joint_delta"),
                    **self._target_debug_fields(best_target),
                    **self._joints_debug_fields(best_joints_act, robot_obs),
                }

            return {
                "sent": False,
                "source": "hold",
                "reason": "hard_binary_search_no_feasible_alpha",
                "ik_attempts": attempts,
                "ik_depth_used": search_iters,
                "ik_selected_alpha": 0.0,
                "ik_residual_pos": last_diag.get("ik_residual_pos"),
                "ik_residual_rot": last_diag.get("ik_residual_rot"),
                "ik_last_reason": last_diag.get("reason", ""),
                "ik_binary_search_alpha_step": alpha_step,
                "ik_trace": ik_trace,
            }

        right_alpha = 1.0
        max_depth = max(0, int(self.internal.ik_bisection_max_depth))

        last_diag = diag
        for depth in range(1, max_depth + 1):
            mid_pose = self._interpolate_pose(actual_pose, right_pose, 0.5)
            mid_alpha = right_alpha * 0.5
            mid_target = self._pose_to_target(mid_pose, target)

            ok, joints_act, diag = self._try_solve_ee_target_hard(mid_target, robot_obs)
            attempts += 1
            last_diag = diag

            ik_trace.append(
                self._trace_entry(
                    depth=depth,
                    alpha=mid_alpha,
                    ok=ok,
                    diag=diag,
                    mode="bisection",
                )
            )

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
                    "ik_trace": ik_trace,
                    "ik_joint_delta_abs_max": diag.get("ik_joint_delta_abs_max"),
                    "ik_joint_delta_abs_max_key": diag.get("ik_joint_delta_abs_max_key"),
                    "ik_joint_delta": diag.get("ik_joint_delta"),
                    **self._target_debug_fields(mid_target),
                    **self._joints_debug_fields(joints_act, robot_obs),
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
            "ik_residual_pos": last_diag.get("ik_residual_pos"),
            "ik_residual_rot": last_diag.get("ik_residual_rot"),
            "ik_last_reason": last_diag.get("reason", ""),
            "ik_trace": ik_trace,
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
        z_offset = float(self.task_config.target_plane_below_eye_m)
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
        tilt = math.radians(self.task_config.screen_tilt_deg)
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
        # if self.prev_normal is not None:
        #     alpha = clamp(self.internal.orientation_alpha, 0.0, 1.0)
        #     n_goal = normalize((1.0 - alpha) * self.prev_normal + alpha * n_goal)
        # self.prev_normal = n_goal

        dist_err = abs(float(np.linalg.norm(eye - target_pos)) - desired_d)
        tilt_deg = math.degrees(
            math.atan2(
                abs(float(n_goal[2])), max(float(np.linalg.norm(n_goal[:2])), 1e-8)
            )
        )
        tilt_err = abs(tilt_deg - self.task_config.screen_tilt_deg)
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
        self.internal_config.ik_pos_tol_m = task_config.ik_pos_tol_m
        self.internal_config.ik_rot_tol_rad = task_config.ik_rot_tol_rad
        self.internal_config.ik_binary_search_alpha_step = task_config.ik_binary_search_alpha_step
        self.internal_config.ik_joint_delta_gate_enabled = task_config.ik_joint_delta_gate_enabled
        self.internal_config.ik_max_joint_delta_deg = task_config.ik_max_joint_delta_deg
        self.internal_config.ik_joint_delta_gate_keys = task_config.ik_joint_delta_gate_keys

        self.internal_config.face_hold_enabled = task_config.face_hold_enabled
        self.internal_config.face_hold_release_m = task_config.face_hold_release_m
        self.internal_config.face_hold_candidate_m = task_config.face_hold_candidate_m
        self.internal_config.face_hold_stable_s = task_config.face_hold_stable_s

        self.internal_config.target_slew_enabled = task_config.target_slew_enabled
        self.internal_config.target_pos_deadband_m = task_config.target_pos_deadband_m
        self.internal_config.target_normal_deadband_deg = task_config.target_normal_deadband_deg
        self.internal_config.target_max_pos_step_m = task_config.target_max_pos_step_m
        self.internal_config.target_max_normal_step_deg = task_config.target_max_normal_step_deg
        self.internal_config.ik_debug_enabled = task_config.ik_debug_enabled
        self.internal_config.ik_debug_print = task_config.ik_debug_print
        self.internal_config.ik_debug_jsonl_path = task_config.ik_debug_jsonl_path

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

        self.last_slew_target: Optional[EEPoseTarget] = None
        self.target_slew_status: str = "disabled"
        self.target_slew_pos_delta_m: float = 0.0
        self.target_slew_normal_delta_deg: float = 0.0

        # Face stationary hold state.
        self.face_hold_anchor_ik: Optional[np.ndarray] = None
        self.face_hold_candidate_ik: Optional[np.ndarray] = None
        self.face_hold_candidate_since: float = 0.0
        self.face_hold_status: str = "disabled"
        self.face_hold_dist_m: float = 0.0

        self.ik_debug_seq: int = 0
        self.last_ik_debug_snapshot: Optional[dict[str, Any]] = None

    def connect(self):
        self.arm.connect()
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

        # Reset target slew state.
        self.last_slew_target = None
        self.target_slew_status = "reset"
        self.target_slew_pos_delta_m = 0.0
        self.target_slew_normal_delta_deg = 0.0

        # Reset face stationary hold state.
        self.face_hold_anchor_ik = None
        self.face_hold_candidate_ik = None
        self.face_hold_candidate_since = 0.0
        self.face_hold_status = "reset"
        self.face_hold_dist_m = 0.0

        # Reset target-model normal smoothing / up-down branch memory.
        if self.model is not None:
            self.model.prev_normal = None

        # Optional but cleaner: next frame can immediately send a control command.
        self.last_control_time = 0.0

        # Optional: clear displayed target until next valid face frame.
        self.last_target = None

    def _slerp_unit_vector(self, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        a = normalize(np.asarray(a, dtype=np.float64).reshape(3))
        b = normalize(np.asarray(b, dtype=np.float64).reshape(3))
        alpha = clamp(float(alpha), 0.0, 1.0)

        dot = clamp(float(np.dot(a, b)), -1.0, 1.0)

        if dot > 0.9995:
            return normalize((1.0 - alpha) * a + alpha * b)

        # Very unlikely for screen normals, but handle near-opposite safely.
        if dot < -0.9995:
            alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(a, alt))) > 0.9:
                alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            ortho = normalize(alt - a * float(np.dot(a, alt)))
            theta = math.pi * alpha
            return normalize(math.cos(theta) * a + math.sin(theta) * ortho)

        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        return normalize(
            math.sin((1.0 - alpha) * theta) / sin_theta * a
            + math.sin(alpha * theta) / sin_theta * b
        )

    def _rot_from_normal_preserve_roll(
        self,
        normal: np.ndarray,
        roll_ref_rot: np.ndarray,
    ) -> np.ndarray:
        """
        Use your keep-current-roll helper if present; otherwise fall back to
        build_rotmat_with_forward_axis().
        """
        builder = globals().get("build_rotmat_with_forward_axis_keep_current_roll")
        if callable(builder):
            return builder(
                normal,
                self.internal_config.ee_forward_axis,
                roll_ref_rot,
            )
        return build_rotmat_with_forward_axis(
            normal,
            self.internal_config.ee_forward_axis,
        )

    def _target_rotmat(self, target: EEPoseTarget) -> np.ndarray:
        rvec = np.array([target.wx, target.wy, target.wz], dtype=np.float64).reshape(3, 1)
        rot, _ = cv2.Rodrigues(rvec)
        return rot

    def _copy_target_with_pose(
        self,
        template: EEPoseTarget,
        pos: np.ndarray,
        normal: np.ndarray,
        roll_ref_rot: np.ndarray,
    ) -> EEPoseTarget:
        normal = normalize(np.asarray(normal, dtype=np.float64).reshape(3))
        rot = self._rot_from_normal_preserve_roll(normal, roll_ref_rot)
        rvec, _ = cv2.Rodrigues(rot.astype(np.float64))
        wx, wy, wz = map(float, rvec.reshape(3))

        return EEPoseTarget(
            pos=np.asarray(pos, dtype=np.float64).reshape(3).copy(),
            normal=normal.copy(),
            wx=wx,
            wy=wy,
            wz=wz,
            distance_error_m=template.distance_error_m,
            tilt_error_deg=template.tilt_error_deg,
            degraded=template.degraded,
            degraded_reason=template.degraded_reason,
        )

    def _target_from_ik_result(
        self,
        template: EEPoseTarget,
        ik_result: dict[str, Any],
    ) -> Optional[EEPoseTarget]:
        pos = ik_result.get("ik_selected_target_pos")
        normal = ik_result.get("ik_selected_target_normal")
        wxyz = ik_result.get("ik_selected_target_wxyz")

        if pos is None or normal is None or wxyz is None:
            return None

        wx, wy, wz = wxyz
        return EEPoseTarget(
            pos=np.asarray(pos, dtype=np.float64).reshape(3).copy(),
            normal=normalize(np.asarray(normal, dtype=np.float64).reshape(3)),
            wx=float(wx),
            wy=float(wy),
            wz=float(wz),
            distance_error_m=template.distance_error_m,
            tilt_error_deg=template.tilt_error_deg,
            degraded=template.degraded,
            degraded_reason=template.degraded_reason,
        )

    def _slew_limit_target(
        self,
        raw_target: EEPoseTarget,
        actual_state: Optional[EEPoseState],
    ) -> EEPoseTarget:
        """
        Clamp EE target movement per control tick.

        This is the new IK/EE equivalent of max_pan_step/max_lift_step:
        - small target drift is ignored by deadband
        - large target jumps are approached gradually
        """
        internal = self.internal_config

        if not getattr(internal, "target_slew_enabled", False):
            self.last_slew_target = raw_target
            self.target_slew_status = "disabled"
            self.target_slew_pos_delta_m = 0.0
            self.target_slew_normal_delta_deg = 0.0
            return raw_target

        # Seed from current actual pose if available, otherwise from first raw target.
        if self.last_slew_target is None:
            if actual_state is not None:
                seeded = self._copy_target_with_pose(
                    raw_target,
                    actual_state.pos,
                    actual_state.normal,
                    actual_state.rot,
                )
                self.last_slew_target = seeded
                self.target_slew_status = "seed_actual"
                return seeded

            self.last_slew_target = raw_target
            self.target_slew_status = "seed_raw"
            return raw_target

        prev = self.last_slew_target
        prev_rot = self._target_rotmat(prev)

        # Position slew.
        raw_pos = np.asarray(raw_target.pos, dtype=np.float64).reshape(3)
        prev_pos = np.asarray(prev.pos, dtype=np.float64).reshape(3)
        pos_delta = raw_pos - prev_pos
        pos_dist = float(np.linalg.norm(pos_delta))
        self.target_slew_pos_delta_m = pos_dist

        pos_deadband = max(0.0, float(internal.target_pos_deadband_m))
        max_pos_step = max(1e-9, float(internal.target_max_pos_step_m))

        if pos_dist <= pos_deadband:
            new_pos = prev_pos.copy()
            pos_status = "pos_hold"
        elif pos_dist > max_pos_step:
            new_pos = prev_pos + pos_delta / pos_dist * max_pos_step
            pos_status = "pos_step"
        else:
            new_pos = raw_pos.copy()
            pos_status = "pos_raw"

        # Normal slew.
        raw_n = normalize(np.asarray(raw_target.normal, dtype=np.float64).reshape(3))
        prev_n = normalize(np.asarray(prev.normal, dtype=np.float64).reshape(3))
        normal_delta_deg = normal_angle_deg(prev_n, raw_n)
        self.target_slew_normal_delta_deg = normal_delta_deg

        normal_deadband = max(0.0, float(internal.target_normal_deadband_deg))
        max_normal_step_deg = max(1e-6, float(internal.target_max_normal_step_deg))

        if normal_delta_deg <= normal_deadband:
            new_n = prev_n.copy()
            normal_status = "normal_hold"
        elif normal_delta_deg > max_normal_step_deg:
            alpha = max_normal_step_deg / normal_delta_deg
            new_n = self._slerp_unit_vector(prev_n, raw_n, alpha)
            normal_status = "normal_step"
        else:
            new_n = raw_n.copy()
            normal_status = "normal_raw"

        limited = self._copy_target_with_pose(
            raw_target,
            new_pos,
            new_n,
            prev_rot,
        )

        self.last_slew_target = limited
        self.target_slew_status = f"{pos_status},{normal_status}"
        return limited

    def _stabilize_face_ik(self, measured_face_ik: np.ndarray, now: float) -> np.ndarray:
        """
        Optional face stationary hold.

        Behavior:
        - If measured face stays near the current anchor, return the anchor.
        - If measured face jumps/drifts away, require it to remain stable around
          a candidate for face_hold_stable_s before accepting a new anchor.
        """
        internal = self.internal_config
        measured = np.asarray(measured_face_ik, dtype=np.float64).reshape(3)

        if not getattr(internal, "face_hold_enabled", False):
            self.face_hold_status = "disabled"
            self.face_hold_dist_m = 0.0
            return measured

        release_m = max(0.0, float(internal.face_hold_release_m))
        candidate_m = max(1e-6, float(internal.face_hold_candidate_m))
        stable_s = max(0.0, float(internal.face_hold_stable_s))

        if self.face_hold_anchor_ik is None:
            self.face_hold_anchor_ik = measured.copy()
            self.face_hold_candidate_ik = None
            self.face_hold_candidate_since = 0.0
            self.face_hold_status = "anchor_init"
            self.face_hold_dist_m = 0.0
            return self.face_hold_anchor_ik.copy()

        anchor = self.face_hold_anchor_ik
        dist_to_anchor = float(np.linalg.norm(measured - anchor))
        self.face_hold_dist_m = dist_to_anchor

        # Still close enough: treat as no real face movement.
        if dist_to_anchor <= release_m:
            self.face_hold_candidate_ik = None
            self.face_hold_candidate_since = 0.0
            self.face_hold_status = "hold"
            return anchor.copy()

        # Far from anchor: maybe real movement, maybe detector drift/outlier.
        if self.face_hold_candidate_ik is None:
            self.face_hold_candidate_ik = measured.copy()
            self.face_hold_candidate_since = now
            self.face_hold_status = "candidate_new"
            return anchor.copy()

        dist_to_candidate = float(np.linalg.norm(measured - self.face_hold_candidate_ik))

        # Candidate itself is drifting too much; restart candidate timer.
        if dist_to_candidate > candidate_m:
            self.face_hold_candidate_ik = measured.copy()
            self.face_hold_candidate_since = now
            self.face_hold_status = "candidate_reset"
            return anchor.copy()

        # Candidate is stable; accept it as new anchor.
        candidate_age = now - self.face_hold_candidate_since
        if candidate_age >= stable_s:
            self.face_hold_anchor_ik = measured.copy()
            self.face_hold_candidate_ik = None
            self.face_hold_candidate_since = 0.0
            self.face_hold_status = "anchor_update"
            self.face_hold_dist_m = 0.0
            return self.face_hold_anchor_ik.copy()

        self.face_hold_status = f"candidate_wait_{candidate_age:.2f}s"
        return anchor.copy()

    def _debug_arr(self, v: Any) -> Optional[list[float]]:
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        return [float(x) for x in a.tolist()]

    def _debug_target(self, target: Optional[EEPoseTarget]) -> Optional[dict[str, Any]]:
        if target is None:
            return None
        return {
            "pos": self._debug_arr(target.pos),
            "normal": self._debug_arr(target.normal),
            "rvec": [float(target.wx), float(target.wy), float(target.wz)],
            "distance_error_m": float(target.distance_error_m),
            "tilt_error_deg": float(target.tilt_error_deg),
            "degraded": bool(target.degraded),
            "degraded_reason": target.degraded_reason,
        }

    def _debug_state(self, state: Optional[EEPoseState]) -> Optional[dict[str, Any]]:
        if state is None:
            return None
        return {
            "pos": self._debug_arr(state.pos),
            "normal": self._debug_arr(state.normal),
            "rvec": self._debug_arr(state.rvec),
        }

    def _debug_normal_err_deg(
        self, a: Optional[np.ndarray], b: Optional[np.ndarray]
    ) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(normal_angle_deg(a, b))

    def _debug_pos_err_m(
        self, a: Optional[np.ndarray], b: Optional[np.ndarray]
    ) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))

    def _json_safe(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        return obj

    def _log_ik_control_tick(
        self,
        *,
        now: float,
        robot_obs: dict[str, Any],
        actual_state: Optional[EEPoseState],
        face_ik_measured: np.ndarray,
        face_ik: np.ndarray,
        raw_target: EEPoseTarget,
        last_slew_before: Optional[EEPoseTarget],
        target_sent_to_ik: EEPoseTarget,
        ik_result: dict[str, Any],
    ) -> None:
        if not getattr(self.internal_config, "ik_debug_enabled", False):
            return

        self.ik_debug_seq += 1

        selected_pos = ik_result.get("ik_selected_target_pos")
        selected_normal = ik_result.get("ik_selected_target_normal")

        record: dict[str, Any] = {
            "seq": self.ik_debug_seq,
            "t": float(now),
            "dt_from_last_control_s": float(
                0.0 if self.last_control_time <= 0.0 else now - self.last_control_time
            ),
            "ik": {
                "mode": self.internal_config.ik_mode,
                "source": ik_result.get("source"),
                "sent": bool(ik_result.get("sent", False)),
                "reason": ik_result.get("reason", ""),
                "attempts": ik_result.get("ik_attempts"),
                "depth_used": ik_result.get("ik_depth_used"),
                "alpha": ik_result.get("ik_selected_alpha"),
                "residual_pos": ik_result.get("ik_residual_pos"),
                "residual_rot": ik_result.get("ik_residual_rot"),
                "joint_delta_abs_max": ik_result.get("ik_joint_delta_abs_max"),
                "joint_delta_abs_max_key": ik_result.get("ik_joint_delta_abs_max_key"),
                "trace": ik_result.get("ik_trace"),
            },
            "face": {
                "measured_ik": self._debug_arr(face_ik_measured),
                "used_ik": self._debug_arr(face_ik),
                "hold_status": self.face_hold_status,
                "hold_dist_m": self.face_hold_dist_m,
            },
            "slew": {
                "status": self.target_slew_status,
                "pos_delta_m": self.target_slew_pos_delta_m,
                "normal_delta_deg": self.target_slew_normal_delta_deg,
                "last_slew_before": self._debug_target(last_slew_before),
            },
            "actual": self._debug_state(actual_state),
            "raw_target": self._debug_target(raw_target),
            "target_sent_to_ik": self._debug_target(target_sent_to_ik),
            "selected_by_ik": {
                "pos": self._debug_arr(selected_pos),
                "normal": self._debug_arr(selected_normal),
                "rvec": ik_result.get("ik_selected_target_rvec"),
            },
            "errors": {
                "actual_to_target_pos_m": self._debug_pos_err_m(
                    actual_state.pos if actual_state is not None else None,
                    target_sent_to_ik.pos,
                ),
                "actual_to_target_normal_deg": self._debug_normal_err_deg(
                    actual_state.normal if actual_state is not None else None,
                    target_sent_to_ik.normal,
                ),
                "actual_to_selected_pos_m": self._debug_pos_err_m(
                    actual_state.pos if actual_state is not None else None,
                    np.asarray(selected_pos, dtype=np.float64) if selected_pos is not None else None,
                ),
                "actual_to_selected_normal_deg": self._debug_normal_err_deg(
                    actual_state.normal if actual_state is not None else None,
                    np.asarray(selected_normal, dtype=np.float64) if selected_normal is not None else None,
                ),
            },
            "joints": {
                "command": ik_result.get("ik_command_joints"),
                "current": ik_result.get("ik_current_joints"),
                "delta": ik_result.get("ik_joint_delta"),
            },
        }

        prev = self.last_ik_debug_snapshot
        if prev is not None:
            prev_alpha = prev.get("alpha")
            cur_alpha = ik_result.get("ik_selected_alpha")

            record["prev_delta"] = {
                "source_transition": f"{prev.get('source')}->{ik_result.get('source')}",
                "alpha_delta": (
                    None
                    if prev_alpha is None or cur_alpha is None
                    else float(cur_alpha) - float(prev_alpha)
                ),
                "target_sent_pos_delta_m": self._debug_pos_err_m(
                    np.asarray(prev.get("target_sent_pos"), dtype=np.float64)
                    if prev.get("target_sent_pos") is not None
                    else None,
                    target_sent_to_ik.pos,
                ),
                "target_sent_normal_delta_deg": self._debug_normal_err_deg(
                    np.asarray(prev.get("target_sent_normal"), dtype=np.float64)
                    if prev.get("target_sent_normal") is not None
                    else None,
                    target_sent_to_ik.normal,
                ),
                "actual_pos_delta_m": self._debug_pos_err_m(
                    np.asarray(prev.get("actual_pos"), dtype=np.float64)
                    if prev.get("actual_pos") is not None
                    else None,
                    actual_state.pos if actual_state is not None else None,
                ),
                "actual_normal_delta_deg": self._debug_normal_err_deg(
                    np.asarray(prev.get("actual_normal"), dtype=np.float64)
                    if prev.get("actual_normal") is not None
                    else None,
                    actual_state.normal if actual_state is not None else None,
                ),
            }

        self.last_ik_debug_snapshot = {
            "source": ik_result.get("source"),
            "alpha": ik_result.get("ik_selected_alpha"),
            "target_sent_pos": self._debug_arr(target_sent_to_ik.pos),
            "target_sent_normal": self._debug_arr(target_sent_to_ik.normal),
            "actual_pos": self._debug_arr(actual_state.pos) if actual_state is not None else None,
            "actual_normal": self._debug_arr(actual_state.normal) if actual_state is not None else None,
        }

        line = json.dumps(self._json_safe(record), ensure_ascii=False)

        path = getattr(self.internal_config, "ik_debug_jsonl_path", "ik_debug_log.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if getattr(self.internal_config, "ik_debug_print", False):
            print("[IKDBG]", line)

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
        # self.face_x_s = ema(
        #     self.face_x_s, float(face_xyz[0]), self.task_config.alpha_xyz
        # )
        # self.face_y_s = ema(
        #     self.face_y_s, float(face_xyz[1]), self.task_config.alpha_xyz
        # )
        # self.face_z_s = ema(
        #     self.face_z_s, float(face_xyz[2]), self.task_config.alpha_xyz
        # )
        # face_cam = np.array(
        #     [self.face_x_s, self.face_y_s, self.face_z_s], dtype=np.float64
        # )
        face_cam = face_xyz
        # face_ik = self.model.cam_to_ik_point(face_cam)
        # target = self.model.solve(face_ik)

        face_ik_measured = self.model.cam_to_ik_point(face_cam)
        face_ik = self._stabilize_face_ik(face_ik_measured, now)

        raw_target = self.model.solve(face_ik)

        # keep-current-roll
        if actual_state is not None:
            rot = build_rotmat_with_forward_axis_keep_current_roll(
                raw_target.normal,
                self.internal_config.ee_forward_axis,
                actual_state.rot,
            )
            rvec, _ = cv2.Rodrigues(rot.astype(np.float64))
            raw_target.wx, raw_target.wy, raw_target.wz = map(float, rvec.reshape(3))

        target = raw_target
        if (
            getattr(self.internal_config, "target_slew_enabled", False)
            and self.last_slew_target is not None
        ):
            # Between control ticks, display/hold the last commanded target.
            target = self.last_slew_target

        if now - self.last_control_time >= 1.0 / self.task_config.control_hz:
            last_slew_before = self.last_slew_target
            target = self._slew_limit_target(raw_target, actual_state)
            self.last_target = target

            ik_result = self.arm.send_ee_target(
                target, robot_obs=robot_obs, execute=execute
            )

            self._log_ik_control_tick(
                now=now,
                robot_obs=robot_obs,
                actual_state=actual_state,
                face_ik_measured=face_ik_measured,
                face_ik=face_ik,
                raw_target=raw_target,
                last_slew_before=last_slew_before,
                target_sent_to_ik=target,
                ik_result=ik_result,
            )

            selected_target = None
            if ik_result.get("sent", False):
                selected_target = self._target_from_ik_result(target, ik_result)

            if selected_target is not None:
                # Important:
                # target_slew should remember what IK actually selected/sent,
                # not the final target that may have been rejected.
                target = selected_target
                self.last_target = selected_target

                if getattr(self.internal_config, "target_slew_enabled", False):
                    self.last_slew_target = selected_target
            else:
                self.last_target = target
            # # only for DEBUG
            # if (
            #     ik_result.get("source") == "hold"
            #     and not ik_result.get("sent", False)
            # ):
            #     print("[IK HOLD_FAIL]", ik_result)
            self.last_ik_diag = {
                "ik_status": f"{ik_result.get('source', 'hold')}_{'ok' if ik_result.get('sent', False) else 'fail'}",
                "ik_attempts": ik_result.get("ik_attempts", 0),
                "ik_depth_used": ik_result.get("ik_depth_used", 0),
                "ik_selected_alpha": ik_result.get("ik_selected_alpha", 0.0),
                "ik_residual_pos": ik_result.get("ik_residual_pos"),
                "ik_residual_rot": ik_result.get("ik_residual_rot"),
                "ik_reason": ik_result.get("reason", ""),
                "ik_mode": self.internal_config.ik_mode,
                "ik_joint_delta_abs_max": ik_result.get("ik_joint_delta_abs_max"),
                "ik_joint_delta_abs_max_key": ik_result.get("ik_joint_delta_abs_max_key"),
            }
            self.last_control_time = now
        else:
            self.last_target = target

        result.update(
            {
                "ok": True,
                "has_face": True,
                "face_xyz_ik": face_ik,
                "face_xyz_ik_measured": face_ik_measured,
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
                    "face_hold_status": self.face_hold_status,
                    "face_hold_dist_m": self.face_hold_dist_m,
                    "ik_status": self.last_ik_diag.get("ik_status", "idle"),
                    "ik_reason": self.last_ik_diag.get("ik_reason", ""),
                    "ik_attempts": self.last_ik_diag.get("ik_attempts", 0),
                    "ik_depth_used": self.last_ik_diag.get("ik_depth_used", 0),
                    "ik_selected_alpha": self.last_ik_diag.get(
                        "ik_selected_alpha", 0.0
                    ),
                    "ik_residual_pos": self.last_ik_diag.get("ik_residual_pos"),
                    "ik_residual_rot": self.last_ik_diag.get("ik_residual_rot"),
                    "ik_joint_delta_abs_max": self.last_ik_diag.get("ik_joint_delta_abs_max"),
                    "ik_joint_delta_abs_max_key": self.last_ik_diag.get("ik_joint_delta_abs_max_key"),
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
            if tn is not None and an is not None:
                dot = clamp(
                    float(np.dot(normalize(tn), normalize(an))), -1.0, 1.0
                )
                normal_err_deg = math.degrees(math.acos(dot))
                cv2.putText(
                    display,
                    f"screen_normal_err={normal_err_deg:.2f}deg",
                    (20, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 200, 255),
                    2,
                )
            cv2.putText(
                display,
                f"face_hold={overlay.get('face_hold_status', '')} d={overlay.get('face_hold_dist_m', 0.0):.3f}m",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 180, 255),
                2,
            )
            cv2.putText(
                display,
                f"joint_delta_max={overlay.get('ik_joint_delta_abs_max', 0.0) or 0.0:.1f} "
                f"{overlay.get('ik_joint_delta_abs_max_key', '')}",
                (20, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 180, 255),
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
        self, return_on_finish=True, args=None, execute=True
    ) -> dict[str, Any] | None:
        def on_key(key: int) -> None:
            if key == ord("h"):
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
