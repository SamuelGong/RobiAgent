from __future__ import annotations

import time
from typing import Any


DEFAULT_JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class SimpleMove:
    """
    Move one SO arm to a target joint pose from task_config.target_pose.

    Expected YAML:

    task:
      be_there:
        target_pose:
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
        move_steps: 50      # optional
        step_sleep_s: 0.02  # optional
        hold_s: 0.5         # optional

    target_pose order:
      shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    """

    def __init__(self, task_config: Any, body_config: Any):
        self.task_config = task_config
        self.body_config = body_config

        self.robot = None
        self.action_keys: list[str] = []
        self.start_pose: dict[str, float] | None = None

    def _cfg(self, name: str, default: Any = None) -> Any:
        if isinstance(self.task_config, dict):
            return self.task_config.get(name, default)
        return getattr(self.task_config, name, default)

    def _make_robot(self):
        """
        Prefer the newer SOFollower API used by face_track_new.py.
        Fallback to old SO101Follower API if needed.
        """
        try:
            from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

            return SOFollower(
                SOFollowerRobotConfig(
                    id=self.body_config.id,
                    port=self.body_config.port,
                    use_degrees=True,
                    disable_torque_on_disconnect=False,
                )
            )
        except Exception:
            from robiagent.backend.arm.face_track import import_so101_classes

            SO101FollowerConfig, SO101Follower = import_so101_classes()
            return SO101Follower(
                SO101FollowerConfig(
                    id=self.body_config.id,
                    port=self.body_config.port,
                    disable_torque_on_disconnect=False,
                )
            )

    def connect(self):
        self.robot = self._make_robot()

        # Assume already calibrated.
        self.robot.connect(calibrate=False)

        self.action_keys = list(self.robot.action_features.keys())
        self._validate_required_joints()

        self.start_pose = self.get_pose()

    def disconnect(self):
        if self.robot is not None:
            # Keep holding torque after normal disconnect.
            self.robot.disconnect()
            self.robot = None

    def _validate_required_joints(self):
        missing = [k for k in DEFAULT_JOINT_ORDER if k not in self.action_keys]
        if missing:
            raise RuntimeError(
                f"Robot action keys missing required joints: {missing}. "
                f"Available action keys: {self.action_keys}"
            )

    def get_pose(self) -> dict[str, float]:
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        obs = self.robot.get_observation()
        return {k: float(obs[k]) for k in self.action_keys if k in obs}

    def _target_pose_to_action(self, raw_target) -> dict[str, float]:
        if self.start_pose is None:
            raise RuntimeError("start_pose is not initialized. Call connect() first.")

        # Start with current pose so any extra action keys remain unchanged.
        target_action = dict(self.start_pose)

        if isinstance(raw_target, dict):
            for key, value in raw_target.items():
                if key not in self.action_keys:
                    raise ValueError(
                        f"Unknown joint key in target_pose: {key}. "
                        f"Available action keys: {self.action_keys}"
                    )
                target_action[key] = float(value)

        else:
            values = list(raw_target)
            if len(values) != len(DEFAULT_JOINT_ORDER):
                raise ValueError(
                    f"target_pose must contain {len(DEFAULT_JOINT_ORDER)} values, "
                    f"got {len(values)}. Expected order: {DEFAULT_JOINT_ORDER}"
                )

            for key, value in zip(DEFAULT_JOINT_ORDER, values):
                target_action[key] = float(value)

        # Only send keys the robot accepts.
        return {k: float(target_action[k]) for k in self.action_keys if k in target_action}

    def send_pose(self, pose: dict[str, float]):
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        self.robot.send_action(dict(pose))

    def move_to_target(self, target_pose) -> dict[str, Any]:
        """
        Move from current joint pose to target joint pose.

        hold_s semantics:
        - hold_s < 0: hold forever until KeyboardInterrupt / external interruption
        - hold_s = 0: return immediately after reaching target
        - hold_s > 0: hold for hold_s seconds, then return

        Torque behavior:
        - This class creates the robot with disable_torque_on_disconnect=False.
        - So even after disconnect(), normal disconnect should not disable torque.
        """
        target_action = self._target_pose_to_action(target_pose)

        move_steps = max(1, int(self._cfg("move_steps", 50)))
        step_sleep_s = max(0.0, float(self._cfg("step_sleep_s", 0.02)))
        hold_s = float(self._cfg("hold_s", 0.5))

        current = self.get_pose()

        if move_steps <= 1:
            self.send_pose(target_action)
        else:
            for i in range(1, move_steps + 1):
                alpha = i / move_steps
                interp = dict(current)

                for key, target_value in target_action.items():
                    if key in current:
                        interp[key] = (
                            (1.0 - alpha) * float(current[key])
                            + alpha * float(target_value)
                        )
                    else:
                        interp[key] = float(target_value)

                self.send_pose(interp)

                if step_sleep_s > 0:
                    time.sleep(step_sleep_s)

        # Send exact final target once more after interpolation.
        self.send_pose(target_action)

        if hold_s < 0:
            print("Reached target pose. Holding forever. Press Ctrl+C to stop.")
            while True:
                # Re-send occasionally so the commanded target remains explicit.
                self.send_pose(target_action)
                time.sleep(1.0)

        if hold_s > 0:
            time.sleep(hold_s)

        final_pose = self.get_pose()

        return {
            "ok": True,
            "joint_order": DEFAULT_JOINT_ORDER,
            "target_pose": [target_action[k] for k in DEFAULT_JOINT_ORDER],
            "final_pose": {k: final_pose.get(k) for k in DEFAULT_JOINT_ORDER},
            "hold_s": hold_s,
        }

    def run_forever(self, return_on_finish=True, args=None, execute=True) -> dict[str, Any]:
        """
        Kept for interface similarity with FaceTrack / PhoneTouch.
        This is actually a one-shot movement.
        """
        return self.move_to_target(args["target_pose"])