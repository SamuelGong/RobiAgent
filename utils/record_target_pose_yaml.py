from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# =========================
# Hard-coded config
# =========================
ROBOT_ID = "shanghai_tracking_arm"       # TODO: change me
PORT = "/dev/tty.usbmodem5B141139931"    # TODO: change me
OUTPUT_PATH = Path("target_pose.txt")

# Keep this False if you want the arm to remain free after recording.
# Set True if you want the arm to hold the recorded pose after you press Enter.
REENABLE_TORQUE_AFTER_RECORD = False

# The output YAML list will follow this order if all keys exist.
PREFERRED_JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def make_robot(robot_id: str, port: str) -> Any:
    """Import whichever SO follower class is available in the current lerobot version."""
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        try:
            config = SOFollowerRobotConfig(
                id=robot_id,
                port=port,
                use_degrees=True,
                disable_torque_on_disconnect=False,
            )
        except TypeError:
            config = SOFollowerRobotConfig(id=robot_id, port=port)
        return SOFollower(config)
    except Exception as first_error:
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

            try:
                config = SO101FollowerConfig(
                    id=robot_id,
                    port=port,
                    disable_torque_on_disconnect=False,
                )
            except TypeError:
                config = SO101FollowerConfig(port=port, id=robot_id)
            return SO101Follower(config)
        except Exception as second_error:
            raise RuntimeError(
                "Could not import a compatible SO follower class.\n"
                f"so_follower error: {first_error}\n"
                f"so101_follower error: {second_error}"
            ) from second_error


def get_motor_names(robot: Any) -> list[str]:
    bus = getattr(robot, "bus", None)
    motors = getattr(bus, "motors", None)
    if isinstance(motors, dict):
        return list(motors.keys())
    if isinstance(motors, (list, tuple)):
        return list(motors)
    return []


def _try_method(obj: Any, method_name: str, *args: Any) -> bool:
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            method(*args)
            return True
        except TypeError:
            return False
    return False


def set_torque_enabled(robot: Any, enabled: bool) -> None:
    """
    Best-effort torque control across a few lerobot/bus API variants.

    If your local bus API is different, add the exact call in this function.
    """
    bus = getattr(robot, "bus", None)
    motor_names = get_motor_names(robot)

    if enabled:
        for obj in (robot, bus):
            if obj is not None and _try_method(obj, "enable_torque"):
                return
            if obj is not None and motor_names and _try_method(obj, "enable_torque", motor_names):
                return
    else:
        for obj in (robot, bus):
            if obj is not None and _try_method(obj, "disable_torque"):
                return
            if obj is not None and motor_names and _try_method(obj, "disable_torque", motor_names):
                return

    # Common bus.write variants.
    value = 1 if enabled else 0
    field_names = [
        "Torque_Enable",
        "Torque Enable",
        "torque_enable",
        "Torque",
    ]

    write = getattr(bus, "write", None)
    if callable(write):
        for field in field_names:
            variants = [
                (field, value),
                (field, value, motor_names),
                (field, {name: value for name in motor_names}),
            ]
            for args in variants:
                try:
                    write(*args)
                    return
                except Exception:
                    pass

    raise RuntimeError(
        "Could not toggle torque automatically. Your local lerobot bus API may use "
        "a different method name. Edit set_torque_enabled() with the exact torque call."
    )


def get_pose(robot: Any, action_keys: list[str]) -> dict[str, float]:
    obs = robot.get_observation()
    return {key: float(obs[key]) for key in action_keys if key in obs}


def choose_output_order(action_keys: list[str], pose: dict[str, float]) -> list[str]:
    if all(key in pose for key in PREFERRED_JOINT_ORDER):
        return list(PREFERRED_JOINT_ORDER)
    return [key for key in action_keys if key in pose]


def main() -> int:
    print(f"Connecting robot id={ROBOT_ID!r}, port={PORT!r} ...")
    robot = make_robot(ROBOT_ID, PORT)

    try:
        robot.connect(calibrate=False)
        action_keys = list(robot.action_features.keys())
        print("Connected. action_keys:")
        for i, key in enumerate(action_keys):
            print(f"  {i}: {key}")

        print("\nDisabling torque. Hold the arm before it becomes free.")
        time.sleep(0.5)
        set_torque_enabled(robot, False)
        print("Torque disabled. Move the arm by hand to the desired pose.")
        input("\nPress Enter to record current pose... ")

        pose = get_pose(robot, action_keys)
        order = choose_output_order(action_keys, pose)
        values = [round(float(pose[key]), 6) for key in order]
        yaml_list_text = "\n".join(f"- {value}" for value in values) + "\n"

        # OUTPUT_PATH.write_text(yaml_list_text, encoding="utf-8")
        # print(f"\nSaved YAML list to: {OUTPUT_PATH.resolve()}")
        # print("\nYAML list:")
        # print(yaml_list_text, end="")

        json_list_text = "\n".join(f"{value}," for value in values)[:-1]
        OUTPUT_PATH.write_text(json_list_text, encoding="utf-8")
        print(f"\nSaved JSON list to: {OUTPUT_PATH.resolve()}")
        print("\nJSON list:")
        print(json_list_text, end="")

        print("\nJoint order for the YAML list above:")
        for i, key in enumerate(order):
            print(f"  {i}: {key} = {pose[key]:.6f}")

        print("\nFull pose dict:")
        print(json.dumps({key: round(float(pose[key]), 6) for key in order}, indent=2, ensure_ascii=False))

        if REENABLE_TORQUE_AFTER_RECORD:
            print("\nRe-enabling torque to hold current pose...")
            set_torque_enabled(robot, True)
            print("Torque enabled.")
        else:
            print("\nTorque left disabled. Support the arm before exiting if needed.")

        return 0

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
