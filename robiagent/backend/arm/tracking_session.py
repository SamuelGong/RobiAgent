from __future__ import annotations

import cv2
import time
from cv2_enumerate_cameras import enumerate_cameras
from typing import Any, Callable, Optional, Protocol


class HasFaceTrackTaskConfig(Protocol):
    track_duration_s: Any
    hold_last_on_timeout: bool
    lost_timeout: float


class TrackingSession:
    def __init__(self, task_config: HasFaceTrackTaskConfig):
        self.task_config = task_config
        self.track_start_time: float | None = None
        self.tracking_active: bool = True
        self.holding_last_pose: bool = False
        self.last_face_time: float = 0.0

    def begin_after_connect(self) -> None:
        self.track_start_time = time.time()
        self.tracking_active = True
        self.holding_last_pose = False

    def restart(self) -> None:
        self.track_start_time = time.time()
        self.tracking_active = True
        self.holding_last_pose = False

    def update_timeout(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        if self.task_config.track_duration_s is None:
            return
        if self.track_start_time is None:
            self.track_start_time = now
            return
        if (now - self.track_start_time) >= self.task_config.track_duration_s:
            self.tracking_active = False
            self.holding_last_pose = self.task_config.hold_last_on_timeout

    def timed_out_hold_last(self) -> bool:
        return not self.tracking_active and self.holding_last_pose

    def mark_face_seen(self, now: float) -> None:
        self.last_face_time = now


def _uid_candidates(uid) -> set[str]:
    s = str(uid).strip()
    candidates = {s, s.lower()}

    # 0x... 十六进制
    if s.lower().startswith("0x"):
        try:
            n = int(s, 16)
            candidates.add(str(n))
            candidates.add(hex(n).lower())
            candidates.add(format(n, "x").lower())
        except ValueError:
            pass

    # 纯数字，可能是十进制 UID
    elif s.isdigit():
        try:
            n = int(s, 10)
            candidates.add(str(n))
            candidates.add(hex(n).lower())
            candidates.add(format(n, "x").lower())
        except ValueError:
            pass

    # 裸十六进制，不带 0x，例如 2400000bda5883
    elif all(c in "0123456789abcdefABCDEF" for c in s):
        try:
            n = int(s, 16)
            candidates.add(str(n))
            candidates.add(hex(n).lower())
            candidates.add(format(n, "x").lower())
        except ValueError:
            pass

    return candidates


def get_camera_id_by_uid(
    uid: str | int,
    backend: int = cv2.CAP_AVFOUNDATION,
) -> int:
    target_candidates = _uid_candidates(uid)

    for cam in enumerate_cameras(backend):
        cam_candidates = _uid_candidates(cam.path)

        if target_candidates & cam_candidates:
            return cam.index

    available = [
        {
            "index": cam.index,
            "name": getattr(cam, "name", None),
            "path": str(cam.path),
        }
        for cam in enumerate_cameras(backend)
    ]

    raise RuntimeError(
        f"camera not found: {uid}\n"
        f"normalized candidates: {sorted(target_candidates)}\n"
        f"available cameras: {available}"
    )


# Open camera, run process_frame + draw_debug + imshow until quit or finished.
def run_face_tracking_loop(
    tracker: Any,
    camera_config: Any,
    *,
    window_title: str,
    return_on_finish: bool = True,
    execute: bool = True,
    on_key: Optional[Callable[[int], None]] = None,
) -> dict | None:
    # Lazy-imported from ``face_track`` to avoid import cycles with modules that import ``TrackingSession`` from this module.
    from robiagent.backend.arm.face_track import open_camera

    id = get_camera_id_by_uid(uid=camera_config.uid)
    cap = open_camera(
        camera_id=id,
        width=camera_config.width,
        height=camera_config.height,
        fps=camera_config.fps,
    )
    last_result: dict | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            result = tracker.process_frame(frame, execute=execute)
            last_result = result
            display = tracker.draw_debug(frame, result)
            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF

            if return_on_finish and result.get("finished", False):
                break
            if key == 27 or key == ord("q"):
                break
            if on_key is not None:
                on_key(key)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return last_result
