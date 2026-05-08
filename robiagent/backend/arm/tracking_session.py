from __future__ import annotations

import cv2
import time
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

    cap = open_camera(
        camera_id=camera_config.id,
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
