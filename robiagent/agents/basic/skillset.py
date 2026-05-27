"""
In case one need shell code execution, consider reusing the following built-in ability:

from robiagent.backend.host.shell import Shell
shell = Shell()
shell.execute_code(command)
"""
import logging
import os

from robiagent.backend.arm.face_track import FaceTrack
from robiagent.backend.arm.face_track_new import FaceTrackNew
from robiagent.backend.arm.phone_touch import PhoneTouch
from robiagent.backend.arm.simple_movement import SimpleMove
from robiagent.backend.host.vocal_intro import VocalIntro


class BasicSkillset(object):
    def __init__(self, config):
        super().__init__()
        self.physical_config = config.physical
        self.task_config = config.task
        self.initialized_bodies = {}

    @staticmethod
    def get_body_need_initializing(task):
        task_type = task['skill']
        if task_type not in [
            "simple_movement", "touch_phone",
            "track_face_new", "track_face"
        ]:
            return None
        return task["arguments"]["body"]

    def get_initialized_body(self, body, task_type):
        body_id = f"{body}_{task_type}"
        return self.initialized_bodies[body_id]

    def initialize_body(self, body, task_type):
        body_id = f"{body}_{task_type}"
        logging.info(f"[Skillset] Initializing body {body_id}")

        if body == 'left':
            body_config = self.physical_config.arm.left
            camera_config = self.physical_config.camera.left
        elif body == 'right':
            body_config = self.physical_config.arm.right
            camera_config = self.physical_config.camera.right
        else:
            raise NotImplementedError

        if "track_face" in task_type:
            if "new" in task_type:
                task_config = self.task_config.track_face_new
                face_track = FaceTrackNew
            else:
                task_config = self.task_config.track_face
                face_track = FaceTrack

            arm = face_track(
                task_config=task_config,
                body_config=body_config,
                camera_config=camera_config
            )
        elif task_type == 'simple_movement':
            task_config = self.task_config.simple_movement
            arm = SimpleMove(
                task_config=task_config,
                body_config=body_config
            )
        elif task_type == 'touch_phone':
            task_config = self.task_config.touch_phone
            arm = PhoneTouch(
                task_config=task_config,
                body_config=body_config,
                camera_config=camera_config
            )
        else:
            raise NotImplementedError

        arm.connect()
        logging.info(f"[Skillset] Body {body_id} initialized for task {task_type}")
        self.initialized_bodies[body_id] = arm

    def body_action(self, task_type, args):
        # Currently a body uses its own camera
        body = args['body']
        logging.info(f"[DEBUG] (pid: {os.getpid()}) Getting initialized body {body} for task {task_type}")
        arm = self.get_initialized_body(body, task_type)

        try:
            arm.run_forever(return_on_finish=True, args=args)
            # arm.run_forever(return_on_finish=True, execute=False)  # for debugging with no real movement
            resp = "done"
        except KeyboardInterrupt:
            resp = "interrupted"
            arm.disconnect()

        result = {
            "err_code": 0,
            "detail": resp
        }
        return result

    def disconnect_all_bodies(self):
        for body in self.initialized_bodies.values():
            body.disconnect()

    def pc_action(self, task_type, args):
        if task_type == "vocal_intro":
            task_config = self.task_config.vocal_intro
            task_config.text = args["text"]

            vocal_intro = VocalIntro(task_config)
            result = vocal_intro.play()
        else:
            raise NotImplementedError

        return result

    def use_skill(self, skill_name, skill_args):
        if skill_name in [
            "simple_movement",
            "track_face_new",
            "track_face",
            "touch_phone"
        ]:
            result = self.body_action(skill_name, skill_args)
        elif skill_name in ["vocal_intro"]:
            result = self.pc_action(skill_name, skill_args)
        else:
            raise NotImplementedError

        return result
