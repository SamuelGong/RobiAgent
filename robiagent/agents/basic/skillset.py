from robiagent.backend.arm.face_track import FaceTrack
from robiagent.backend.arm.face_track_new import FaceTrackNew
from robiagent.backend.host.vocal_intro import VocalIntro
from robiagent.backend.arm.phone_touch import PhoneTouch


"""
In case one need shell code execution, consider reusing the following built-in ability:

from robiagent.backend.host.shell import Shell
shell = Shell()
shell.execute_code(command)
"""


class BasicSkillset(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.physical_config = self.config.physical
        self.task_config = self.config.task

    def single_arm_action(self, args):
        # Currently a body uses its own camera
        body = args['body']
        if body == 'left':
            body_config = self.physical_config.arm.left
            camera_config = self.physical_config.camera.left
        elif body == 'right':
            body_config = self.physical_config.arm.right
            camera_config = self.physical_config.camera.right
        else:
            raise NotImplementedError

        action = args['action']
        if 'tracking face' in action.lower():
            if "new" in action.lower() in action.lower():
                task_config = self.task_config.face_track_new
                face_track = FaceTrackNew
            else:
                task_config = self.task_config.face_track
                face_track = FaceTrack

            arm = face_track(
                task_config=task_config,
                body_config=body_config,
                camera_config=camera_config
            )
            arm.connect()

            try:
                arm.run_forever(return_on_finish=True)
                # arm.run_forever(return_on_finish=True, execute=False)  # for debugging with no real movement
                resp = "done"
            except KeyboardInterrupt:
                resp = "interrupted"
            finally:
                arm.disconnect()
        elif 'touch phone' in action.lower():
            task_config = self.task_config.phone_touch
            arm = PhoneTouch(task_config, body_config, camera_config)
            arm.connect()
            try:
                arm.run_forever()
                resp = "done"
            except KeyboardInterrupt:
                resp = "interrupted"
            finally:
                arm.disconnect()

        else:
            raise NotImplementedError

        result = {
            "err_code": 0,
            "detail": resp
        }
        return result

    def pc_action(self, args):
        action = args['action']
        if 'playing introduction' in action.lower():
            task_config = self.task_config.vocal_intro
            vocal_intro = VocalIntro(task_config)
            result = vocal_intro.play()
        else:
            raise NotImplementedError

        return result

    def use_skill(self, skill_name, skill_args):
        if skill_name == 'single_arm':
            result = self.single_arm_action(skill_args)
        elif skill_name == 'pc':
            result = self.pc_action(skill_args)
        else:
            raise NotImplementedError

        return result
