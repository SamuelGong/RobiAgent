from robiagent.backend.lerobot.so101.face_track import FaceTrack


class BasicSkillset(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.physical_config = self.config.physical
        self.task_config = self.config.task

    def single_arm_action(self, args):
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
        if action == 'Tracking face':
            task_config = self.task_config.face_track
            face_track = FaceTrack(
                task_config=task_config,
                body_config=body_config,
                camera_config=camera_config
            )
            face_track.connect()
            try:
                face_track.run_forever(return_on_finish=True)
            finally:
                face_track.disconnect()

            resp = "done"
        else:
            raise NotImplementedError

        result = {
            "err_code": 0,
            "detail": resp
        }
        return result

    def pc_action(self, args):
        result = {
            "err_code": 0,
            "detail": ""
        }
        return result

    def use_skill(self, skill_name, skill_args):
        if skill_name == 'single_arm':
            result = self.single_arm_action(skill_args)
        elif skill_name == 'pc':
            result = self.pc_action(skill_args)
        else:
            raise NotImplementedError

        return result


'''
In case one need shell code execution:

from robiagent.backend.host.shell import Shell
shell = Shell()
shell.execute_code(command)
'''