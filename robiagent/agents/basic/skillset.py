from robiagent.backend.lerobot.so101.face_track import FaceTrack, CameraIntrinsics, Settings


class BasicSkillset(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def single_arm_action(self, args):
        body = args['body']
        if body == 'left':
            port = self.config.physical.arm.left.port
            robot_id = self.config.physical.arm.left.id
            intrinsics = CameraIntrinsics(
                fx=self.config.physical.camera.left.fx,
                fy=self.config.physical.camera.left.fy,
                cx=self.config.physical.camera.left.cx,
                cy=self.config.physical.camera.left.cy,
            )
            camera_id = self.config.physical.camera.left.id
        else:
            raise NotImplementedError

        action = args['action']
        if action == 'Tracking face':
            settings = Settings(
                camera_id=camera_id,
                track_duration_s=self.config.task.face_track.duration,
                hold_last_on_timeout=True
            )
            face_track = FaceTrack(
                port=port,
                robot_id=robot_id,
                mp_model="third-party/face_detector.task",  # TODO: avoid hard-coding
                intrinsics=intrinsics,
                cfg=settings,
                disable_calibration=True
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