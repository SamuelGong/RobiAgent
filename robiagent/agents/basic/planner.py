import logging


class BasicPlanner():
    def __init__(self, config, environment):
        self.config = config
        self.environment = environment
        self.task_name_history = set()
        self.print_tag = "[Planning]"

    def display_plan(self, task_list):
        task_names = [task["name"] for task in task_list]
        print(f"\n{self.print_tag} I now decompose the task into "
              f"{len(task_names)} sub-task(s): {task_names}.")

    # Yes, for basic planner, we just do hard-coding
    def predefined_decomposition(self, task):
        if task in ["demo"]:
            task_list = [
                # {
                #     "name": "vocal_intro_1",
                #     "description": "Playing introduction using microphone",
                #     "dependencies": [],
                #     "skill": "vocal_intro",
                #     "arguments": {
                #         'text': "你好，欢迎光临华为南京东路旗舰店。眼前的这款手机叫 Pura 90 Pro，它拥有6.6英寸的利落直屏，还有业界首发双色渐变金属中框。简约，精致，出彩！"
                #     }
                # },
                # {
                #     "name": "track_face_new",
                #     "description": "Tracking face (new) using left arm",
                #     "dependencies": [],
                #     "skill": "track_face_new",
                #     "arguments": {
                #         'body': 'left'
                #     }
                # },
                # {
                #     "name": "simple_movement",
                #     "description": "Let left arm be somewhere specified",
                #     "dependencies": ["track_face_new", "vocal_intro_1"],
                #     "skill": "simple_movement",
                #     "arguments": {
                #         'body': 'left',
                #     }
                # },
                # {
                #     "name": "vocal_intro_2",
                #     "description": "Playing introduction using microphone",
                #     "dependencies": ["track_face_new", "vocal_intro_1"],
                #     "skill": "vocal_intro",
                #     "arguments": {
                #         'text': "现在，请到这边来。我来给你演示更多AI拍照特性吧。"
                #     }
                # },
                # {
                #     "name": "touch_phone_1",
                #     "description": "scroll phone and touch phone using right arm",
                #     "dependencies": ["simple_movement", "vocal_intro_2"],
                #     "skill": "touch_phone",
                #     "arguments": {
                #         'body': 'right',
                #     }
                # },
                # {
                #     "name": "vocal_intro_3",
                #     "description": "Playing introduction using microphone",
                #     "dependencies": ["simple_movement", "vocal_intro_2"],
                #     "skill": "vocal_intro",
                #     "arguments": {
                #         'text': "首先是 XMAGE 智拍：Pura 90 Pro 能从姿势、构图到画面氛围全程智能辅助，让随手一拍也更有大片感。"
                #     }
                # },
                # {
                #     "name": "touch_phone_2",
                #     "description": "scroll phone and touch phone using right arm",
                #     "dependencies": ["touch_phone_1", "vocal_intro_3"],
                #     "skill": "touch_phone",
                #     "arguments": {
                #         'body': 'right'
                #     }
                # },
                # {
                #     "name": "vocal_intro_4",
                #     "description": "Playing introduction using microphone",
                #     "dependencies": ["touch_phone_1", "vocal_intro_3"],
                #     "skill": "vocal_intro",
                #     "arguments": {
                #         'text': "其次，AI 姿势推荐：当你不知道怎么站、手怎么放时，它会根据海边、街巷、咖啡店等场景，智能推荐更自然好看的拍照姿势。"
                #     }
                # },
                # {
                #     "name": "touch_phone_3",
                #     "description": "scroll phone and touch phone using right arm",
                #     "dependencies": ["touch_phone_2", "vocal_intro_4"],
                #     "skill": "touch_phone",
                #     "arguments": {
                #         'body': 'right'
                #     }
                # },
                # {
                #     "name": "vocal_intro_5",
                #     "description": "Playing introduction using microphone",
                #     "dependencies": ["touch_phone_2", "vocal_intro_4"],
                #     "skill": "vocal_intro",
                #     "arguments": {
                #         'text': "还有 AI 辅助构图：面对风景、合影或人像时，它会用图标和提示语引导你找到更合适的机位和画面比例，让照片更协调、更出片。"
                #     }
                # },
                {
                    "name": "track_face_new",
                    "description": "Tracking face (new) using left arm",
                    "dependencies": [],
                    "skill": "track_face_new",
                    "arguments": {
                        'body': 'left'
                    }
                },
                # {
                #     "name": "touch_phone",
                #     "description": "scroll phone and touch phone using right arm",
                #     "dependencies": [],
                #     "skill": "touch_phone",
                #     "arguments": {
                #         'body': 'right'
                #     }
                # },
            ]
        else:
            raise NotImplementedError

        return task_list

    def plan(self, task):
        logging.info(f"Starting to plan")

        task_list = self.predefined_decomposition(task)  # TODO: to support model-based planning
        logging.info(f"Planning ended")
        for task in task_list:
            self.task_name_history.add(task["name"])

        self.display_plan(task_list)
        return task_list
