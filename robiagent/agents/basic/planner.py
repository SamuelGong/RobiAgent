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
                {
                    "name": "track_face",
                    "description": "Tracking face using left arm",
                    "dependencies": [],
                    "skill": "single_arm",
                    "arguments": {
                        'body': 'left',
                        'action': "Tracking face"
                    }
                },
                # {
                #     "name": "track_face_new",
                #     "description": "Tracking face (new) using left arm",
                #     "dependencies": [],
                #     "skill": "single_arm",
                #     "arguments": {
                #         'body': 'left',
                #         'action': "Tracking face (new)"
                #     }
                # },
                # {
                #     "name": "vocal_intro",
                #     "description": "Playing introduction using microphone",
                #     # "dependencies": [],
                #     "dependencies": ["track_face_new"],
                #     "skill": "pc",
                #     "arguments": {
                #         'action': "Playing introduction"
                #     }
                # },
                # {
                #     "name": "usage_show",
                #     "description": "Demonstrating a new feature using right arm",
                #     "dependencies": ["track_face_new"],
                #     "skill": "single_arm",
                #     "arguments": {
                #         'body': 'right',
                #         'action': "Demonstrating a new feature"
                #     }
                # }
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
