import os
import time
import json
import dotenv
import logging
import traceback
from multiprocessing import Pool
import multiprocessing as mp
from robiagent.agents.basic.skillset import BasicSkillset
from robiagent.utils.misc import set_log
from robiagent.agents.base import BaseAgent
from robiagent.agents.basic.planner import BasicPlanner
from robiagent.agents.basic.const import END

dir_path = os.path.dirname(os.path.realpath(__file__))


class BasicAgent(BaseAgent):
    TASK_WAITING_TIME_IN_SEC = 0.1

    def __init__(self, config, environment):
        self.environment = environment
        self.planner = BasicPlanner(
            config.agent.planner,
            environment
        )
        self.config = config.agent

        self.all_tasks = {}
        self.completed_task_names = []
        self.running_task_names = []
        self.pending_task_names = []

        self.skillset = BasicSkillset(config)

    @staticmethod
    def preprocessing_display(task):
        print(f"\n[Execution] I am now about to run task {task['name']} with skill {task['skill']}.\n"
              f"\tDetail: {task['description'][:500]}...")

    @staticmethod
    def postprocessing_display(task, task_succeeded, detail):
        print(f'\n[Execution] Task {task['name']} with skill {task['skill']} '
              f'{"succeeded" if task_succeeded else "failed"}.')
        logging.info(f'Task {task['name']} with skill {task['skill']} '
                     f'{"succeeded" if task_succeeded else "failed"}:\n{detail}')

    def process_a_task(self, task):  # this run in a new process
        task_name = task['name']
        task_skill = task['skill']
        dotenv.load_dotenv(dotenv_path='.env', override=True)
        log_path = self.environment.get_log_path()
        set_log(log_path=log_path)

        logging.info(f'Starting to process task {task_name} with skill {task_skill}')
        begin_time = time.perf_counter()

        retry_count = 0
        max_retries = self.config.task_retry_time_limit
        task_succeeded = False
        while retry_count <= max_retries:
            try:
                arguments = task['arguments']
                self.preprocessing_display(task)
                result = self.skillset.use_skill(
                    skill_name=task_skill,
                    skill_args=arguments
                )

                self.environment.set_task_result(
                    task=task,
                    result=result
                )

                task_succeeded = result["err_code"] == 0
                detail = result["detail"]
                self.postprocessing_display(task, task_succeeded, detail)
            except Exception as e:
                result = f"Exception encountered: {e}\n{traceback.format_exc()}"
                logging.info(f'Task {task_name} with skill {task_skill} '
                             f'failed as {result}')

                self.environment.set_task_result(
                    task=task,
                    result=result
                )
            if task_succeeded:
                break

            retry_count += 1
            if retry_count > max_retries:
                logging.error(f'Task {task_name} failed after {max_retries} retries')
                return

            logging.info(f'Retrying task {task_name} (Attempt {retry_count}/{max_retries})')
            task = self.environment.get_task_state(task_name)  # refreshing

        end_time = time.perf_counter()
        duration = end_time - begin_time
        logging.info(f'Processing for task {task_name} with skill {task_skill} '
                     f'finished in {round(duration, 3)}s')

    def refresh_pending_tasks(self):
        self.pending_task_names = []
        for task_name in self.all_tasks.keys():
            if (task_name not in self.running_task_names
                    and task_name not in self.completed_task_names):
                self.pending_task_names.append(task_name)

    def pending_tasks_exist(self):
        return len(self.pending_task_names) > 0

    def running_tasks_exist(self):
        return len(self.running_task_names) > 0

    @staticmethod
    def get_newly_finished_tasks(async_results):
        return [
            task_name for task_name, result in async_results.items()
            if result.ready()
        ]

    def set_tasks(self, task_list):
        for task in task_list:
            task_name = task["name"]
            self.all_tasks[task_name] = task
        self.refresh_pending_tasks()

        logging.info(f"Current plan:\n{json.dumps(self.all_tasks, indent=4)}")

    def get_a_ready_task(self):
        for task_name in self.pending_task_names:
            task = self.all_tasks[task_name]
            if set(task['dependencies']).issubset(self.completed_task_names):
                return task
        return None

    @staticmethod
    def propagate_exception(task_name, async_results):
        async_results[task_name].get(timeout=0)  # To propagate any exceptions

    def mark_completed_task(self, task_name, async_results):
        if task_name in self.running_task_names:
            del async_results[task_name]
            self.running_task_names.remove(task_name)
        elif task_name in self.pending_task_names:
            self.pending_task_names.remove(task_name)

        self.completed_task_names.append(task_name)  # even originally pending task needs to be appended

    def mark_running_task(self, task_name, result, async_results):
        self.pending_task_names.remove(task_name)
        self.running_task_names.append(task_name)
        async_results[task_name] = result

    def serve(self, overall_task):
        start_time = time.perf_counter()
        logging.info(f"Starting serving the overall task: {overall_task}")

        num_tasks_launched = 0
        try:
            all_tasks = self.planner.plan(overall_task)
            self.set_tasks(all_tasks)
            async_results = {}

            abort = False
            mp.set_start_method("spawn", force=True)
            with Pool(processes=self.config.max_workers) as pool:
                # Loop until no pending tasks remain and all running tasks have finished
                while self.pending_tasks_exist() or self.running_tasks_exist():

                    # Step 1: Check which running tasks have completed
                    newly_finished_tasks = self.get_newly_finished_tasks(async_results)
                    for task_name in newly_finished_tasks:
                        try:
                            self.propagate_exception(task_name, async_results)
                        except Exception as e:  # TODO: if necessary, deal with it
                            logging.error(f"Task {task_name} not finished due to {e}\n"
                                          f"{traceback.format_exc()}")
                        self.mark_completed_task(task_name, async_results)

                    # Step 2: Schedule any tasks whose dependencies are met
                    while True:
                        ready_task = self.get_a_ready_task()
                        if not ready_task:
                            break

                        num_tasks_launched += 1
                        logging.info(f"Num tasks launched: {num_tasks_launched}")
                        if num_tasks_launched > self.config.max_num_tasks_launched:
                            logging.info(f"Exceeded max number of tasks. Aborting...")
                            abort = True
                            break
                        result = pool.apply_async(self.process_a_task, (ready_task,))
                        self.mark_running_task(ready_task["name"], result, async_results)
                    if abort:
                        break

                    # Avoid busy waiting
                    if not ready_task and not newly_finished_tasks:
                        time.sleep(self.TASK_WAITING_TIME_IN_SEC)
                        continue

        except KeyboardInterrupt:
            print("Interrupted.")
        except Exception as e:
            print(f"Failed to serve query due to {e}")
            print(traceback.format_exc())
        finally:
            # to notify other threads to stop
            self.environment.publish_a_message(channel=END, message="done")

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"\nTask served in {round(duration, 3)}s")
        logging.info(f"Task served in {round(duration, 3)}s")
