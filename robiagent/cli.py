import sys
from robiagent.utils.misc import setup
from robiagent.agents.registry import get_agent
from robiagent.environments.registry import get_environment


def main():
    if len(sys.argv) < 2:
        print("Usage: ra '<your overall task>'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])  # Combine all arguments into a single string
    config = setup()
    environment = get_environment(config)
    agent = get_agent(config.agent, environment)
    agent.serve(task)


if __name__ == '__main__':
    main()
