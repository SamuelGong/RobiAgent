from robiagent.agents.basic.core import BasicAgent

registered_agents = {
    'basic': BasicAgent
}


def get_agent(config, environment):
    return registered_agents[config.agent.type](config, environment)
