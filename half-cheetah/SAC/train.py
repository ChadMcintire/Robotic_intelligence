import gym
from typing import Optional
from utility import GracefulKiller
from itertools import count


#Optional[X] as a shorthand for Union[X, None]
def train(
    env_name: str,
    env_kwargs: Optional[dict] = None,
    **kwargs
):
    killer = GracefulKiller()
    env_kwargs = env_kwargs or {}
    print("env_kwargs", **env_kwargs)
    print("env_name", env_name)

    # **unpacks a dictionary into keywords args for a function call
    env = gym.make(env_name, **env_kwargs)

    observation_shape = env.observation_space.shape[0]
    print("observation space", observation_shape )

    num_actions = env.action_space.shape[0]
    print("action space", num_actions)

    for episode in count():
        observation = env.reset()

        #while not done:
        #    if start_step > global_step

        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False

