
class Agent:
    def __init__(self, observation_shape: int, num_actions: int, batch_size: int = 256, memory_size: int = 10e6,
                 learning_rate: float = 3e-4, alpha: float = 1, gamma: float = 0.99, tau: float = 0.005,
                 hidden_units: Optional[Sequence[int]] = None, load_models: bool = False,
                 checkpoint_directory: Optional[Union[Path, str]] = None):  # todo implement hard update

    def choose_action(self, observation, deterministically: bool = False) -> np.array:
        while not done:
            if start_step > global_step:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(observation)

        observation = torch.FloatTensor(observation).to(self.device)
        action = self.policy.act(observation, deterministic=deterministically)
        return action
