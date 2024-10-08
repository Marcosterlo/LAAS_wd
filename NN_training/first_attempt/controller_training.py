from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from pendulum_env import Pendulum_env
from torch import nn

# Custom environment import
env = Pendulum_env()

# Check if the environment is correctly defined
check_env(env, warn=True)

class CustomTanhPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomTanhPolicy, self).__init__(*args, **kwargs)

        self.action_net = nn.Sequential(
            self.action_net,
            nn.Tanh()
        )

# Args to define a policy of 4 layers of 32 neurons per layer
policy_kwargs = dict(
    net_arch=[16, 16, 16, 16],
    activation_fn=nn.Tanh
)

# Definition of model along with hyperparameters
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, ent_coef=0.05, gamma = 0.99) # , gamma=0.995) # , learning_rate=1e-3)

# Training of model
model.learn(total_timesteps=100000)