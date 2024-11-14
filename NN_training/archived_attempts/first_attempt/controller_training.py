from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from pendulum_env import Pendulum_env
from torch import nn

# Custom environment import
env = Pendulum_env()

# Check if the environment is correctly defined
check_env(env, warn=True)

# Args to define a policy of 4 layers of 32 neurons per layer
policy_kwargs = dict(
    net_arch=[8, 16, 8],
    activation_fn=nn.Tanh
)

# Definition of model along with hyperparameters
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, ent_coef=0.0, gae_lambda=0.95, gamma=0.9, learning_rate=0.0001, n_steps=4096)

# Training of model
model.learn(total_timesteps=200000)

model.save('linear')