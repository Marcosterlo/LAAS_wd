from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from pendulum_env import Pendulum_env
from torch import nn
import os
import pandas as pd

# Custom environment import
env = Pendulum_env()
# Initiation of Monitor
env = Monitor(env, filename='./monitor_logs')

# Check if the environment is correctly defined
check_env(env, warn=True)

# Args to define a policy of 4 layers of 32 neurons per layer
policy_kwargs = dict(
    net_arch=[32, 32, 32, 32],
    activation_fn=nn.Tanh
)

log_name = os.path.abspath(__file__ + "/../logs/monitor.csv")

# Definition of model along with hyperparameters
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, ent_coef=0.05, gamma=0.995, learning_rate=1e-3)

# Training of model
model.learn(total_timesteps=500000)

# Loading monior for data analysis
monitor_data = pd.read_csv(log_name, skiprows=1)
print(monitor_data.head())