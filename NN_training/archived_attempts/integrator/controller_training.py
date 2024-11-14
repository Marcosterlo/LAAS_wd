from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from integrator_env import Integrator_env
from torch import nn

# Custom environment import
env = Integrator_env()

# Check if the environment is correctly defined
check_env(env, warn=True)

policy_kwargs = dict(
  net_arch=[32, 32, 32],
  activation_fn=nn.Hardtanh
)

model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=500000)

model.save('linear')