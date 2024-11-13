from stable_baselines3 import PPO
from matplotlib import pyplot as plt
import numpy as np
from NN_training.environments.pendulum_integrator_env import NonLinPendulum_env

env = NonLinPendulum_env()

best_model = PPO.load('best_rollout_model.zip', env=env)

states = []
inputs = []
episode_state = []
episode_input = []

n_tot = 0
n_converging = 0

vec_env = best_model.get_env()
obs = vec_env.reset()
for i in range(1000):
  action, _states = best_model.predict(obs, deterministic=True)
  obs, rewards, done, info = vec_env.step(action)
  episode_state.append(obs)
  episode_input.append(action)
  if done:
    n_tot += 1
    states.append(episode_state)
    inputs.append(episode_input)
    episode_state = []
    episode_input = []
    
for episode in states:
  if len(episode) > 50:
    n_converging += 1
    episode = np.squeeze(np.array(episode))
    plt.plot(episode[:-1,0], episode[:-1,1])
plt.grid(True)
plt.show()

for episode in inputs:
  if len(episode) > 50:
    episode = np.squeeze(np.array(episode))
    plt.plot(episode*env.max_torque)
plt.grid(True)
plt.show()

print(f'Converging episodes: {n_converging}/{n_tot}')