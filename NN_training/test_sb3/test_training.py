from stable_baselines3 import PPO
from NN_training.environments.simple_pendulum_env import Simple_pendulum_env
import matplotlib.pyplot as plt
import numpy as np

env = Simple_pendulum_env()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, progress_bar=True)

states = []
inputs = []

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
  action, _states = model.predict(obs, deterministic=True)
  obs, rewards, done, info = vec_env.step(action)
  states.append(obs)
  inputs.append(action)

states = np.squeeze(np.array(states))
inputs = np.squeeze(np.array(inputs))

plt.plot(states[:,0], states[:,1])
plt.show()

plt.plot(inputs)
plt.show()
