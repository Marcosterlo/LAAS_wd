import argparse
from stable_baselines3 import PPO
from integrator_env import Integrator_env
import numpy as np
import matplotlib.pyplot as plt

env = Integrator_env()

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
model = PPO.load(args.model_name)

nstep = 50000
state = []
u = []

obs = env.reset()[0]
for i in range(nstep):
  action, _states = model.predict(obs, deterministic=True)
  obs, reward, done, truncated, info = env.step(action[0])
  if done or truncated:
    break
  state.append(obs)
  u.append(action[0])

state = np.array(state)
u = np.array(u)
time_grid = np.linspace(0, len(state), len(state))

print(f"Initial state: x0: {state[0][0]/np.pi*180:.2f}, dx0: {state[0][1]:.2f},")

plt.plot(time_grid, state[:,0])
plt.grid(True)
plt.title("Position")
plt.show()

plt.plot(time_grid, state[:,1])
plt.grid(True)
plt.title("Velocity")
plt.show()

plt.plot(time_grid, state[:,2])
plt.grid(True)
plt.title("Eta")
plt.show()

plt.plot(time_grid, u)
plt.grid(True)
plt.title("Control")
plt.show()