from systems_and_LMI.systems.NonLinearPendulum import NonLinPendulum
import numpy as np
import pandas as pd
import control

s = NonLinPendulum()

A = s.A
B = s.B

Q = np.array([
  [100, 0, 0],
  [0, 10, 0],
  [0, 0, 1]
])

R = np.array([[1]])

K, _, _ = control.dlqr(A, B, Q, R)

print(f"K: {K}")

eigvals = np.linalg.eigvals(A - B @ K)
print(f"Eigvals of closed loop system: {eigvals}")

stable = all(abs(eig) < 1 for eig in eigvals)
if stable:
  print("System is stable")
  np.save('K.npy', -K)
else:
  print("System is not stable")

size_dataset = 1000000

theta_lim = 60 * np.pi / 180

theta_mean = 0
theta_std = theta_lim / 3  # 3-sigma rule for 99.7% coverage
vtheta_mean = 0
vtheta_std = s.max_speed / 3  # 3-sigma rule for 99.7% coverage
eta_mean = 0
eta_std = 1  # Assuming a standard deviation for eta

dataset = []

for _ in range(size_dataset):
  theta = np.clip(np.random.normal(theta_mean, theta_std), -theta_lim, theta_lim)
  vtheta = np.clip(np.random.normal(vtheta_mean, vtheta_std), -s.max_speed, s.max_speed)
  eta = np.random.normal(eta_mean, eta_std)
  state = np.array([[theta], [vtheta], [eta]])
  u = -K @ state
  u = np.clip(u, -s.max_torque, s.max_torque)/s.max_torque
  data = np.concatenate([state, u], axis=0)
  dataset.append(data)

dataset = np.array(dataset).reshape(size_dataset, s.nx + s.nu)
np.random.shuffle(dataset)

column_labels = ['theta', 'vtheta', 'eta', 'u']
df  = pd.DataFrame(dataset, columns=column_labels)
df.to_csv('dataset.csv', index=False)