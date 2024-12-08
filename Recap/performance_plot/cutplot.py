import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

offset = 0

arr = np.load('update_rates_mean.npy')[offset:]
value_grid = np.arange(offset, offset + len(arr)) / (len(arr) + offset)

ROAs = np.load('ROAs.npy')[offset:]

arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * (np.max(ROAs) - np.min(ROAs)) + np.min(ROAs)
arr_smooth = gaussian_filter1d(arr, sigma=2)

comp_save = -arr + (np.max(arr) + np.min(arr))
comp_save_smooth = gaussian_filter1d(comp_save, sigma=2)


plt.plot(value_grid, ROAs, label='ROAs')
# plt.plot(value_grid, arr_smooth, label='Update rates mean')
# plt.plot(value_grid, arr, label='Update rates mean')
plt.plot(value_grid, comp_save_smooth, label='Computational save smooth')
plt.plot(value_grid, comp_save, label='Computational save')
plt.axvline(x=value_grid[13], color='r', linestyle='--', label='Index 3')
plt.axvline(x=value_grid[36], color='r', linestyle='--', label='Index 3')
plt.axvline(x=value_grid[46], color='r', linestyle='--', label='Index 3')
plt.xlabel('Lambda')
plt.ylabel('Update rate')
plt.legend()
plt.grid(True)
plt.show()

