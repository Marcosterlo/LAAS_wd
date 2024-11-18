import numpy as np
import matplotlib.pyplot as plt

def ellipsoid_plot_3D(P, plot=True, ax=None, color=None, legend=None):

  if color is None:
    color = 'r'

  eigvals, eigvecs = np.linalg.eigh(P)
  axis_length = 1 / np.sqrt(eigvals)
  
  phi = np.linspace(0, 2 * np.pi, 100)
  theta = np.linspace(0, np.pi, 100)
  phi, theta = np.meshgrid(phi, theta)

  x = np.sin(theta) * np.cos(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(theta)
  
  unit_sphere = np.stack((x, y, z), axis=-1)
  ellipsoid_points = unit_sphere @ np.diag(axis_length) @ eigvecs.T
  
  x_ellipsoid = ellipsoid_points[:, :, 0] * 180 / np.pi
  y_ellipsoid = ellipsoid_points[:, :, 1]
  z_ellipsoid = ellipsoid_points[:, :, 2]
  
  if ax is None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    if legend:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=3, cstride=4, color=color, alpha=0.4, linewidth=0, label=legend)
    else:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=3, cstride=4, color=color, alpha=0.4, linewidth=0)

    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('V (rad/s)')
    ax.set_zlabel('Integrator state')
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    if plot:
      plt.show()
    else:
      return fig, ax
  else:
    if legend:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=4, cstride=4, color='r', alpha=0.4, linewidth=0, label=legend)
    else:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=4, cstride=4, color='r', alpha=0.4, linewidth=0)
    return ax
  
  
if __name__ == '__main__':

  P = np.load('P.npy')
  
  fig, ax = ellipsoid_plot_3D(P, False)
  ax.plot([0], [0], [0], marker='o', markersize=5, color='b')
  plt.show()