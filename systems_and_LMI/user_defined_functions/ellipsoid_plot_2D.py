import numpy as np
import matplotlib.pyplot as plt

def ellipsoid_plot_2D(P, plot=True, ax=None, color=None, offset=None, legend=None):

  if color is None:
    color = 'r'
    
  eigvals, eigvecs = np.linalg.eigh(P)
  axis_length = 1 / np.sqrt(eigvals)
  
  theta = np.linspace(0, np.pi, 100)

  theta = np.linspace(0, 2 * np.pi, 100)
  x = axis_length[0] * np.cos(theta)
  y = axis_length[1] * np.sin(theta)
  
  ellipsoid_points = np.stack((x, y), axis=-1) @ eigvecs.T
  
  x_ellipsoid = ellipsoid_points[:, 0] * 180 / np.pi
  y_ellipsoid = ellipsoid_points[:, 1]
  
  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 10))
    if legend:
      ax.plot(x_ellipsoid, y_ellipsoid, color=color, alpha=0.4, label=legend)
    else:
      ax.plot(x_ellipsoid, y_ellipsoid, color=color, alpha=0.4)

    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('V (rad/s)')
    ax.grid(True)
  

    if plot:
      plt.show()
    else:
      return fig, ax
  else:
    if legend:
      if offset is not None:
        ax.plot(x_ellipsoid, y_ellipsoid, offset, color=color, alpha=0.4, label=legend)
      else:
        ax.plot(x_ellipsoid, y_ellipsoid, color=color, alpha=0.4, label=legend)
    else:
      if offset is not None:
        ax.plot(x_ellipsoid, y_ellipsoid, offset, color=color, alpha=0.4)
      else:
        ax.plot(x_ellipsoid, y_ellipsoid, color=color, alpha=0.4)
    return ax


def ellipsoid_plot_XZ_2D(P, plot=True, ax=None, color=None, offset=None, legend=None):

  if color is None:
    color = 'r'
    
  eigvals, eigvecs = np.linalg.eigh(P)
  axis_length = 1 / np.sqrt(eigvals)
  
  theta = np.linspace(0, np.pi, 100)

  theta = np.linspace(0, 2 * np.pi, 100)
  x = axis_length[0] * np.cos(theta)
  z = axis_length[2] * np.sin(theta)
  
  eigvecs = np.delete(np.delete(eigvecs, 1, axis=0), 1, axis=1)
  ellipsoid_points = np.stack((x, z), axis=-1) @ eigvecs.T
  
  x_ellipsoid = ellipsoid_points[:, 0] * 180 / np.pi
  z_ellipsoid = ellipsoid_points[:, 1]
  
  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 10))
    if legend:
      ax.plot(x_ellipsoid, z_ellipsoid, color=color, alpha=0.4, label=legend)
    else:
      ax.plot(x_ellipsoid, z_ellipsoid, color=color, alpha=0.4)

    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('V (rad/s)')
    ax.grid(True)
  

    if plot:
      plt.show()
    else:
      return fig, ax
  else:
    if legend:
      if offset is not None:
        offset = np.ones(len(x_ellipsoid)) * offset
        ax.plot(x_ellipsoid, offset, z_ellipsoid, offset, color=color, alpha=0.4, label=legend)
      else:
        ax.plot(x_ellipsoid, z_ellipsoid, color=color, alpha=0.4, label=legend)
    else:
      if offset is not None:
        offset = np.ones(len(x_ellipsoid)) * offset
        ax.plot(x_ellipsoid, offset, z_ellipsoid, offset, color=color, alpha=0.4)
      else:
        ax.plot(x_ellipsoid, z_ellipsoid, color=color, alpha=0.4)
    return ax
  
if __name__ == '__main__':

  P = np.load('P.npy')
  P = P[:2, :2]
  
  fig, ax = ellipsoid_plot_2D(P, False)
  ax.plot([0], [0], marker='o', markersize=5, color='b')
  plt.show()
  
if __name__ == '__main__':

  P = np.load('P.npy')
  P = P[:2, :2]
  
  fig, ax = ellipsoid_plot_2D(P, False)
  ax.plot([0], [0], marker='o', markersize=5, color='b')
  plt.show()