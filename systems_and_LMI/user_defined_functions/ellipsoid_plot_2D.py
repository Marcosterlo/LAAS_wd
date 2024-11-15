import numpy as np
import matplotlib.pyplot as plt

def ellipsoid_plot_2D(P, plot=True, ax=None, color=None, legend=None):

  if color is None:
    color = 'r'
    
  eigvals, eigvecs = np.linalg.eigh(P)
  axis_length = 1 / np.sqrt(eigvals)
  
  theta = np.linspace(0, np.pi, 100)

  theta = np.linspace(0, 2 * np.pi, 100)
  x = axis_length[0] * np.cos(theta)
  y = axis_length[1] * np.sin(theta)
  
  ellipsoid_points = np.stack((x, y), axis=-1) @ eigvecs.T
  
  x_ellipsoid = ellipsoid_points[:, 0]
  y_ellipsoid = ellipsoid_points[:, 1]
  
  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 10))
    if legend:
      ax.plot(x_ellipsoid*180/np.pi, y_ellipsoid, color=color, alpha=0.4, label=legend)
    else:
      ax.plot(x_ellipsoid*180/np.pi, y_ellipsoid, color=color, alpha=0.4)

    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('V (rad/s)')
    ax.grid(True)

    if plot:
      plt.show()
    else:
      return fig, ax
  else:
    if legend:
      ax.plot(x_ellipsoid*180/np.pi, y_ellipsoid, color=color, alpha=0.4, label=legend)
    else:
      ax.plot(x_ellipsoid*180/np.pi, y_ellipsoid, color=color, alpha=0.4)
    return ax
  
if __name__ == '__main__':

  P = np.load('P.npy')
  P = P[:2, :2]
  
  fig, ax = ellipsoid_plot_2D(P, False)
  ax.plot([0], [0], marker='o', markersize=5, color='b')
  plt.show()