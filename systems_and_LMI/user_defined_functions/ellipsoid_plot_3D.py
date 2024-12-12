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

def ellipsoid_plot_2D_projections(P, plane='xy', offset=0, ax=None, color=None, legend=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if color is None:
        color = 'r'

    if plane == 'xy':
        indices = [0, 1]
        xlabel, ylabel = 'X', 'Y'
    elif plane == 'xz':
        indices = [0, 2]
        xlabel, ylabel = 'X', 'Z'
    elif plane == 'yz':
        indices = [1, 2]
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'.")

    P_sub = P[np.ix_(indices, indices)]
    eigvals, eigvecs = np.linalg.eigh(P_sub)
    axis_length = 1 / np.sqrt(eigvals)

    # Generate a unit circle and transform it into the ellipsoid's projection
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    projection = unit_circle @ np.diag(axis_length) @ eigvecs.T

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if plane == 'xz' or plane == 'xy':
      mul = 180/np.pi
    else:
      mul = 1
    if plane == 'xy':
      ax.plot(projection[:, 0] * mul, projection[:, 1], offset, color=color, label=legend)
    elif plane == 'xz':
      ax.plot(projection[:, 0] * mul, offset, projection[:, 1], color=color, label=legend)
    elif plane == 'yz':
      ax.plot(offset, projection[:, 0], projection[:, 1], color=color, label=legend)

    if legend:
        ax.legend()

    if ax is None:
        plt.show()
  
if __name__ == '__main__':

  P = np.load('P.npy')
  
  fig, ax = ellipsoid_plot_3D(P, False)
  ax.plot([0], [0], [0], marker='o', markersize=5, color='b')
  plt.show()