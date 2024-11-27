import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 1.0  # Length of the pendulum (meters)
dt = 0.05  # Time step (seconds)

# Example input data
t = np.arange(0, 10, dt)  # Time array
theta = 0.1 * np.cos(2 * np.pi * t)+np.pi  # Example angular position (radians)
vtheta = -0.1 * 2 * np.pi * np.sin(2 * np.pi * t)  # Example angular velocity (rad/s)

# Inverted pendulum animation
def animate_pendulum(theta, theta_eq, L=1.0):
    fig, ax = plt.subplots()
    ax.set_xlim(-L - 0.2, L + 0.2)
    ax.set_ylim(-0.2, L + 0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Inverted Pendulum Animation")
    ax.set_facecolor('whitesmoke')  # Light background for axes
    fig.patch.set_facecolor('lightblue')  # Light blue figure background
    ax.grid(True)  # Add gridlines

    x_ref = L * np.sin(theta_eq)
    y_ref = -L * np.cos(theta_eq)
    ax.plot([0, x_ref], [0, y_ref], 'r--', lw=1, markersize=4)  # Reference line


    # Pendulum components
    line, = ax.plot([], [], 'o-', lw=2, markersize=8)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    eq_line, = ax.plot([], [], 'r--', lw=1, markersize=4)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, eq_line, time_text

    def update(frame):
        x = L * np.sin(theta[frame])
        y = -L * np.cos(theta[frame])
        line.set_data([0, x], [0, y])
        time_text.set_text(f'Time: {frame * dt:.2f} s')
        return line, eq_line, time_text

    ani = FuncAnimation(fig, update, frames=len(theta), init_func=init, blit=True, interval=dt*1000)
    plt.show()

# Call the animation function
if __name__ == '__main__':
  animate_pendulum(theta, 10, L=L)
