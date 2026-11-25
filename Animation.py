import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import Constants

#Animation
def animate_mpc_theta1(xx_star, xx_mpc, dt):
    """
    Animates Theta 1 (rad) over time for MPC tracking.

    Args:
        xx_star: Reference trajectory. Shape: (ns, TT)
        xx_mpc: MPC tracking trajectory. Shape: (ns, TT)
        dt: Time step (scalar).
    """
    

    theta1_ref = xx_star[0, :]
    theta1_mpc = xx_mpc[0, :]
    time = np.arange(len(theta1_ref)) * dt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, len(theta1_ref) * dt)
    ax.set_ylim(min(theta1_ref) - 0.5, max(theta1_ref) + 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Theta 1 (rad)')
    ax.set_title('MPC Tracking: Theta 1 (rad)')
    ax.grid(True)

    #Reference
    ax.plot(time, theta1_ref, 'g--', label='Reference Theta 1')

    #Initialize
    mpc_dot, = ax.plot([], [], 'ro', label='MPC Tracking')
    mpc_trail, = ax.plot([], [], 'r-', alpha=0.5, label='MPC Trail')
    def init():
        mpc_dot.set_data([], [])
        mpc_trail.set_data([], [])
        return mpc_dot, mpc_trail

    def update(frame):
        if frame < len(theta1_mpc):  # Ensure frame does not exceed array bounds
            mpc_dot.set_data(time[frame], theta1_mpc[frame])
            mpc_trail.set_data(time[:frame], theta1_mpc[:frame])
        return mpc_dot, mpc_trail

    #Animation
    frames = len(theta1_ref)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=dt*1000)
    ax.legend()

    #Save&illustration
    ani.save('mpc_theta1_animation.gif', writer='pillow')
    plt.show()

def animate_mpc_theta2(xx_star, xx_mpc, dt):
    """
    Animates Theta 2 (rad) over time for MPC tracking.

    Args:
        xx_star: Reference trajectory. Shape: (ns, TT)
        xx_mpc: MPC tracking trajectory. Shape: (ns, TT)
        dt: Time step (scalar).
    """

    theta2_ref = xx_star[1, :]
    theta2_mpc = xx_mpc[1, :]
    time = np.arange(len(theta2_ref)) * dt
    #Figures
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, len(theta2_ref) * dt)
    ax.set_ylim(min(theta2_ref) - 0.5, max(theta2_ref) + 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Theta 2 (rad)')
    ax.set_title('MPC Tracking: Theta 2 (rad)')
    ax.grid(True)

    #Reference
    ax.plot(time, theta2_ref, 'g--', label='Reference Theta 2')
    #Initialization
    mpc_dot, = ax.plot([], [], 'ro', label='MPC Tracking')
    mpc_trail, = ax.plot([], [], 'r-', alpha=0.5, label='MPC Trail')

    def init():
        mpc_dot.set_data([], [])
        mpc_trail.set_data([], [])
        return mpc_dot, mpc_trail

    def update(frame):
        if frame < len(theta2_mpc):
            mpc_dot.set_data(time[frame], theta2_mpc[frame])
            mpc_trail.set_data(time[:frame], theta2_mpc[:frame])
        return mpc_dot, mpc_trail

    #Animation
    frames = len(theta2_ref)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=dt*1000)
    ax.legend()

    #Save&Illustration
    ani.save('mpc_theta2_animation.gif', writer='pillow')
    plt.show()


def animate_mpc_omega1(xx_star, xx_mpc, dt):
    """
    Animates Omega 1 (rad/s) over time for MPC tracking.

    Args:
        xx_star: Reference trajectory. Shape: (ns, TT)
        xx_mpc: MPC tracking trajectory. Shape: (ns, TT)
        dt: Time step (scalar).
    """

    omega1_ref = xx_star[2, :]
    omega1_mpc = xx_mpc[2, :]
    time = np.arange(len(omega1_ref)) * dt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, len(omega1_ref) * dt)
    ax.set_ylim(min(omega1_ref) - 0.5, max(omega1_ref) + 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Omega 1 (rad/s)')
    ax.set_title('MPC Tracking: Omega 1 (rad/s)')
    ax.grid(True)
    ax.plot(time, omega1_ref, 'g--', label='Reference Omega 1')
    mpc_dot, = ax.plot([], [], 'ro', label='MPC Tracking')
    mpc_trail, = ax.plot([], [], 'r-', alpha=0.5, label='MPC Trail')

    def init():
        mpc_dot.set_data([], [])
        mpc_trail.set_data([], [])
        return mpc_dot, mpc_trail

    def update(frame):
        if frame < len(omega1_mpc):
            mpc_dot.set_data(time[frame], omega1_mpc[frame])
            mpc_trail.set_data(time[:frame], omega1_mpc[:frame])
        return mpc_dot, mpc_trail

    frames = len(omega1_ref)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=dt*1000)
    ax.legend()
    ani.save('mpc_omega1_animation.gif', writer='pillow')
    plt.show()


def animate_mpc_omega2(xx_star, xx_mpc, dt):
    """
    Animates Omega 2 (rad/s) over time for MPC tracking.

    Args:
        xx_star: Reference trajectory. Shape: (ns, TT)
        xx_mpc: MPC tracking trajectory. Shape: (ns, TT)
        dt: Time step (scalar).
    """
    omega2_ref = xx_star[3, :]
    omega2_mpc = xx_mpc[3, :]
    time = np.arange(len(omega2_ref)) * dt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, len(omega2_ref) * dt)
    ax.set_ylim(min(omega2_ref) - 0.5, max(omega2_ref) + 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Omega 2 (rad/s)')
    ax.set_title('MPC Tracking: Omega 2 (rad/s)')
    ax.grid(True)
    ax.plot(time, omega2_ref, 'g--', label='Reference Omega 2')
    mpc_dot, = ax.plot([], [], 'ro', label='MPC Tracking')
    mpc_trail, = ax.plot([], [], 'r-', alpha=0.5, label='MPC Trail')

    def init():
        mpc_dot.set_data([], [])
        mpc_trail.set_data([], [])
        return mpc_dot, mpc_trail

    def update(frame):
        if frame < len(omega2_mpc):
            mpc_dot.set_data(time[frame], omega2_mpc[frame])
            mpc_trail.set_data(time[:frame], omega2_mpc[:frame])
        return mpc_dot, mpc_trail

    frames = len(omega2_ref)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=dt*1000)
    ax.legend()
    ani.save('mpc_omega2_animation.gif', writer='pillow')
    plt.show()


def lqr_animation(xx):
    theta1, theta2 = xx[:2]
    l1 = Constants.l1
    l2 = Constants.l2
    dt = Constants.dt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, l1+l2+0.5)
    ax.set_ylim(-(l1+l2+0.5), 2)
    ax.set_aspect('equal')
    ax.grid(True)

    #Initialize pendulum
    line, = ax.plot([], [], 'o-', lw=4, markersize=12)
    trail, = ax.plot([], [], 'r-', alpha=0.5)
    trace_x, trace_y = [], []

    def init():
        line.set_data([], [])
        trail.set_data([], [])
        return line, trail

    def animate(i):
        x1 = l1 * np.sin(theta1[i])
        y1 = -l1 * np.cos(theta1[i])
        x2 = x1 + l2 * np.sin(theta1[i] + theta2[i])
        y2 = y1 - l2 * np.cos(theta1[i] + theta2[i])
        line.set_data([0, x1, x2], [0, y1, y2])
        trace_x.append(x2)
        trace_y.append(y2)
        trail.set_data(trace_x, trace_y)
        return line, trail

    #Animation
    ani = FuncAnimation(
        fig, animate, frames=len(theta1),
        init_func=init, blit=True, interval=dt*100
    )
    ani.save('Tracking.gif', writer='pillow', fps=1/dt)   #Save
    #Illustration
    plt.title("Double Pendulum LQR Trajectory Tracking")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()