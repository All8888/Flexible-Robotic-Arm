import matplotlib.pyplot as plt
import numpy as np


# |||||||||||||||||||||||||||| Smooth Trajectory Plot ||||||||||||||||||||||||||||
# --------------------------------------------------------------------------------
def plot_smooth(xx, uu, dt):

    theta1, theta2, theta1_dot, theta2_dot = xx
    hor = len(xx[0])
    time = np.arange(hor) * dt

    # Create a subplot figure
    fig, axes = plt.subplots(3, 2, figsize=(10, 6))

  # Plot theta1
    axes[0, 0].plot(time, theta1)
    axes[0, 0].set_title('Theta 1')
    axes[0, 0].set_xlabel('Time (s)', fontsize=8)
    axes[0, 0].set_ylabel('Theta 1 (rad)')
    axes[0, 0].grid(True)  # Add grid

    # Plot theta2
    axes[0, 1].plot(time, theta2)
    axes[0, 1].set_title('Theta 2')
    axes[0, 1].set_xlabel('Time (s)', fontsize=8)
    axes[0, 1].set_ylabel('Theta 2 (rad)')
    axes[0, 1].grid(True)  # Add grid

    # Plot theta1_dot
    axes[1, 0].plot(time, theta1_dot)
    axes[1, 0].set_title('Theta 1 Dot (Velocity)')
    axes[1, 0].set_xlabel('Time (s)', fontsize=8)
    axes[1, 0].set_ylabel('Theta 1 Dot (rad/s)')
    axes[1, 0].grid(True)  # Add grid

    # Plot theta2_dot
    axes[1, 1].plot(time, theta2_dot)
    axes[1, 1].set_title('Theta 2 Dot (Velocity)')
    axes[1, 1].set_xlabel('Time (s)', fontsize=8)
    axes[1, 1].set_ylabel('Theta 2 Dot (rad/s)')
    axes[1, 1].grid(True)  # Add grid

    # Plot input U
    axes[2, 0].plot(time, uu[0])
    axes[2, 0].set_title('Input U')
    axes[2, 0].set_xlabel('Time (s)', fontsize=8)
    axes[2, 0].set_ylabel('U')
    axes[2, 0].grid(True)  # Add grid

    # Hide the empty subplot
    fig.delaxes(axes[2, 1])

    # Adjust layout
    fig.tight_layout()
    plt.show()


# |||||||||||||||||||||||||||| ARMIO PLOT ||||||||||||||||||||||||||||
# --------------------------------------------------------------------


# |||||||||||||||||||||||||||| Descent PLOT ||||||||||||||||||||||||||
# --------------------------------------------------------------------
def descent_plot(max_iters, delta_u_norm, JJ):
    plt.figure('Descent Plot')
    plt.plot(np.arange(max_iters), delta_u_norm[:max_iters])
    plt.xlabel('Iterations')
    plt.ylabel('Descent Value')
    plt.grid()
    plt.show()

    plt.figure('Cost Plot')
    plt.plot(np.arange(max_iters), JJ[:max_iters])
    plt.xlabel('Iterations')
    plt.ylabel('Cost Value')
    plt.grid()
    plt.show()

# ||||||||||||||||||||||| Track Trajectory PLOT ||||||||||||||||||||||
# --------------------------------------------------------------------
def track_trajectory(xx_star, uu_star, xx_ref, uu_ref, tf, dt):
    
    tt = np.linspace(0, tf, int(tf/dt))

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))


    axes[0, 0].plot(tt, xx_star[0, :], 'r', label='Theta 1 star')
    axes[0, 0].plot(tt, xx_ref[0, :], 'g--', label='Theta 1 ref')
    axes[0, 0].set_ylabel('Theta 1 (rad)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(tt, xx_star[1, :], 'r', label='Theta 2 star')
    axes[0, 1].plot(tt, xx_ref[1, :], 'g--', label='Theta 2 ref')
    axes[0, 1].set_ylabel('Theta 2 (rad)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(tt, xx_star[2, :], 'r', label='Theta 1 dot star')
    axes[1, 0].plot(tt, xx_ref[2, :], 'g--', label='Theta 1 dot ref')
    axes[1, 0].set_ylabel('Omega 1 (rad/s)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(tt, xx_star[3, :], 'r', label='Theta 2 dot star')
    axes[1, 1].plot(tt, xx_ref[3, :], 'g--', label='Theta 2 dot ref')
    axes[1, 1].set_ylabel('Omega 2 (rad/s)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    axes[2, 0].plot(tt, uu_star[0, :], 'r', label='U star')
    axes[2, 0].plot(tt, uu_ref[0, :], 'g--', label='U ref')
    axes[2, 0].set_ylabel('Input U')
    axes[2, 0].grid()
    axes[2, 0].legend()


    fig.delaxes(axes[2,1])

    plt.tight_layout(pad=1.5)
    plt.show()

# ||||||||||||||||||| Quasi-Static Trajectory PLOT |||||||||||||||||||
# --------------------------------------------------------------------
def quasi_static_trajectory(xx, uu, tf, dt):
    tt = int(tf/dt)
    time = np.linspace(0, tf, tt)

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes[0, 0].plot(time, xx[0, :], 'b', label='Theta 1 ref', linewidth = 2.2)
    axes[0, 0].set_ylabel('Theta 1 (rad)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(time, xx[1, :], 'b', label='Theta 2 ref', linewidth = 2.2)
    axes[0, 1].set_ylabel('Theta 2 (rad)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(time, xx[2, :], 'r', label='Theta 1 dot ref', linewidth = 2.2)
    axes[1, 0].set_ylabel('Omega 1 (rad/s)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(time, xx[3, :], 'r', label='Theta 2 dot ref', linewidth = 2.2)
    axes[1, 1].set_ylabel('Omega 2 (rad/s)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    axes[2, 0].plot(time, uu[0,:], 'g', label='U ref', linewidth = 2.2)
    axes[2, 0].set_ylabel('Input U')
    axes[2, 0].grid()
    axes[2, 0].legend()

    fig.delaxes(axes[2,1])

    plt.tight_layout(pad=1.5)
    plt.show()

# |||||||||||||||  Track Quasi-Static Trajectory PLOT ||||||||||||||||
# --------------------------------------------------------------------
def qs_track_trajectory(xx_star, uu_star, xx_ref, uu_ref, tf, dt):
    tt = np.linspace(0, tf, int(tf/dt))

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))


    axes[0, 0].plot(tt, xx_star[0, :], 'r', label='Theta 1 star')
    axes[0, 0].plot(tt, xx_ref[0, :], 'g--', label='Theta 1 ref')
    axes[0, 0].set_ylabel('Theta 1 (rad)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(tt, xx_star[1, :], 'r', label='Theta 2 star')
    axes[0, 1].plot(tt, xx_ref[1, :], 'g--', label='Theta 2 ref')
    axes[0, 1].set_ylabel('Theta 2 (rad)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(tt, xx_star[2, :], 'r', label='Theta 1 dot star')
    axes[1, 0].plot(tt, xx_ref[2, :], 'g--', label='Theta 1 dot ref')
    axes[1, 0].set_ylabel('Omega 1 (rad/s)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(tt, xx_star[3, :], 'r', label='Theta 2 dot star')
    axes[1, 1].plot(tt, xx_ref[3, :], 'g--', label='Theta 2 dot ref')
    axes[1, 1].set_ylabel('Omega 2 (rad/s)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    axes[2, 0].plot(tt, uu_star[0, :], 'r', label='U star')
    axes[2, 0].plot(tt, uu_ref[0, :], 'g--', label='U ref')
    axes[2, 0].set_ylabel('Input U')
    axes[2, 0].grid()
    axes[2, 0].legend()


    fig.delaxes(axes[2,1])

    plt.tight_layout(pad=1.5)
    plt.show()

# ||||||||||||||||||| MPC Track Smooth Reference |||||||||||||||||||||
# --------------------------------------------------------------------
def mpc_track_task1(xx, uu, xx_star, uu_star, dt):
    """
    Generates a plot comparing the MPC-controlled trajectory with a reference trajectory 
    for a double robotic arm system, including error visualization.
    """

    time = xx.shape[1]
    xx_star = xx_star[:, :time]
    uu_star = uu_star[:, :time]

    tf_mpc = time * dt
    tt_mpc = np.linspace(0, tf_mpc, time)

    fig, axes = plt.subplots(3,2, figsize=(12, 12))
    axes[0, 0].plot(tt_mpc, xx[0, :], 'r', label='Theta 1 mpc')
    axes[0, 0].plot(tt_mpc, xx_star[0, :], 'g--', label='Theta 1 star')
    axes[0, 0].plot(tt_mpc, xx[0, :] - xx_star[0, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[0, 0].set_ylabel('Theta 1 (rad)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(tt_mpc, xx[1, :], 'r', label='Theta 2 mpc')
    axes[0, 1].plot(tt_mpc, xx_star[1, :], 'g--', label='Theta 2 star')
    axes[0, 1].plot(tt_mpc, xx[1, :] - xx_star[1, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[0, 1].set_ylabel('Theta 2 (rad)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(tt_mpc, xx[2, :], 'r', label='Theta 1 dot mpc')
    axes[1, 0].plot(tt_mpc, xx_star[2, :], 'g--', label='Theta 1 dot star')
    axes[1, 0].plot(tt_mpc, xx[2, :] - xx_star[2, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[1, 0].set_ylabel('Omega 1 (rad/s)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(tt_mpc, xx[3, :], 'r', label='Theta 2 dot mpc')
    axes[1, 1].plot(tt_mpc, xx_star[3, :], 'g--', label='Theta 2 dot star')
    axes[1, 1].plot(tt_mpc, xx[3, :] - xx_star[3, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[1, 1].set_ylabel('Omega 2 (rad/s)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    axes[2, 0].plot(tt_mpc, uu[0, :], 'r', label='U mpc')
    axes[2, 0].plot(tt_mpc, uu_star[0, :], 'g--', label='U star')
    axes[2, 0].plot(tt_mpc, uu[0, :] - uu_star[0, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[2, 0].set_ylabel('Input U')
    axes[2, 0].grid()
    axes[2, 0].legend()

    fig.suptitle("MPC tracked Smooth Trajectory", fontsize=14)

    fig.delaxes(axes[2,1])

    plt.tight_layout(pad=1.5)

    plt.show()

# |||||||||||||||| MPC Track Quasi Static Reference ||||||||||||||||||
# --------------------------------------------------------------------
def mpc_track_task2(xx, uu, xx_star, uu_star, dt):
    """
    Generates a plot comparing the MPC-controlled trajectory with a reference trajectory 
    for a double robotic arm system, including error visualization.
    """

    time = xx.shape[1]
    xx_star = xx_star[:, :time]
    uu_star = uu_star[:, :time]

    tf_mpc = time * dt
    tt_mpc = np.linspace(0, tf_mpc, time)

    fig, axes = plt.subplots(3,2, figsize=(12, 12))
    axes[0, 0].plot(tt_mpc, xx[0, :], 'r', label='Theta 1 mpc')
    axes[0, 0].plot(tt_mpc, xx_star[0, :], 'g--', label='Theta 1 star (Task2)')
    axes[0, 0].plot(tt_mpc, xx[0, :] - xx_star[0, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[0, 0].set_ylabel('Theta 1 (rad)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(tt_mpc, xx[1, :], 'r', label='Theta 2 mpc')
    axes[0, 1].plot(tt_mpc, xx_star[1, :], 'g--', label='Theta 2 star (Task2)')
    axes[0, 1].plot(tt_mpc, xx[1, :] - xx_star[1, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[0, 1].set_ylabel('Theta 2 (rad)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(tt_mpc, xx[2, :], 'r', label='Theta 1 dot mpc')
    axes[1, 0].plot(tt_mpc, xx_star[2, :], 'g--', label='Theta 1 dot star (Task2)')
    axes[1, 0].plot(tt_mpc, xx[2, :] - xx_star[2, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[1, 0].set_ylabel('Omega 1 (rad/s)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(tt_mpc, xx[3, :], 'r', label='Theta 2 dot mpc')
    axes[1, 1].plot(tt_mpc, xx_star[3, :], 'g--', label='Theta 2 dot star (Task2)')
    axes[1, 1].plot(tt_mpc, xx[3, :] - xx_star[3, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[1, 1].set_ylabel('Omega 2 (rad/s)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    axes[2, 0].plot(tt_mpc, uu[0, :], 'r', label='U mpc')
    axes[2, 0].plot(tt_mpc, uu_star[0, :], 'g--', label='U star (Task2)')
    axes[2, 0].plot(tt_mpc, uu[0, :] - uu_star[0, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[2, 0].set_ylabel('Input U')
    axes[2, 0].grid()
    axes[2, 0].legend()

    fig.suptitle("MPC tracked Quasi Static Trajectory", fontsize=14)

    fig.delaxes(axes[2,1])

    plt.tight_layout(pad=1.5)

    plt.show()


# |||||||||||||||| LQR Track Quasi Static Reference ||||||||||||||||||
# --------------------------------------------------------------------
def qs_lqr_track(xx, uu, xx_star, uu_star, dt):
    """
    Generates a plot comparing the LQR-controlled trajectory with a reference trajectory 
    for a double robotic arm system, including error visualization.
    """
    uu[:, -1] = uu[:,-2]
    time = xx.shape[1]
    xx_star = xx_star[:, :time]
    uu_star = uu_star[:, :time]

    tf_mpc = time * dt
    tt_mpc = np.linspace(0, tf_mpc, time)

    fig, axes = plt.subplots(3,2, figsize=(12, 12))
    axes[0, 0].plot(tt_mpc, xx[0, :], 'r', label='Theta 1 LQR')
    axes[0, 0].plot(tt_mpc, xx_star[0, :], 'g--', label='Theta 1 star (Task2)')
    axes[0, 0].plot(tt_mpc, xx[0, :] - xx_star[0, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[0, 0].set_ylabel('Theta 1 (rad)')
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(tt_mpc, xx[1, :], 'r', label='Theta 2 LQR')
    axes[0, 1].plot(tt_mpc, xx_star[1, :], 'g--', label='Theta 2 star (Task2)')
    axes[0, 1].plot(tt_mpc, xx[1, :] - xx_star[1, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[0, 1].set_ylabel('Theta 2 (rad)')
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].plot(tt_mpc, xx[2, :], 'r', label='Theta 1 dot LQR')
    axes[1, 0].plot(tt_mpc, xx_star[2, :], 'g--', label='Theta 1 dot star (Task2)')
    axes[1, 0].plot(tt_mpc, xx[2, :] - xx_star[2, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[1, 0].set_ylabel('Omega 1 (rad/s)')
    axes[1, 0].grid()
    axes[1, 0].legend()

    axes[1, 1].plot(tt_mpc, xx[3, :], 'r', label='Theta 2 dot LQR')
    axes[1, 1].plot(tt_mpc, xx_star[3, :], 'g--', label='Theta 2 dot star (Task2)')
    axes[1, 1].plot(tt_mpc, xx[3, :] - xx_star[3, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[1, 1].set_ylabel('Omega 2 (rad/s)')
    axes[1, 1].grid()
    axes[1, 1].legend()

    axes[2, 0].plot(tt_mpc, uu[0, :], 'r', label='U LQR')
    axes[2, 0].plot(tt_mpc, uu_star[0, :], 'g--', label='U star (Task2)')
    axes[2, 0].plot(tt_mpc, uu[0, :] - uu_star[0, :], 'k--', label = 'Error', linewidth = 0.7, alpha = 0.7)
    axes[2, 0].set_ylabel('Input U')
    axes[2, 0].grid()
    axes[2, 0].legend()

    fig.suptitle("LQR tracked Quasi Static Trajectory", fontsize=14)

    fig.delaxes(axes[2,1])

    plt.tight_layout(pad=1.5)

    plt.show()