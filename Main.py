import time
import psutil
import numpy as np

import Constants
import Functions as funcs
import Plots as plt
from LQR_tracking import lqr_tracking
from MPC_task1 import mpc_taks1
from MPC_task2 import mpc_task2
import Animation as anim

# --------- Resources Monitoring:
process = psutil.Process()
start_memory = process.memory_info().rss
start_cpu = psutil.cpu_percent(interval=0.1)
start_time = time.time()
# -------------------------------

ns = Constants.ns
ni = Constants.ni
tf = Constants.tf
dt = Constants.dt
TD = Constants.TD


DO_SMOOTH_PLOT = Constants.DO_SMOOTH_PLOT
DO_QS_PLOT = Constants.DO_QUASI_STATIC_PLOT
SOLVE_SMOOTH = Constants.SOLVE_SMOOTH
SOLVE_QS = Constants.SOLVE_QUASI_STATIC

print("WARNING, Make sure you are not running task 1 and task 2 flags simultaneously!!!")
print("         To make changes, use Constants Module.\n")
###############################################################################
###################################### Part 1: Smooth Trajectory and qusi_static
theta1_eq1 = Constants.theta1_eq1
theta1_eq2 = Constants.theta1_eq2


# ........................... Smooth Trajectory
xx_ref, uu_ref = funcs.smooth_trajectory(theta1_eq1, theta1_eq2, TD, ns , ni)
if DO_SMOOTH_PLOT:
    plot_smooth_trajectory = plt.plot_smooth(xx_ref, uu_ref, dt)


# ........................... Quasi - Static Trajectory
xx_ref_qs, uu_ref_qs = funcs.quasi_static_trajectory(theta1_eq1, theta1_eq2, tf, dt, ni)
if DO_QS_PLOT:
    plt.quasi_static_trajectory(xx_ref_qs, uu_ref_qs, tf, dt)

###############################################################################
################################# Part 2: Newton and Armijo of smooth reference
xx_init = np.repeat(xx_ref[:,0].reshape(-1,1), TD, axis = 1)
uu_init = np.repeat(uu_ref[:,0].reshape(-1,1), TD, axis = 1)

max_iters = Constants.max_iters
armijo_max_iters = Constants.armijo_max_iters
stepsize_0 = Constants.stepsize_0
cc = Constants.cc
beta = Constants.beta
QQ = Constants.QQt
RR = Constants.RRt
QQT = Constants.QQt
stopping_criteria = Constants.stopping_criteria
open_loop = Constants.open_loop

if SOLVE_SMOOTH:
    xx_star, uu_star = funcs.newton_method(xx_init,uu_init, xx_ref, uu_ref, ns, ni, TD, max_iters, QQ, RR, QQT, stopping_criteria, open_loop)

########################### Part 2: Newton and Armijo of quasi-static reference
xx_init_quasi = np.repeat(xx_ref_qs[:,0].reshape(-1,1), TD, axis = 1)
uu_init_quasi = np.repeat(uu_ref_qs[:,0].reshape(-1,1), TD, axis = 1)


if SOLVE_QS:
    QQ_s = np.diag([10000, 10000, 0.01, 0.01])
    RR_s = 10 * RR
    QQT_s = QQ_s
    # xx_ref_qs[2,:] = 0
    # xx_ref_qs[3,:] = 0
    xx_star_quasi, uu_star_quasi = funcs.newton_method(xx_init_quasi,uu_init_quasi, xx_ref_qs, uu_ref_qs, ns, ni, TD, max_iters, QQ_s, RR_s, QQT_s, stopping_criteria, open_loop)
    

###############################################################################
################################## Part 3: Track Trajectory of smooth reference
DO_TRACK_TRAJECTORY = Constants.DO_TRACK_TRAJECTORY
DO_QS_TRACK_TRAJECTORY = Constants.DO_QS_TRACK_PLOT
# ---------------------------------------------------------- Task 1:
if DO_TRACK_TRAJECTORY:
    plt.track_trajectory(xx_star, uu_star, xx_ref, uu_ref, tf, dt)

# ---------------------------------------------------------- Task 2:
if DO_QS_TRACK_TRAJECTORY:
    plt.qs_track_trajectory(xx_star_quasi, uu_star_quasi, xx_ref_qs, uu_ref_qs, tf, dt)



###############################################################################
############################### Part 4: LQR tracking of quasi static ( Task 2 )
SOLVE_QS_LQR = Constants.SOLVE_QUASI_STATIC
DO_QS_LQR_PLOT = Constants.DO_QS_LQR_PLOT

if SOLVE_QS_LQR:
    x_0 = xx_star_quasi[:, 0]
    perturbation = np.array([-np.pi/36, np.pi/36, 0, 0])
    x_0 += perturbation
    xx_tracked_lqr, uu_tracked_lqr = lqr_tracking(xx_star_quasi, uu_star_quasi, x_0, TD, QQ, RR, QQT)

    if DO_QS_LQR_PLOT:
        plt.qs_lqr_track(xx_tracked_lqr, uu_tracked_lqr, xx_star_quasi, uu_star_quasi, dt)


###############################################################################
################################## Part 5: MPC tracking
# ---------------------------------------------------------- Task 1:
SOLVE_MPC_TASK1 = Constants.SOLVE_MPC_T1
DO_MPC_T1_PLOT = Constants.DO_MPC_TASK1_PLOT
if SOLVE_MPC_TASK1:
    T_pred = 5 # Prediction Time Horizon
    T_pred = 3
    perturbation = np.array([-np.pi/36, np.pi/36, 0, 0])    # Pertubation applied to initial state

    xx_lb = np.array([-np.pi, -np.pi, -5, -5])               # States Lower Bound
    xx_ub = np.array([np.pi, np.pi, 5, 5])                   # States Upper Bound

    uu_lb = np.array([-40])                                  # Inputs Lower Bound
    uu_ub = np.array([40])                                   # Inputs Upper Bound

    xx_mpc, uu_mpc = mpc_taks1(xx_star, uu_star, T_pred, perturbation, QQ, RR, QQT, xx_lb, xx_ub, uu_lb, uu_ub)
    uu_mpc[:, -1] = uu_mpc[:, -2]
    if DO_MPC_T1_PLOT:
        plt.mpc_track_task1(xx_mpc, uu_mpc, xx_star, uu_star, dt)

# ---------------------------------------------------------- Task 2:
SOLVE_MPC_TASK2 = Constants.SOLVE_MPC_TASK2
DO_MPC_TASK2_PLOT = Constants.DO_MPC_TASK2_PLOT

# takes xx_star_quasi and uu_star_quasi as input
if SOLVE_MPC_TASK2:
    T_pred = 5 # Prediction Time Horizon
    T_pred = 3
    perturbation = np.array([-np.pi/36, np.pi/36, 0, 0])    # Pertubation applied to initial state

    xx_mpc_quasi, uu_mpc_quasi = mpc_task2(xx_star_quasi, uu_star_quasi, T_pred, perturbation, QQ, RR, QQT)
    uu_mpc_quasi[:, -1] = uu_mpc_quasi[:, -2]
    
    if DO_MPC_TASK2_PLOT:
        plt.mpc_track_task2(xx_mpc_quasi, uu_mpc_quasi, xx_star_quasi, uu_star_quasi, dt)


###############################################################################
################################## Animation (Default: Flags = True)
LQR_ANIMATION = Constants.LQR_ANIMATION
MPC_ANIMATION = Constants.MPC_ANIMATION

if MPC_ANIMATION:
    anim.animate_mpc_theta1(xx_star_quasi, xx_mpc_quasi, dt)
    anim.animate_mpc_theta2(xx_star_quasi, xx_mpc_quasi, dt)
    anim.animate_mpc_omega1(xx_star_quasi, xx_mpc_quasi, dt)
    anim.animate_mpc_omega2(xx_star_quasi, xx_mpc_quasi, dt)

if LQR_ANIMATION:
    anim.lqr_animation(xx_tracked_lqr)

# --------- Resources Monitoring:
end_memory = process.memory_info().rss
end_cpu = psutil.cpu_percent(interval=0.1)
end_time = time.time()
memory_used = (end_memory - start_memory) / (1024 * 1024)
cpu_used = end_cpu - start_cpu
time_elapsed = end_time - start_time
# Results
print(f"Memory used: {memory_used:.2f} MB")
print(f"CPU usage change: {cpu_used:.2f}%")
print(f"Total execution time: {time_elapsed:.2f} seconds")
# -------------------------------