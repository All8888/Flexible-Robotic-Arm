import numpy as np

# --------------------------------------------------------------------- +
#   WARNING!!! Do not Run both taks's related functions simultaneously  |
# - Guide: ------------------------------------------------------------ +
# For task 1: Set                                                       |
#                   Solve_Smooth = True                                 |
#                   DO_SMOOTH_PLOT = True                               |
#                   DO_TRACK_TRAJECTORY = True                          |
#                   SOLVE_MPC_T1 = True                                 |
#                   DO_MPC_TASK1_PLOT = True                            |
#                                    activating plots flags is optional |
# --------------------------------------------------------------------- +
# to solve open_loop or closed_loop change their corresponding flags    |
# for armijo plots or descent plot change their corresponding flags     |
# --------------------------------------------------------------------- +
# For task 2: Set                                                       |
#                   SOLVE_QUASI_STATIC = True                           |
#                   DO_QUASI_STATIC_PLOT = True                         |
#                   DO_QS_TRACK_PLOT = True                             |
#                   SOLVE_QS_LQR = True                                 |
#                   SOLVE_MPC_TASK2 = True                              |
#                   DO_MPC_TASK2_PLOT = TRUE                            |
#                                    activating plots flags is optional |
# ---------------------------------------------------------------------/


############# States and Inputs #####
ns = 4                              #
ni = 1                              #
#####################################



############# Time and steps ########
tf = 20
dt = 1e-2
TD = int(tf/dt)                     #
#####################################



############# Initial EQ points #####
theta1_eq1 = 0
theta1_eq2 = 45
#####################################



############# System Parameters #####
m1 = 1.5                            #
m2 = 1.5                            #
l1 = 2                              #
l2 = 2                              #
r1 = 1                              #
r2 = 1                              #
I1 = 2                              #
I2 = 2                              #
f1 = 0.1                            #
f2 = 0.1                            #
g = 9.81                            #
#####################################



############# Newton & Armijo #######
max_iters = 10
armijo_max_iters = 200
stepsize_0 = 1
cc = 0.4
beta = 0.7
stopping_criteria = 1e-16
#
# ---- Choose Solver approach       #
open_loop = True
# if open_loop = False              #
# solver goes with closed_loop      #
#####################################



############# Cost Weights ##########
C_QQt = 10
C_RRt = 1
QQt = C_QQt * np.diag([10,10,1,1])  #
RRt = C_RRt * np.eye(ni)            #
QQT = QQt                           #
#####################################



############# Solver Flags ##########
# ---------- Task 1
SOLVE_SMOOTH = False
SOLVE_MPC_T1 = False

# ---------- Task 2
SOLVE_QUASI_STATIC = True
SOLVE_QS_LQR = False
SOLVE_MPC_TASK2 = False



############# Plotting flags ########
# ---------- Task 1
DO_SMOOTH_PLOT = False
DO_TRACK_TRAJECTORY = False
DO_MPC_TASK1_PLOT = False

# ---------- Task 2
DO_QUASI_STATIC_PLOT = True
DO_QS_TRACK_PLOT = False
DO_QS_LQR_PLOT = False
DO_MPC_TASK2_PLOT = False



# ---------- General
DO_ARMIJO_PLOT = False
DO_DESCENT_PLOT = False


############ Animation flags ########
MPC_ANIMATION = False
LQR_ANIMATION = False