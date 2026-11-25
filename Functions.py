import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from time import sleep

import Constants
import Dynamics as dyn
import Costs as cst
from LQR import affine_lqr
from Plots import descent_plot



m1 = Constants.m1
m2 = Constants.m2
l1 = Constants.l1
l2 = Constants.l2
r1 = Constants.r1
r2 = Constants.r2
I1 = Constants.I1
I2 = Constants.I2
f1 = Constants.f1
f2 = Constants.f2
g = Constants.g

# //////////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ SMOOTH TRAJECTORY \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# ///////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def smooth_trajectory(theta1_eq1, theta1_eq2, TD, ns, ni):

    """
    Generates a smooth reference trajectory for a double pendulum, transitioning
    between two equilibrium points.

    Args:
        theta1_eq1: Initial equilibrium angle for theta1 (degrees).
        theta1_eq2: Final equilibrium angle for theta1 (degrees).
        TD: Total time horizon (number of time steps).
        ns: Number of states.
        ni: Number of inputs.

    Returns:
        xx_ref: Reference state trajectory. Shape: (ns, TD)
        uu_ref: Reference input trajectory. Shape: (ni, TD)
    """
    # Constant term related to gravity and system parameters
    KKeq = g * (m1 * r1 + m2 * l1)
    theta1_eq1, theta1_eq2 = np.deg2rad(theta1_eq1), np.deg2rad(theta1_eq2)

    xx_ref = np.zeros((ns, TD))                 # Initialize reference state trajectory
    uu_ref = np.zeros((ni, TD))                 # Initialize reference input trajectory

    # Time ranges for applying the control
    init_apply_u = int(TD / 8)
    end_apply_u = int(7 * TD / 8)

    # Initial constant trajectory values
    xx_ref[0, :init_apply_u] = theta1_eq1
    xx_ref[1, :init_apply_u] = -theta1_eq1
    uu_ref[0, :init_apply_u] = KKeq * np.sin(theta1_eq1)

    # Apply S-shaped smooth transition (S-shaped polynomial)
    t_range = np.linspace(0, 1, end_apply_u - init_apply_u)  # Normalized time
    s_poly = 3 * t_range**2 - 2 * t_range**3  # Cubic S-shaped polynomial (smoothstep)

    u_start = KKeq * np.sin(theta1_eq1)
    u_end = KKeq * np.sin(theta1_eq2)

    # Transition for xx_ref
    xx_ref[0, init_apply_u:end_apply_u] = (1 - s_poly) * theta1_eq1 + s_poly * theta1_eq2
    xx_ref[1, init_apply_u:end_apply_u] = (1 - s_poly) * -theta1_eq1 + s_poly * -theta1_eq2
    uu_ref[0, init_apply_u:end_apply_u] = (1 - s_poly) * u_start + s_poly * u_end

    # After end_apply_u, maintain the final value
    xx_ref[0, end_apply_u:] = theta1_eq2
    xx_ref[1, end_apply_u:] = -theta1_eq2
    uu_ref[0, end_apply_u:] = KKeq * np.sin(theta1_eq2)

    return xx_ref, uu_ref

# //////////////////////////////////////////////////////////////////////////////////   In order to switch Armijo Plotting and Descent Plotting
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Newton + Armijo \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   On or Off, go the Constants Module and use the
# ///////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   corresponding flags.

stepsize_0 = Constants.stepsize_0
beta = Constants.beta
cc = Constants.cc
armijo_max_iters = Constants.armijo_max_iters
DO_ARMIJO_PLOT = Constants.DO_ARMIJO_PLOT
DO_DESCENT_PLOT = Constants.DO_DESCENT_PLOT


def newton_method(xx_init, uu_init, xx_ref, uu_ref, ns, ni, TD, max_iters, QQ, RR, QQT, stopping_criteria, open_loop):
    """
    Implements a Newton-based optimization method for trajectory optimization.
    
    Args:
        xx_init (ndarray): Initial state trajectory (ns, TD).
        uu_init (ndarray): Initial input trajectory (ni, TD).
        xx_ref (ndarray): Reference state trajectory (ns, TD).
        uu_ref (ndarray): Reference input trajectory (ni, TD).
        ns (int): Number of state variables.
        ni (int): Number of input variables.
        TD (int): Time horizon.
        max_iters (int): Maximum iterations for Newton's method.
        QQ (ndarray): State cost weight matrix (ns, ns).
        RR (ndarray): Input cost weight matrix (ni, ni).
        QQT (ndarray): Terminal state cost weight matrix (ns, ns).
        stopping_criteria (float): Convergence threshold for the descent direction.
        open_loop (bool): Whether to use open-loop control updates.

    Returns:
        xx_star (ndarray): Optimized state trajectory (ns, TD).
        uu_star (ndarray): Optimized input trajectory (ni, TD).
    """

    QQ = QQ[:, :, None]                                                 # QQ Weight Matrix
    QQ = QQ.repeat(TD, axis = 2)

    RR = RR[:, :, None]                                                 # RR Weight Matrix
    RR = RR.repeat(TD, axis = 2)

    xx = np.zeros((ns, TD, max_iters))                                  # State trajectory
    uu = np.zeros((ni, TD, max_iters))                                  # Input trajectory

    AA = np.zeros((ns, ns, TD))                                         # A matrix of Linearization
    BB = np.zeros((ns, ni, TD))                                         # B matrix of Linearization

    qq = np.zeros((ns, TD))                                             # dJ/dx
    rr = np.zeros((ni, TD))                                             # dJ/du
    lmbd = np.zeros((ns, TD, max_iters))                                # Lambda

    JJ = np.zeros((max_iters))                                          # Cost function
    dJ = np.zeros((ni, TD, max_iters))                                  # Gradient of the cost function
    delta_u = np.zeros((ni, TD, max_iters))                             # Descent direction
    deltua_u_norm = np.zeros((max_iters))                               # Norm of the descent direction

    xx[:, :, 0] = xx_init
    uu[:, :, 0] = uu_init
    x_0 = xx_ref[:, 0]

    for kk in range(max_iters - 1):

        # Compute the dynamics and the Jacobian
        for tt in range(TD - 1):
            AA[:, :, tt], BB[:, :, tt] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1: 3]
            qq[:, tt] = cst.stage_cost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt], QQ[:, :, tt], RR[:, :, tt])[1]
            rr[:, tt] = cst.stage_cost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt], QQ[:, :, tt], RR[:, :, tt])[2]

        qqT = cst.terminal_cost(xx[:, -1, kk], xx_ref[:, -1], QQT)[1]
        KK, _ , _, delta_u[:, :, kk], sigma = affine_lqr(AA, BB, 0, QQ, RR, qq, rr, QQT, qqT, TD)

        lmbd[:, -1, kk] = cst.terminal_cost(xx[:, -1, kk], xx_ref[:, -1], QQT)[1]

        for tt in reversed(range(TD - 1)):
            lmbd[:, tt, kk] = AA[:, :, tt].T @ lmbd[:, tt + 1, kk] + qq[:, tt]
            dJ[:, tt, kk] = BB[:, :, tt].T @ lmbd[:, tt + 1, kk] + rr[:, tt]

        JJ[kk] = 0
        for tt in range(TD-1):
            JJ[kk] += cst.stage_cost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt], QQ[:, :, tt], RR[:, :, tt])[0]
        
        JJ[kk] += cst.terminal_cost(xx[:, -1, kk], xx_ref[:, -1], QQT)[0]


        ### armijo function to calculate stepsize
        stepsize = stepsize_0
        stepsizes = []
        armijo_costs = []
        descent_value = 0

        for tt in range(TD -1):
            descent_value += dJ[:, tt, kk].T @ delta_u[:, tt, kk]

        for ii in range(armijo_max_iters):

            xx_temp = np.zeros((ns, TD))
            xx_temp[:, 0] = xx_ref[:, 0]
            uu_temp = np.zeros((ni, TD))
            JJ_temp = 0
            
            for tt in range(TD - 1):
                if open_loop:
                    uu_temp[:, tt] = uu[:, tt, kk] + stepsize * delta_u[:, tt, kk]
                else:
                    uu_temp[:, tt] = uu[:, tt, kk] + KK[:, :, tt] @ (xx_temp[:, tt] - xx[:, tt, kk]) + stepsize * sigma[:, tt]

                xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]
            
            for tt in range(TD -1):
                JJ_temp += cst.stage_cost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt], QQ[:, :, tt], RR[:, :, tt])[0]
            
            JJ_temp += cst.terminal_cost(xx_temp[:, -1], xx_ref[:, -1], QQT)[0]

            stepsizes.append(stepsize)
            armijo_costs.append(JJ_temp)


            if (JJ_temp > JJ[kk] + cc * stepsize * descent_value):
                stepsize *= beta

            else:
                print(f"--- Armijo Step Size: {stepsize} Found after {ii} iterations in Newton Iteration {kk} ---")
                break

            if ii == armijo_max_iters - 1:
                print("with current configurations , Armijo Rule could not find any stepsize. ||| Change setiings!")
        
        if DO_ARMIJO_PLOT:
            steps = np.linspace(0, 1, int(2e1))
            costs = np.zeros(len(steps))

            for ii , step in enumerate(steps):
                xx_temp = np.zeros((ns, TD))
                uu_temp = np.zeros((ni, TD))
                xx_temp[:, 0] = xx_ref[:, 0]

                for tt in range(TD - 1):
                    if open_loop:
                        uu_temp[:, tt] = uu[:, tt, kk] + step * delta_u[:, tt, kk]
                    else:
                        uu_temp[:, tt] = uu[:, tt, kk] + KK[:, :, tt] @ (xx_temp[:, tt] - xx[:, tt, kk]) + step * sigma[:, tt]

                    xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

                JJ_temp = 0

                for tt in range(TD - 1):
                    JJ_temp += cst.stage_cost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt],QQ[:,:,tt],RR[:,:,tt])[0]

                JJ_temp += cst.terminal_cost(xx_temp[:, -1], xx_ref[:, -1], QQT)[0]

                costs[ii] = JJ_temp

            plt.figure()
            plt.clf()

            plt.plot(steps, costs, 'b', label='Cost Function')
            plt.plot(steps, JJ[kk] + descent_value * steps, 'r', label='Descent Line')
            plt.plot(steps, JJ[kk] + descent_value * steps * cc, 'g--', label='Armijo Line')
            plt.scatter(stepsizes, armijo_costs, marker = '*')
            plt.legend()
            plt.grid()
            plt.show()

        ### Updating the Solution

        # -------------------------------------- 01/29 commented these
        # xx_temp = np.zeros((ns, TD))
        # uu_temp = np.zeros((ni, TD))
        # xx_temp[:, 0] = x_0

        # for tt in range(TD - 1):

        #     # uu_temp[:, tt] = uu[:, tt, kk] + stepsize * delta_u[:, tt, kk]
        #     uu_temp[:, tt] = uu[:, tt, kk] + KK[:, :, tt] @ ( xx[:, tt, kk+1] - xx[:, tt, kk]) + stepsize * sigma[:, tt]              # 1/29 xx[:, tt, kk] - xx[:, tt, kk] to xx[:, tt, kk+1] - xx[:, tt, kk]

        #     xx_temp[:, tt+1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]
        #     deltua_u_norm[kk] += delta_u[:, tt, kk].T @ delta_u[:, tt, kk]

        # uu_temp[:, -1] = uu_temp[:, -2]
        # xx[:, :, kk+1] = xx_temp.copy()
        # uu[:, :, kk+1] = uu_temp.copy()
        # -------------------------------------- 01/29 up to here: next is new
        xx[:, 0, kk + 1] = x_0

        for tt in range(0, TD - 1):
            uu[:, tt, kk+1] = uu[:, tt, kk] + KK[:, :, tt] @ (xx[:, tt, kk+1] - xx[:, tt, kk]) + stepsize * sigma[:, tt]
            xx[:, tt+1, kk+1] = dyn.dynamics(xx[:, tt, kk+1], uu[:, tt, kk+1])[0]

            deltua_u_norm[kk] += delta_u[:, tt, kk].T @ delta_u[:, tt, kk]



        if (deltua_u_norm[kk] < stopping_criteria) and kk > 0:
            max_iters = kk + 1
            print(f"Newton Method Converged after {kk} iterations")
            break

        print(f"=== iteration {kk} Descent: {deltua_u_norm[kk]} Cost function: {JJ[kk]} ===\n")
    
    if DO_DESCENT_PLOT:
        descent_plot(max_iters, deltua_u_norm, JJ)

    xx_star = xx[:, :, max_iters - 1]
    uu_star = uu[:, :, max_iters - 1]
    uu_star[:, -1] = uu_star[:, -2]
    return xx_star, uu_star

# //////////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Quasi  Static \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# ///////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def control_input(theta_1, theta_2, theta_1_dot, theta_2_dot, theta_1_d2dot, theta_2_d2dot):

    """
    Calculates the required control input (torque) for the double pendulum's
    first joint, given the angular positions, velocities, and accelerations
    of both links.

        M11 * theta1_d2dot_t + M12 * theta2_d2dot_t + C1 + F1 + G1 = u

    Args:
        theta_1: Angular position of the first link (radians).
        theta_2: Angular position of the second link (radians).
        theta_1_dot: Angular velocity of the first link (radians/s).
        theta_2_dot: Angular velocity of the second link (radians/s).
        theta_1_d2dot: Desired angular acceleration of the first link (radians/s^2).
        theta_2_d2dot: Desired angular acceleration of the second link (radians/s^2).

    Returns:
        uu: Required torque (control input) for the first joint (N*m).
    """

    # Mass matrix elements
    M11 = I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) + 2 * m2 * l1 * r2 * np.cos(theta_2)
    M12 = I2 + m2 * r2**2 + m2 * l1 * r2 * np.cos(theta_2)


    # Coriolis and centrifugal forces
    C1 = -m2 * l1 * r2 * np.sin(theta_2) * (2 * theta_1_dot * theta_2_dot + theta_2_dot**2)

    # Gravity forces
    G1 = g * (m1 * r1 + m2 * l1) * np.sin(theta_1) + g * m2 * r2 * np.sin(theta_1 + theta_2)

    F1 = f1 * theta_1_dot

    # Compute required torque (u)
    uu = M11 * theta_1_d2dot + M12 * theta_2_d2dot + C1 + G1 + F1
    
    return uu



def quasi_static_trajectory(theta1_eq1_deg, theta1_eq2_deg, tf, dt, ni):

    """
    Generates a quasi-static reference trajectory for a double pendulum, 
    transitioning between two equilibrium points.

    Args:
        theta1_eq1_deg: Initial equilibrium angle for theta1 (degrees).
        theta1_eq2_deg: Final equilibrium angle for theta1 (degrees).
        tf: Total time for the trajectory (seconds).
        dt: Time step (seconds).
        ns: Number of states.
        ni: Number of inputs.

    Returns:
        xx_ref: Reference state trajectory. Shape: (ns, TD)
        uu_ref: Reference input trajectory. Shape: (ni, TD)
    """

    theta1_eq1, theta1_eq2 = np.deg2rad(theta1_eq1_deg), np.deg2rad(theta1_eq2_deg)
    TD = int(tf/dt)
    theta1 = np.full(TD, theta1_eq1)


    transition_start = TD // 4                  # Start transition at 1/4th of total time
    transition_end = 3 * TD // 4                # End transition at 3/4th of total time
    duration = transition_end - transition_start

    if duration > 0:
        t_norm = np.linspace(0, 1, duration)
        transition_curve = theta1_eq1 + (theta1_eq2 - theta1_eq1) * (3 * t_norm**2 - 2 * t_norm**3)
        theta1[transition_start:transition_end] = transition_curve
        theta1[transition_end:] = theta1_eq2
    
    theta2 = np.zeros_like(theta1)
    theta1_dot = np.gradient(theta1, dt)
    theta1_ddot = np.gradient(theta1_dot, dt)
    theta2_dot = np.zeros_like(theta1)
    theta2_ddot = np.zeros_like(theta1)
    uu_ref = np.zeros((ni, TD))
    
    # Compute theta2, theta2_dot and theta2_d2dot
    for tt in range(TD - 1):
        ### From dynamics we have that: M12 * theta1_d2dot + M22 * theta2_d2dot = -(G2 + C2 + F22)
        theta2_ddot[tt] = dyn.theta2_d2dot(theta1[tt], theta2[tt], theta1_dot[tt], theta2_dot[tt], theta1_ddot[tt])
        theta2_dot[tt+1] = theta2_dot[tt] + dt * theta2_ddot[tt]
        theta2[tt+1] = theta2[tt] + dt * theta2_dot[tt]
    
    # Compute control input 
    for tt in range(TD):
        uu_ref[:, tt] = control_input(theta1[tt], theta2[tt], theta1_dot[tt], theta2_dot[tt], 0, 0)
    
    xx_ref = np.vstack((theta1, theta2, theta1_dot, theta2_dot))
    return xx_ref, uu_ref