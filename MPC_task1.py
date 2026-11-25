import numpy as np
import Dynamics as dyn
import LQR as lqr
import Constants

ns = Constants.ns
ni = Constants.ni

def lin_dynamics(xx, uu):
    """
    Linearizes the double pendulum dynamics along a given trajectory.

    Args:
        xx: A NumPy array representing the state trajectory. 
           Shape: (ns, TD) where ns is the number of states and TD is 
           the time dimension.
        uu: A NumPy array representing the input trajectory.
           Shape: (TD, ni) where ni is the number of inputs.

    Returns:
        AA_m: A NumPy array containing the linearized state matrices (A) 
             at each time step. Shape: (TD, ns, ns)
        BB_m: A NumPy array containing the linearized input matrices (B) 
             at each time step. Shape: (TD, ns, ni)
    """

    AA_m = np.zeros((xx.shape[1], ns, ns))
    BB_m = np.zeros((uu.shape[1], ns, ni))

    for tt in range(xx.shape[1]):
        # Compute the Jacobians (A and B) at the current time step using the
        AA_m[tt], BB_m[tt] = dyn.dynamics(xx[:,tt], uu[:,tt])[1:3]
    
    return AA_m, BB_m


def mpc_taks1(xx_star, uu_star, T_pred, perturbation, QQ, RR, QQT, xx_lb = None, xx_ub = None, uu_lb = None, uu_ub = None):
    """
    Performs Model Predictive Control (MPC) tracking for a double robotic arm.

    Args:
        xx_star: NumPy array representing the desired state trajectory.
                 Shape: (ns, TT) where ns is the number of states and TT is 
                 the total time horizon.
        uu_star: NumPy array representing the desired input trajectory.
                 Shape: (ni, TT) where ni is the number of inputs.
        T_pred:  Prediction horizon for the MPC controller (integer).
        perturbation: NumPy array representing the initial state perturbation.
                     Shape: (ns,)
        QQ:      Weight matrix for the state error in the LQR cost function.
                 Shape: (ns, ns)
        RR:      Weight matrix for the input effort in the LQR cost function.
                 Shape: (ni, ni)
        QQT:     Weight matrix for the terminal state error in the LQR cost function.
                 Shape: (ns, ns)

    Returns:
        xx_mpc: NumPy array representing the MPC controlled state trajectory.
                Shape: (ns, TT - T_pred)
        uu_mpc: NumPy array representing the MPC controlled input trajectory.
                Shape: (ni, TT - T_pred)
    """
    # QQ = np.diag([1000,1000,0.01,0.01])
    # RR = 0.01 * np.eye(ni)

    # QQ = np.diag([100,100,1,1])
    # RR = 0.1 * np.eye(ni)

    QQ = np.diag([100000,100000,0.1,0.1])
    RR = 0.001 * np.eye(ni)
    QQT = 10 * QQ

    # Linearize the dynamics along the desired trajectory to get time-varying A and B matrices
    AA, BB = lin_dynamics(xx_star, uu_star)

    xx_input = xx_star[:, 0] + perturbation

    TT = xx_star.shape[1]

    # Arrays to store the MPC-controlled trajectory.
    xx_mpc = np.zeros((ns, int(TT - T_pred)))
    xx_mpc[:, 0] = xx_input

    uu_mpc = np.zeros((ni, int(TT - T_pred)))

    # Main MPC loop
    for tt in range(TT - T_pred -1):
        # Calculate the LQR gain matrix for the prediction horizon
        KK = lqr.mpc_lqr(AA, BB, QQ, RR, QQT, tt, T_pred)

        # Calculate the control input for the current time step
        uu = uu_star[:, tt] + KK[0] @ (xx_input - xx_star[:, tt])
        # Apply Constraints
        uu = np.clip(uu, uu_lb, uu_ub)

        # Update the state of the system based on the calculated control input
        xx_input = dyn.dynamics(xx_input, uu)[0]
        # Apply Constraints
        xx_input = np.clip(xx_input, xx_lb, xx_ub)

        uu_mpc[:,tt] = uu
        # xx_mpc[:, tt+1] = xx_input

        if tt == TT:
            break
        else:
            xx_mpc[:, tt+1] = xx_input
    
    return xx_mpc, uu_mpc
    



