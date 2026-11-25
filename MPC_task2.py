import numpy as np
import Constants
import Dynamics as dyn
from LQR import mpc_lqr


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

def mpc_task2(xx, uu, T_pred, perturbation, QQ, RR, QQT):


    # QQ = np.diag([100, 100, 0.01, 0.01])
    # RR = 0.1 * np.eye(ni)
    # QQ = np.diag([1000, 1000, 0.01, 0.01])
    # RR = 0.01 * np.eye(ni)
    QQ = np.diag([100000, 100000, 0.001, 0.001])
    RR = 0.01 * np.eye(ni)
    QQT = QQ

    # Linearize the dynamics along the desired trajectory to get time-varying A and B matrices
    AA, BB = lin_dynamics(xx, uu)

    xx_input = xx[:,0] + perturbation

    TT = xx.shape[1]

    # Arrays to store the MPC-controlled trajectory.
    xx_mpc = np.zeros((ns, int(TT - T_pred)))
    xx_mpc[:, 0] = xx_input

    uu_mpc = np.zeros((ni, int(TT - T_pred)))

    # Main MPC loop
    for tt in range(TT - T_pred -1):
        # Calculate the LQR gain matrix for the prediction horizon
        KK = mpc_lqr(AA, BB, QQ, RR, QQT, tt, T_pred)

        # Calculate the control input for the current time step
        uu_temp = uu[:, tt] + KK[0] @ (xx_input - xx[:, tt])

        # Update the state of the system based on the calculated control input
        xx_input = dyn.dynamics(xx_input, uu_temp)[0]

        uu_mpc[:, tt] = uu_temp

        if tt == TT:
            break
        else:
            xx_mpc[:, tt+1] = xx_input

    return xx_mpc, uu_mpc