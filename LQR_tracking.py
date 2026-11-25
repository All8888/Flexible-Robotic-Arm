import numpy as np

import Dynamics as dyn
from LQR import lqr
import Constants

ns = Constants.ns
ni = Constants.ni

def lqr_tracking(xx_star, uu_star, x_0, TD, QQ, RR, QQT):
    """
    This function calculates the LQR control inputs to track a given reference trajectory
    for a discrete-time dynamical system. It uses the LQR method to determine the optimal
    feedback gains and then simulates the system's response to these gains, starting from
    the provided initial state.

    Args:
        xx_star (numpy.ndarray): Reference state trajectory. Shape (ns, TD).
        uu_star (numpy.ndarray): Reference input trajectory. Shape (ni, TD).
        x_0 (numpy.ndarray): Initial state of the system. Shape (ns, 1) or (ns,).
        TD (int): Number of Time Discretization
        QQ (numpy.ndarray): State cost matrix. Shape (ns, ns).
        RR (numpy.ndarray): Input cost matrix. Shape (ni, ni).
        QQT (numpy.ndarray): Terminal state cost matrix. Shape (ns, ns).

    Returns:
        tuple: A tuple containing the state trajectory (xx) and input trajectory (uu)
               achieved by the LQR controller.
               xx: LQR-controlled state trajectory. Shape (ns, TD).
               uu: LQR-controlled input trajectory. Shape (ni, TD).
    """

    xx = np.zeros((ns, TD))
    uu = np.zeros((ni, TD))

    AA = np.zeros((ns, ns, TD))         # A matrix at each time step
    BB = np.zeros((ns, ni, TD))         # A matrix at each time step

    # Compute Jacobians along the reference trajectory
    for tt in range(TD - 1):
        AA[:, :, tt], BB[:, :, tt] = dyn.dynamics(xx_star[:, tt], uu_star[:, tt])[1:3]
    
    # Compute LQR gains
    KKlqr = lqr(AA, BB, TD, QQ, RR, QQT)[1]

    xx[:, 0] = x_0

    # Simulate the system using the LQR controller
    for tt in range(TD - 1):
        # Calculate control input
        uu[:, tt] = uu_star[:, tt] + KKlqr[:, :, tt] @ ( xx[:, tt] - xx_star[:, tt])
        # Update state using dynamics
        xx[:, tt+1] = dyn.dynamics(xx[:, tt], uu[:, tt])[0]

    return xx, uu