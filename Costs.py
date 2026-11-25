import numpy as np
import Constants

ns = Constants.ns
ni = Constants.ni

def stage_cost(xx, uu, xx_ref, uu_ref, Q, R):
    """
    Calculates the stage cost and its derivatives for a given state and input.

    Args:
        xx: Current state vector. Shape: (ns,)
        uu: Current input vector. Shape: (ni,)
        xx_ref: Reference state vector. Shape: (ns,)
        uu_ref: Reference input vector. Shape: (ni,)
        Q: State weighting matrix. Shape: (ns, ns)
        R: Input weighting matrix. Shape: (ni, ni)

    Returns:
        ll: Stage cost (scalar).
        llx: Gradient of the stage cost with respect to the state. Shape: (ns,)
        llu: Gradient of the stage cost with respect to the input. Shape: (ni,)
        llxx: Hessian of the stage cost with respect to the state. Shape: (ns, ns)
        lluu: Hessian of the stage cost with respect to the input. Shape: (ni, ni)
        llux: Mixed Hessian of the stage cost (input wrt state). Shape: (ni, ns)
    """
    # Quadratic Cost Function
    ll = 0.5 * (xx - xx_ref).T @ Q @ (xx - xx_ref) + 0.5 * (uu - uu_ref).T @ R @ (uu - uu_ref)

    # gradient of the stage cost with respect to the state
    llx = Q @ (xx - xx_ref)

    # gradient of the stage cost with respect to the input
    llu = R @ (uu - uu_ref)

    # Hessian of the stage cost with respect to the state
    llxx = Q

    # Hessian of the stage cost with respect to the input
    lluu = R

    # mixed Hessian
    llux = np.zeros((ni, ns))

    return ll, llx, llu, llxx, lluu, llux



def terminal_cost(xx, xx_ref, QT):
    """
    Calculates the terminal cost and its derivatives for a given state.

    Args:
        xx: Current state vector at the terminal time. Shape: (ns,)
        xx_ref: Reference state vector at the terminal time. Shape: (ns,)
        QT: Terminal state weighting matrix. Shape: (ns, ns)

    Returns:
        llt: Terminal cost (scalar).
        lltx: Gradient of the terminal cost with respect to the state. Shape: (ns,)
        lltxx: Hessian of the terminal cost with respect to the state. Shape: (ns, ns)
    """

    # Quadratic cost function (terminal cost)
    llt = 0.5 * (xx - xx_ref).T @ QT @ (xx - xx_ref)

    # gradient of the terminal cost with respect to the state
    lltx = QT @ (xx - xx_ref)

    # Hessian of the terminal cost with respect to the state
    llxx = QT

    return llt, lltx, llxx