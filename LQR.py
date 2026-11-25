import numpy as np
import Constants

ns = Constants.ns
ni = Constants.ni

def affine_lqr(AA, BB, x_0, QQ, RR, qq, rr, QQT, qqT, TD, SS = None):

    """
    Calculates the LQR gain matrix (K), Riccati matrix (P), state trajectory (xx),
    input trajectory (uu), and feedforward term (sigma) for a finite-horizon 
    affine LQR problem.

    Args:
        AA: NumPy array of state matrices (A) along the time horizon. 
           Shape: (ns, ns, TD)
        BB: NumPy array of input matrices (B) along the time horizon. 
           Shape: (ns, ni, TD)
        x_0: Initial state vector. Shape: (ns,)
        QQ: State weighting matrix (Q). Shape: (ns, ns, TD)
        RR: Input weighting matrix (R). Shape: (ni, ni, TD)
        qq: Linear term in the state cost. Shape: (ns, TD)
        rr: Linear term in the input cost. Shape: (ni, TD)
        QQT: Terminal state weighting matrix (Q_T). Shape: (ns, ns)
        qqT: Linear term in the terminal state cost. Shape: (ns,)
        TD: Total time horizon (integer).
        SS: Cross-term weighting matrix (S, optional). Shape: (ns, ni, TD)

    Returns:
        KK: NumPy array of LQR gain matrices. Shape: (ni, ns, TD - 1)
        PP: NumPy array of Riccati matrices. Shape: (ns, ns, TD)
        xx: NumPy array of state trajectories. Shape: (ns, TD)
        uu: NumPy array of input trajectories. Shape: (ni, TD)
        sigma: NumPy array of feedforward terms. Shape: (ni, TD-1)
    """

    xx = np.zeros((ns, TD))                 # Initialize state trajectory
    uu = np.zeros((ni, TD))                 # Initialize input trajectory
    xx[:,0] = x_0                           # Set initial state

    PP = np.zeros((ns, ns, TD))             # Initialize Riccati matrices
    pp = np.zeros((ns, TD))                 # Initialize affine term for P
    KK = np.zeros((ni, ns, TD))             # Initialize gain matrices
    sigma = np.zeros((ni, TD))              # Initialize feedforward terms

    if AA.ndim < 3:
        AA = AA[:, :, None]
        AA = AA.repeat(TD, axis = 2)

    if BB.ndim < 3:
        BB = BB[:, :, None]
        BB = BB.repeat(TD, axis = 2)

    if QQ.ndim < 3:
        QQ = QQ[:, :, None]
        QQ = QQ.repeat(TD, axis = 2)

    if RR.ndim < 3:
        RR = RR[:, :, None]
        RR = RR.repeat(TD, axis = 2)

    if qq.ndim < 2:
        qq = qq[:, None]
        qq = qq.repeat(TD, axis = 1)
    
    if rr.ndim < 2:
        rr = rr[:, None]
        rr = rr.repeat(TD, axis = 1)

    if SS is not None:
        if SS.ndim < 3:
            SS = SS[:, :, None]
            SS = SS.repeat(TD, axis = 2)
    
    PP[:, :, -1] = QQT
    pp[:, -1] = qqT

    # Backward Riccati equation
    for tt in reversed(range(TD - 1)):
        AAt = AA[:, :, tt]
        BBt = BB[:, :, tt]
        PPtp = PP[:, :, tt + 1]
        pptp = pp[:, tt + 1]
        QQt = QQ[:, :, tt]
        RRt = RR[:, :, tt]
        qqt = qq[:, tt]
        rrt = rr[:, tt]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp

        if SS is None:
            PP[:, :, tt] = QQt + AAt.T @ PPtp @ AAt - AAt.T @ PPtp @ BBt @ MMt_inv @ BBt.T @ PPtp @ AAt
            pp[:, tt] = qqt + AAt.T @ pptp - (BBt.T @ PPtp @ AAt).T @ MMt_inv @ mmt
        else:
            SSt = SS[:, :, tt]
            PP[:, :, tt] = QQt + AAt.T @ PPtp @ AAt - (BBt.T @ PPtp @ AAt + SSt).T @ MMt_inv @ (BBt.T @ PPtp @ AAt + SSt)
            pp[:, tt] = qqt + AAt.T @ pptp - (BBt.T @ PPtp @ AAt).T @ MMt_inv @ mmt
        
    # Gain and feed forward calculation
    for tt in range(TD - 1):
        AAt = AA[:, :, tt]
        BBt = BB[:, :, tt]
        RRt = RR[:, :, tt]
        PPtpt = PP[:, :, tt + 1]
        ppt = pp[:, tt+1]
        qqt = qq[:, tt]
        rrt = rr[:, tt]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtpt @ BBt)
        mmt = rrt + BBt.T @ ppt

        if SS is None:
            KK[:, :, tt] = -MMt_inv @ BBt.T @ PPtpt @ AAt
        else:
            SSt = SS[:, :, tt]
            KK[:, :, tt] = -MMt_inv @ (BBt.T @ PPtpt @ AAt + SSt)
        
        sigma[:, tt] = -MMt_inv @ mmt
    
    # Simulate the system using calculated gains and feedforward
    for tt in range(TD-1):
        uu[:, tt] = KK[:, :, tt] @ xx[:, tt] + sigma[:, tt]
        xx[:, tt+1] = AA[:, :, tt] @ xx[:, tt] + BB[:, :, tt] @ uu[:, tt]
    
    return KK, PP, xx, uu, sigma

        

def lqr(AA, BB, TD, QQ, RR, QQT, SS = None):

    """
    Calculates the LQR gain matrix (K) and Riccati matrix (P) for a finite horizon 
    optimal control problem.

    Args:
        AA: NumPy array of state matrices (A) along the time horizon. 
           Shape: (ns, ns, TD)
        BB: NumPy array of input matrices (B) along the time horizon. 
           Shape: (ns, ni, TD)
        TD: Total time horizon (integer).
        QQ: State weighting matrix (Q). Shape: (ns, ns, TD)
        RR: Input weighting matrix (R). Shape: (ni, ni, TD)
        QQT: Terminal state weighting matrix (Q_T). Shape: (ns, ns)
        SS: Cross-term weighting matrix (S, optional). Shape: (ns, ni, TD)

    Returns:
        PP: NumPy array of Riccati matrices. Shape: (ns, ns, TD)
        KK: NumPy array of LQR gain matrices. Shape: (ni, ns, TD - 1)
    """

    if AA.ndim < 3:
        AA = AA[:, :, None]
        AA = AA.repeat(TD, axis = 2)

    if BB.ndim < 3:
        BB = BB[:, :, None]
        BB = BB.repeat(TD, axis = 2)

    if QQ.ndim < 3:
        QQ = QQ[:, :, None]
        QQ = QQ.repeat(TD, axis = 2)

    if RR.ndim < 3:
        RR = RR[:, :, None]
        RR = RR.repeat(TD, axis = 2)


    if SS is not None:
        if SS.ndim < 3:
            SS = SS[:, :, None]
            SS = SS.repeat(TD, axis = 2)
    

    KK = np.zeros((ni, ns, TD))                 # Initialize gain matrices
    PP = np.zeros((ns, ns, TD))                 # Initialize Riccati matrices

    PP[:, :, -1] = QQT

    # Backward Riccati equation
    for tt in reversed(range(TD - 1)):
        AAt  = AA[:, :, tt]
        BBt = BB[:, :, tt]
        PPtp = PP[:, :, tt + 1]
        QQt = QQ[:, :, tt]
        RRt = RR[:, :, tt]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)

        if SS is not None:
            SSt = SS[:, :, tt]
            PP[:, :, tt] = QQt + AAt.T @ PPtp @ AAt - (BBt.T @ PPtp @ AAt + SSt).T  @ MMt_inv @ (BBt.T @ PPtp @ AAt + SSt )
        else:
            PP[:, :, tt] = QQt + AAt.T @ PPtp @ AAt - AAt.T @ PPtp @ BBt @ MMt_inv @ BBt.T @ PPtp @ AAt
        
    # Gain calculation
    for tt in range(TD - 1):
        PPtp = PP[:, :, tt+1]
        AAt = AA[:, :, tt]
        BBt = BB[:, :, tt]
        QQt = QQ[:, :, tt]
        RRt = RR[:, :, tt]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)

        if SS is not None:
            SSt = SS[:, :, tt]
            KK[:, :, tt] = -MMt_inv @ (BBt.T @ PPtp @ AAt + SSt)
        else:
            KK[:, :, tt] = -MMt_inv @ BBt.T @ PPtp @ AAt
    
    return PP, KK



def mpc_lqr(AA, BB, QQ, RR, QQT, tt, TD):

    """
    Computes the LQR gain matrix (K) for Model Predictive Control (MPC).

    Args:
        AA: NumPy array of linearized state matrices (A) along the prediction 
           horizon.  Shape: (T_hor, ns, ns).
        BB: NumPy array of linearized input matrices (B) along the prediction 
           horizon. Shape: (T_hor, ns, ni)
        QQ: State weighting matrix in the LQR cost function. Shape: (ns, ns)
        RR: Input weighting matrix in the LQR cost function. Shape: (ni, ni)
        QQT: Terminal state weighting matrix in the LQR cost function. Shape: (ns, ns)
        tt: Current time step (scalar). Used to index into the AA and BB arrays.
        TD: Total prediction horizon (integer).

    Returns:
        KK: NumPy array of LQR gain matrices. Shape: (TD - 1, ni, ns) or (TD - 1, ns) if ni=1.
    """
    PP = np.zeros((TD, ns, ns))
    KK = np.zeros((TD, ns))

    T_hor = min(TD, len(AA) - tt)
    AA_m = AA[tt: tt + T_hor]
    BB_m = BB[tt: tt + T_hor]

    PP[-1] = QQT

    for tt in reversed(range(TD - 1)):
        AAt = AA_m[tt]
        BBt = BB_m[tt]
        PPtp = PP[tt + 1]
        QQt = QQ
        RRt = RR

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)

        PP[tt] = QQt + AAt.T @ PPtp @ AAt - AAt.T @ PPtp @ BBt @ MMt_inv @ BBt.T @ PPtp @ AAt
    
    for tt in range(TD - 1):
        AAt = AA_m[tt]
        BBt = BB_m[tt]
        PPtp = PP[tt + 1]
        RRt = RR

        MMt_inv = np.linalg.inv(RRt + RRt + BBt.T @ PPtp @ BBt)

        KK[tt] = -MMt_inv @ BBt.T @ PPtp @ AAt

    return KK