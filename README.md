# Flexible-Robotic-Arm
# Double Pendulum Optimal Control & Trajectory Tracking

A complete Python framework for **trajectory optimization** and **closed-loop tracking** of a planar double pendulum (underactuated, single torque input on the first joint).

The project implements:

- Smooth reference trajectory generation between two equilibria
- Quasi-static reference trajectory generation
- Newton method + Armijo line search for open-loop optimal control
- Finite-horizon LQR tracking
- Model Predictive Control (MPC) tracking (with and without state/input constraints)

**Author:** Alireza Mansouri

---

## System Description

The system is a classic underactuated double pendulum with the following parameters (see `Constants.py`):

| Parameter | Value | Description              |
|-----------|-------|--------------------------|
| \( m_1, m_2 \) | 1.5 kg | Link masses             |
| \( l_1, l_2 \) | 2.0 m  | Link lengths            |
| \( r_1, r_2 \) | 1.0 m  | Center of mass distances|
| \( I_1, I_2 \) | 2.0    | Moments of inertia      |
| \( f_1, f_2 \) | 0.1    | Viscous friction        |
| \( g \)       | 9.81   | Gravity                 |

**State vector**  
\[
\mathbf{x} = [\theta_1,\ \theta_2,\ \dot{\theta}_1,\ \dot{\theta}_2]^\top
\]

**Input**  
\[
u = \tau_1 \quad \text{(torque applied only at the first joint)}
\]

Dynamics are discretized with a fixed step \( dt = 0.01\,\text{s} \) (Euler integration).

---

## Project Structure
├── Main.py                 # Entry point – run everything from here
├── Constants.py            # All parameters, cost weights and flags
├── Dynamics.py             # Continuous → discrete dynamics + Jacobians
├── Costs.py                # Stage & terminal quadratic costs
├── Functions.py            # Smooth / quasi-static trajectories + Newton method
├── LQR.py                  # Finite-horizon LQR & affine LQR
├── LQR_tracking.py         # LQR trajectory tracking
├── MPC_task1.py            # MPC for smooth trajectory (with constraints)
├── MPC_task2.py            # MPC for quasi-static trajectory
├── Plots.py                # All plotting functions
└── Animation.py            # GIF animations of MPC & LQR trackingText
---

## Features

### 1. Reference Trajectory Generation
- **Smooth trajectory** – cubic smoothstep transition between two equilibria
- **Quasi-static trajectory** – very slow transition that respects the dynamics of the passive joint

### 2. Open-loop Optimal Control
- Newton method with exact Hessian information
- Armijo backtracking line search
- Open-loop or closed-loop (feedback) update modes

### 3. Closed-loop Tracking
- Finite-horizon LQR
- Model Predictive Control (receding-horizon LQR)
  - Task 1 version includes hard state & input constraints
  - Task 2 version is unconstrained

### 4. Visualization
- Static plots of all trajectories and tracking errors
- Animated GIFs of state tracking and of the swinging pendulum

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/double-pendulum-optimal-control.git
cd double-pendulum-optimal-control

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install numpy matplotlib scipy pillow psutilHow to RunEverything is controlled by flags in Constants.py.Recommended workflow
Open Constants.py
Enable only one task at a time (the code prints a warning if both are active)
Task 1 – Smooth trajectoryPythonSOLVE_SMOOTH          = True
DO_SMOOTH_PLOT        = True
DO_TRACK_TRAJECTORY   = True
SOLVE_MPC_T1          = True
DO_MPC_TASK1_PLOT     = TrueTask 2 – Quasi-static trajectoryPythonSOLVE_QUASI_STATIC    = True
DO_QUASI_STATIC_PLOT  = True
DO_QS_TRACK_PLOT      = True
SOLVE_QS_LQR          = True
DO_QS_LQR_PLOT        = True
SOLVE_MPC_TASK2       = True
DO_MPC_TASK2_PLOT     = True
Run the main script:
Bashpython Main.pyOptional animationsPythonMPC_ANIMATION = True   # creates four GIFs of state tracking
LQR_ANIMATION = True   # creates a GIF of the swinging double pendulumKey AlgorithmsNewton Method (Trajectory Optimization)Solves the discrete-time optimal control problem
$$\min_{\mathbf{u}} \quad J = \sum_{t=0}^{T-1} \ell(\mathbf{x}_t,\mathbf{u}_t) + \ell_T(\mathbf{x}_T)$$
subject to the nonlinear dynamics. The search direction is obtained from an affine LQR problem (Riccati recursion) and the step size is selected by the Armijo rule.LQR TrackingLinearizes the dynamics along the optimal trajectory and designs a time-varying feedback
$$\mathbf{u}_t = \mathbf{u}_t^\star + K_t (\mathbf{x}_t - \mathbf{x}_t^\star)$$MPCAt every time step a short-horizon LQR problem is solved on the linearized dynamics and only the first control is applied (receding horizon). Task 1 version also enforces box constraints on states and inputs via clipping.ResultsTypical outputs include:
Optimized state and input trajectories
Tracking error plots (LQR vs MPC)
Animated GIFs of the pendulum and of the state evolution
Memory / CPU usage is automatically reported at the end of each run.
