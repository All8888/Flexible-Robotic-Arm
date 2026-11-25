import numpy as np
# import matplotlib.pyplot as plt
import Constants

################# Import System Constants
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
dt = Constants.dt
ns = Constants.ns





def dynamics(xx, uu):
    """
    Discretizes the dynamics of a double pendulum using Euler integration.

    Args:
        xx: Current state vector [theta1, theta2, theta1_dot, theta2_dot] with shape (4,)
        uu: Control Input (Torque applied to the first joint) which is a Scalar

    Returns:
        xx_next: Next State Vector with shape (4,)
        A: Jacobian matrix of the discrete-time dynamics with shape (4, 4)
        B: Jacobian matrix of the discrete-time dynamics with respect to the input with shape (4, 1)
    """


    ### Extracting State Variables
    theta1, theta2, theta1_dot, theta2_dot = xx

    ### Compute Mass Matrix Elements
    M11 = I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) + 2*m2 * l1 * r2 * np.cos(theta2)
    M12 = I2 + m2 * r2**2 + m2*l1 * r2 * np.cos(theta2)
    M21 = I2 + m2 * r2**2 +m2* l1 * r2 * np.cos(theta2)
    M22 = I2 + m2 * r2**2

    ### Compute Coriolis, Gravity, and Friction terms
    C1 = -m2 * l1 * r2 * theta2_dot * np.sin(theta2)*(theta2_dot+2*theta1_dot)
    C2 = m2 * l1 * r2 * theta1_dot**2 * np.sin(theta2)
    G1 = g * (m1 * r1 + m2 * l1) * np.sin(theta1) + g * m2 * r2 * np.sin(theta1 + theta2)
    G2 = g * m2 * r2 * np.sin(theta1 + theta2)
    F1 = f1 * theta1_dot
    F2 = f2 * theta2_dot

    M = np.array([[M11, M12],
              [M21, M22]])

    # Compute theta_ddot using matrix inversion
    inv_M = np.linalg.inv(M)

    theta_ddot = inv_M @ np.array([uu.squeeze() - C1 - F1 - G1,
                                -C2 - F2 - G2])

    # Update velocities
    theta1_dot_next = theta1_dot + dt * theta_ddot[0]
    theta2_dot_next = theta2_dot + dt * theta_ddot[1]

    # Update positions using new velocities
    theta1_next = theta1 + dt * theta1_dot_next
    theta2_next = theta2 + dt * theta2_dot_next

    # Next state
    xx_next = np.array([theta1_next, theta2_next, theta1_dot_next.squeeze(), theta2_dot_next.squeeze()])

    ##### A matrix (Jacobian of discretized dynamics with respect to states)
    A31 = -dt * (g * (l1 * m2**2 * r2**2 * np.cos(theta1) + 2 * I2 * l1 * m2 * np.cos(theta1) + 2 * I2 * m1 * r1 * np.cos(theta1) - l1 * m2**2 * r2**2 * np.cos(theta1 + 2 * theta2) + 2 * m1 * m2 * r1 * r2**2 * np.cos(theta1))) / \
            (2 * I1 * I2 + l1**2 * m2**2 * r2**2 + 2 * I2 * l1**2 * m2 + 2 * I2 * m1 * r1**2 + 2 * I1 * m2 * r2**2 - l1**2 * m2**2 * r2**2 * np.cos(2 * theta2) + 2 * m1 * m2 * r1**2 * r2**2)

    #------------------------------------------------------------------------------------- Checked
    A32 = dt * (2 * (m2*l1*r2)**2 * (I2 + m2*r2**2) * np.sin(theta2) * np.cos(theta2)) / ((I2 + m2*r2**2) * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) - (I2 + m2*r2**2)) - (m2*l1*r2)**2 * np.cos(theta2)**2)**2 * (m2*l1*r2*theta2_dot*np.sin(theta2)*(-theta2_dot-2*theta1_dot) + f1*theta1_dot + g*(m1*r1 + m2*l1)*np.sin(theta1) + g*m2*r2*np.sin(theta1+theta2) - uu)

    A32 = A32 - dt * (I2 + m2*r2**2) / ((I2 + m2*r2**2) * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) - (I2 + m2*r2**2)) - (m2*l1*r2)**2 * np.cos(theta2)**2) * (m2*l1*r2*theta2_dot*np.cos(theta2)*(-theta2_dot-2*theta1_dot) + g*m2*r2*np.cos(theta1+theta2))

    A32 = A32 - dt * (m2*l1*r2*np.sin(theta2) * ((I2 + m2*r2**2) * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) - (I2 + m2*r2**2)) + (m2*l1*r2)**2 * np.cos(theta2)**2 + 2 * m2*l1*r2 * (I2 + m2*r2**2) * np.cos(theta2))) / ((I2 + m2*r2**2) * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) - (I2 + m2*r2**2)) - (m2*l1*r2)**2 * np.cos(theta2)**2)**2 * (m2*l1*r2*np.sin(theta2)*theta1_dot**2 + f2*theta2_dot + g*m2*r2*np.sin(theta1+theta2))

    A32 = A32 + dt * ((I2 + m2*r2**2) + m2*l1*r2*np.cos(theta2)) / ((I2 + m2*r2**2) * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) - (I2 + m2*r2**2)) - (m2*l1*r2)**2 * np.cos(theta2)**2) * (m2*l1*r2*np.cos(theta2)*theta1_dot**2 + g*m2*r2*np.cos(theta1+theta2)) 
    #------------------------------------------------------------------------------------- Ckeacked
    # A33
    term1 = 1 - dt * (I2 + m2 * r2**2) / ((I2 + m2 * r2**2) * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2)) - (m2 * l1 * r2)**2 * np.cos(theta2)**2) * (-2 * m2 * l1 * r2 * theta2_dot * np.sin(theta2) + f1)

    term2 = term1 + dt * ((I2 + m2 * r2**2) + m2 * l1 * r2 * np.cos(theta2)) / ((I2 + m2 * r2**2) * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2)) - (m2 * l1 * r2)**2 * np.cos(theta2)**2) * (2 * m2 * l1 * r2 * np.sin(theta2) * theta1_dot)

    A33 = 1 - term1 + term2
    #------------------------------------------------------------------------------------- Correct
    A34 = (
    -dt * (I2 + m2 * r2**2)
    / ((I2 + m2 * r2**2) * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2)) - (m2 * l1 * r2)**2 * np.cos(theta2)**2)
    * (
        -2 * (m2 * l1 * r2) * theta2_dot * np.sin(theta2) 
        - 2 * (m2 * l1 * r2) * theta1_dot * np.sin(theta2)
        )
    )
        
    A34 += (
        dt * ((I2 + m2 * r2**2) + (m2 * l1 * r2) * np.cos(theta2))
        / ((I2 + m2 * r2**2) * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2)) - (m2 * l1 * r2)**2 * np.cos(theta2)**2)
        * f2
    )
    #------------------------------------------------------------------------------------- Correct
    A41 = (
    dt
    * (
        ((I2 + m2 * r2**2) + (m2 * l1 * r2) * np.cos(theta2))
        * (g * (m1 * r1 + m2 * l1) * np.cos(theta1) + g * m2 * r2 * np.cos(theta1 + theta2))
    )
    / (
        ((I2 + m2 * r2**2))
        * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
        - (m2 * l1 * r2)**2 * np.cos(theta2)**2
    )
    - dt
    * (
        ((I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2)) + 2 * (m2 * l1 * r2) * np.cos(theta2))
        * (g * m2 * r2 * np.cos(theta1 + theta2))
    )
    / (
        ((I2 + m2 * r2**2))
        * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
        - (m2 * l1 * r2)**2 * np.cos(theta2)**2
        )
    )
    #------------------------------------------------------------------------------------- Correct
    A42 = (
    dt
    * (
        (
            -(m2 * l1 * r2) * np.sin(theta2)
            * ((I2 + m2 * r2**2) * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
               - (m2 * l1 * r2)**2 * np.cos(theta2)**2)
            - 2 * (m2 * l1 * r2)**2 * np.sin(theta2) * np.cos(theta2)
            * ((I2 + m2 * r2**2) + (m2 * l1 * r2) * np.cos(theta2))
        )
        / (
            ((I2 + m2 * r2**2))
            * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
            - (m2 * l1 * r2)**2 * np.cos(theta2)**2
        )**2
        )
        * ((m2 * l1 * r2) * theta2_dot * np.sin(theta2) * (-theta2_dot - 2 * theta1_dot) + f1 * theta1_dot + g * (m1 * r1 + m2 * l1) * np.sin(theta1) + g * m2 * r2 * np.sin(theta1 + theta2) - uu)
    )

    A42 += (
        dt
        * ((I2 + m2 * r2**2) + (m2 * l1 * r2) * np.cos(theta2))
        / (
            ((I2 + m2 * r2**2))
            * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
            - (m2 * l1 * r2)**2 * np.cos(theta2)**2
        )
        * ((m2 * l1 * r2) * theta2_dot * np.cos(theta2) * (-theta2_dot - 2 * theta1_dot) + g * m2 * r2 * np.cos(theta1 + theta2))
    )

    A42 += (
        dt
        * (
            2 * (m2 * l1 * r2) * np.sin(theta2) * ((m2 * l1 * r2) * np.cos(theta2) + (I2 + m2 * r2**2))
            * ((I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2)) + (m2 * l1 * r2) * np.cos(theta2) - (I2 + m2 * r2**2))
        )
        / (
            ((I2 + m2 * r2**2))
            * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
            - (m2 * l1 * r2)**2 * np.cos(theta2)**2
        )**2
        * ((m2 * l1 * r2) * np.sin(theta2) * theta1_dot**2 + f2 * theta2_dot + g * m2 * r2 * np.sin(theta1 + theta2))
    )

    A42 -= (
        dt
        * ((I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2)) + 2 * (m2 * l1 * r2) * np.cos(theta2))
        / (
            ((I2 + m2 * r2**2))
            * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
            - (m2 * l1 * r2)**2 * np.cos(theta2)**2
        )
        * ((m2 * l1 * r2) * np.cos(theta2) * theta1_dot**2 + g * m2 * r2 * np.cos(theta1 + theta2))
    )
    #------------------------------------------------------------------------------------- Checked
    term43 = dt * ((I2 + m2*r2**2) + m2*l1*r2*np.cos(theta2)) / ((I2 + m2*r2**2) * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) - (I2 + m2*r2**2)) - (m2*l1*r2)**2 * np.cos(theta2)**2) * (-2*m2*l1*r2*theta2_dot*np.sin(theta2) + f1)

    term43_2 = dt * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) + 2*m2*l1*r2*np.cos(theta2)) / ((I2 + m2*r2**2) * (I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) - (I2 + m2*r2**2)) - (m2*l1*r2)**2 * np.cos(theta2)**2) * (2*m2*l1*r2*np.sin(theta2)*theta1_dot)

    A43 = term43 - term43_2
    #------------------------------------------------------------------------------------- Correct
    A44 = 1 + (
        dt
        * (
            (I2 + m2 * r2**2 + m2 * l1 * r2 * np.cos(theta2))
            + (m2 * l1 * r2) * np.cos(theta2)
        )
        / (
            (I2 + m2 * r2**2)
            * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
            - (m2 * l1 * r2) ** 2 * np.cos(theta2) ** 2
        )
        * (-2 * (m2 * l1 * r2) * theta2_dot * np.sin(theta2) - 2 * (m2 * l1 * r2) * theta1_dot * np.sin(theta2))
    )

    A44 -= (
        dt
        * (
            I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) + 2 * (m2 * l1 * r2) * np.cos(theta2)
        )
        / (
            (I2 + m2 * r2**2)
            * (I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) - (I2 + m2 * r2**2))
            - (m2 * l1 * r2) ** 2 * np.cos(theta2) ** 2
        )
        * f2
    )
    #-------------------------------------------------------------------------------------
    A = np.zeros((ns, ns))

    # 1/29
    # change A[0,0] = 1 A[1,1] = 0 to 1
    A[0,0] = 1
    A[0,1] = 0
    A[0,2] = dt
    A[0,3] = 0
    A[1,0] = 0
    A[1,1] = 1
    A[1,2] = 0
    A[1,3] = dt
    A[2,0] = A31
    A[2,1] = A32.item()
    A[2,2] = A33
    A[2,3] = A34
    A[3,0] = A41
    A[3,1] = A42.item()
    A[3,2] = A43
    A[3,3] = A44
    ##### B matrix

    # Compute the determinant of M
    det_M = M11 * M22 - M12 * M21

    B31_tau = (1 / det_M) * M22
    B41_tau = (1 / det_M) * -M12

    # Construct the B matrix (Jacobian of discretized dynamics with respect to input)
    B = np.array([
        [0],        # No direct effect on theta1
        [0],        # No direct effect on theta2
        [B31_tau],  # Effect on theta1_dot
        [B41_tau]   # Effect on theta2_dot
    ]) * dt

    return xx_next, A, B



def qs_dynamics(t, state, theta_1_t, theta_1_dot_t, theta_1_d2dot_t):
    """
    Calculates the dynamics of the second link (theta2) of a double pendulum,
    given the motion of the first link (theta1).

    Args:
        t: Current time (scalar).
        state: Current state of the second link [theta2, theta2_dot] (NumPy array, shape (2,)).
        theta_1_t: Angular position of the first link at time t (scalar).
        theta_1_dot_t: Angular velocity of the first link at time t (scalar).
        theta_1_d2dot_t: Angular acceleration of the first link at time t (scalar).

    Returns:
        A 'list' containing:
            - theta_2_dot: Angular velocity of the second link.
            - theta_2_d2dot: Angular acceleration of the second link.
    """
    theta_2, theta_2_dot = state

    numerator = (
            -(I2 + m2 * r2**2 + m2 * l1 * r2 * np.cos(theta_2)) * theta_1_d2dot_t
            - m2 * l1 * r2 * np.sin(theta_2) * theta_1_dot_t**2
            - f2 * theta_2_dot
            - g * m2 * r2 * np.sin(theta_1_t + theta_2)
        )
    denominator = I2 + m2 * r2**2
    theta_2_d2dot = numerator / denominator
    return [theta_2_dot, theta_2_d2dot]


def theta2_d2dot(theta1, theta2, theta1_dot, theta2_dot,theta1_d2dot):

    """
    Calculates the angular acceleration of the second link (theta2_d2dot) 
    of a double pendulum, given the angles, angular velocities, and the 
    angular acceleration of the first link.

    Args:
        theta1: Angle of the first link (radians).
        theta2: Angle of the second link (radians).
        theta1_dot: Angular velocity of the first link (rad/s).
        theta2_dot: Angular velocity of the second link (rad/s).
        theta1_d2dot: Angular acceleration of the first link (rad/s^2).

    Returns:
        t2d2: Angular acceleration of the second link (rad/s^2).
    """

    # Define required Matrices of Dynamics
    M12 = I2 + m2 * r2**2 + m2*l1 * r2 * np.cos(theta2)
    M22 = I2 + m2 * r2**2
    C2 = m2 * l1 * r2 * theta1_dot**2 * np.sin(theta2)
    G2 = g * m2 * r2 * np.sin(theta1 + theta2)
    F2 = f2 * theta2_dot

    # Theta2_d2dot_t
    t2d2 = - (M12 * theta1_d2dot + C2 + F2 + G2 ) / M22

    return t2d2


### Test of dynamics
# x_0 = np.array([0,0,0,0]).reshape(4,1)
# u_i = np.array([31])

# num_samples = 100
# xx_temp = np.zeros((4, num_samples))
# uu_temp = np.zeros((1, num_samples))

# xx = x_0.copy()

# for i in range(num_samples):
#     xx_temp[:, i] = xx.flatten()
#     uu_temp[:, i] = u_i.flatten()

#     xx_next = dynamics(xx.flatten(), u_i)[0]
#     xx = xx_next.reshape(4,1)

# plt.plot(np.arange(num_samples), xx_temp[0])
# plt.plot(np.arange(num_samples), xx_temp[1], 'r')
# plt.plot(np.arange(num_samples), xx_temp[3], 'g--')
# plt.plot(np.arange(num_samples), xx_temp[3], 'k--')
# plt.show()
