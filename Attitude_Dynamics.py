import numpy as np


# Skew-symmetric matrix of a vector
def skew(v):
    return np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

# Convert MRP to DCM
def mrp_to_dcm(s):
    """DCM C_B/H (H-frame components -> B-frame components) from MRP sigma."""
    s2 = s @ s
    ss = skew(s)
    C = np.eye(3) + (8*ss@ss - 4*(1-s2)*ss) / (1+s2)**2
    return C

# Kinematic differential equation for MRP
def mrp_kde(s, w):
    """sigma_dot = 1/4 [ (1-s^2) I + 2[s x] + 2 s s^T ] w """
    s2 = s @ s
    B = (1-s2)*np.eye(3) + 2*skew(s) + 2*np.outer(s, s)
    return 0.25 * B @ w

# Dynamics of a rigid body in torque-free motion
def euler_rot_eq(w, I, I_inv):
    """Torque-free Euler's equation: I wdot = -[w x] I w"""
    return I_inv @ (-skew(w) @ (I @ w))

# Chief's attitude dynamics
def chief_attitude_eom(t,state, I, w_HN_H):
    """Chief's attitude dynamics (MRP + body rates)."""
    s = state[:3]  # MRP
    w = state[3:]  # Body rates (rad/s)
    I_inv = np.linalg.inv(I)

    w_dot = euler_rot_eq(w, I, I_inv)

    C_BH = mrp_to_dcm(s)
    w_BH = w - C_BH @ w_HN_H   # relative angular velocity, expressed in B frame

    sdot = mrp_kde(s, w_BH)
    return np.concatenate((sdot, w_dot))
