import numpy as np

# Orbit Elements to Cartesian Coordinates
def oe2cc(oe, mu):
    a, e, i, raan, argp, ta = oe
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(ta))
    h = np.sqrt(mu * p)
    theta = argp + ta

    RP_ON = np.array([[np.cos(raan)*np.cos(theta) - np.sin(raan)*np.sin(theta)*np.cos(i)],
                      [np.sin(raan)*np.cos(theta) + np.cos(raan)*np.sin(theta)*np.cos(i)],
                      [np.sin(theta)*np.sin(i)]])

    RV_ON = np.array([[np.cos(raan)*(np.sin(theta) + e*np.sin(argp)) + np.sin(raan)*(np.cos(theta) + e*np.cos(argp))*np.cos(i)],
                      [np.sin(raan)*(np.sin(theta) + e*np.sin(argp)) - np.cos(raan)*(np.cos(theta) + e*np.cos(argp))*np.cos(i)],
                      [-(np.cos(theta) + e*np.cos(argp))*np.sin(i)]])

    # Position and velocity in inertial frame
    r_eci = r * RP_ON
    v_eci = RV_ON * -mu/h

    return r_eci, v_eci


# Equations of motion (two-body, point-mass gravity)
def inertial_eom(t, state, mu):
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)

    # Gravitational acceleration
    a_gravity = -mu * r / r_norm**3

    return np.concatenate((v, a_gravity))


# Clohessy-Wiltshire-Hill (CWH) Equations of motion Analytical solution
def cwh_analytical(t, cwh_params, n):
    """Analytical solution to the CWH equations."""
    A0, B0, alpha, beta, xoff, yoff = cwh_params

    x = A0 * np.cos(n*t + alpha) + xoff
    y = -2 * A0 * np.sin(n*t + alpha) -3/2 * n * t * xoff + yoff
    z = B0 * np.cos(n*t + beta)

    xdot = -A0 * n * np.sin(n*t + alpha)
    ydot = -2 * A0 * n * np.cos(n*t + alpha) - 1.5 * n * xoff
    zdot = -B0 * n * np.sin(n*t + beta)

    xddot = -A0 * n**2 * np.cos(n*t + alpha)
    yddot =  2 * A0 * n**2 * np.sin(n*t + alpha)
    zddot = -B0 * n**2 * np.cos(n*t + beta)

    return np.array([x, y, z, xdot, ydot, zdot, xddot, yddot, zddot])


# CONTROL

def get_A1(n):
    return np.array([[3*n**2, 0, 0],
                      [0,      0, 0],
                      [0,      0, -n**2]])

def get_A2(n):
    return np.array([[0,   2*n, 0],
                      [-2*n, 0,  0],
                      [0,    0,  0]])

def cwh_control(state, n, reference, K, P, stage):
    """Returns u, dx_err, dv_err for a given state/reference."""
    pos = state[0:3]
    vel = state[3:6]

    A1 = get_A1(n)
    A2 = get_A2(n)

    dx_err = pos - reference[0:3]
    dv_err = vel - reference[3:6]

    if stage == 0:
        acc_ref = reference[6:9]
        u = acc_ref - A1 @ pos - A2 @ vel - K*dx_err - P*dv_err
    else:
        u = - A1 @ pos - A2 @ vel - K*dx_err - P*dv_err

    return u, dx_err, dv_err

def cwh_equations_controlled(t, state, n, reference, K, P, stage):
    pos = state[0:3]
    vel = state[3:6]

    u, _, _ = cwh_control(state, n, reference, K, P, stage)

    accel = get_A1(n) @ pos + get_A2(n) @ vel + u
    return np.concatenate([vel, accel])



# FRAME TRANSFORMATIONS

def lvlh_dcm(rc, vc):
    """Inertial -> LVLH rotation matrix and the chief's angular rate (rad/s)."""
    o_r = rc / np.linalg.norm(rc)
    h_vec = np.cross(rc, vc)
    o_h = h_vec / np.linalg.norm(h_vec)
    o_t = np.cross(o_h, o_r)
    fdot = np.linalg.norm(h_vec) / np.linalg.norm(rc)**2
    return np.array([o_r, o_t, o_h]), fdot

def inertial_to_lvlh(xc, xd):
    rc, vc = xc[:3], xc[3:]
    NO, fdot = lvlh_dcm(rc, vc)
    omega = np.array([0.0, 0.0, fdot])           # in LVLH components
    rho = NO @ (xd[:3] - rc)
    rho_dot = NO @ (xd[3:] - vc) - np.cross(omega, rho)
    return np.concatenate((rho, rho_dot))

def lvlh_to_inertial(xc, xrho):
    rc, vc = xc[:3], xc[3:]
    NO, fdot = lvlh_dcm(rc, vc)
    omega = np.array([0.0, 0.0, fdot])
    rho, rho_dot = xrho[:3], xrho[3:]
    rd = rc + NO.T @ rho
    vd = vc + NO.T @ (rho_dot + np.cross(omega, rho))
    return np.concatenate((rd, vd))