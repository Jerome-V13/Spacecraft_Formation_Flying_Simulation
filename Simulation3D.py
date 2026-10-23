"""
Simulation3D.py  -  Panda3D visualisation of the Chief / Deputy rendezvous & proximity-ops run
=============================================================================================
 
Run:      python Simulation3D.py            (needs: pip install panda3d numpy)
Options:  --start SECONDS   start at a given simulation time
          --speed N         initial speed multiplier (1,2,5,10,20,50,100,200,500)
          --size WxH        window size (default 1600x900)
          --no-cache        recompute the simulation instead of using sim_cache.npz
          --check           print a velocity-continuity check at burn 2 and at the CWH hand-over
 
Files expected next to this script:
    Orbital_Mechanics.py, Attitude_Dynamics.py     
    chief.bam / chief.egg, deputy.bam / deputy.egg 
    earth_texture.jpg                              
 
Layout / phases
---------------
  Phase 0  Transfer + drift (before the CWH controller starts)
             MAIN  : whole Earth, both orbit trails, exaggerated markers/labels
             INSET : camera radially above the deputy, looking down, following it in
  Stage 1  Safety spiral          MAIN: fixed on the chief (LVLH axes)   INSET: plots (error + control input)
  Stage 2  Hold at 25 m           MAIN: fixed on the chief               INSET: plots
  Stage 3  Approach to 5 m        (same)
  Stage 4  Hold at 2 m            (same)
  The layout switches automatically when the phase changes (C / V override it at any time).
  The MAIN camera glides smoothly between modes (TRANSITION_SECONDS); the inset switches instantly.
 
Controls
--------
  Space        play / pause                      Up / Down      speed up / down
  Left / Right seek -/+ 60 s (Shift = 10x)       Home           restart
  N / B        jump to next / previous stage     R              re-apply the automatic camera layout
  C            cycle MAIN camera                 V              cycle INSET camera
  X            swap MAIN <-> INSET cameras
  Mouse wheel / + / -   zoom the view under the cursor        Left-drag   orbit that view
  F            toggle auto-fit zoom on the view under the cursor (zoom follows the range)
  P            plots <-> inset camera view       [ / ]          plot time window shorter / longer
  A            body axes   T trails   L labels   G LVLH guides   H help   F2 FPS meter   Esc quit
 
Camera modes
------------
  EARTH            Earth-centred, inertial axes (whole orbit view)
  CHIEF  (LVLH)    centred on the chief, axes = radial / along-track / cross-track (R-bar up, V-bar right)
  CHIEF  (BODY)    centred on the chief, axes = chief body axes (the chief never moves in this view)
  DEPUTY (RADIAL)  centred on the deputy, camera on the radial line above it looking at the orbit
 
Numerics
--------
  The scene spans 6378 km (Earth) down to meters (satellites), far beyond float32.  Every frame all
  positions are computed in float64 relative to the camera's focus point ("floating origin"), and
  each view owns its own scene graph, so nothing near the camera is ever at a large coordinate.
"""
 
import os
import sys
import math
import time
import hashlib
import inspect
import argparse
import panda3d  # not used directly, only the submodules below
import numpy as np
 
# --------------------------------------------------------------------------------------------
# Helper modules
# --------------------------------------------------------------------------------------------
# Make the script work from any launch directory: helpers and models are looked up next to this file
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import Orbital_Mechanics as OM
import Attitude_Dynamics as AD
 
# --------------------------------------------------------------------------------------------
# Panda3D
# --------------------------------------------------------------------------------------------
from panda3d.core import (
    loadPrcFileData, unloadPrcFile, Filename, getModelPath,
    NodePath, PandaNode, Camera, PerspectiveLens, Vec3, Point3, LQuaternionf,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter, Geom, GeomNode,
    GeomLinestrips, GeomLines, GeomPoints, GeomTriangles,
    LineSegs, TextNode, CardMaker, AmbientLight, DirectionalLight,
    TransparencyAttrib, AntialiasAttrib, Texture, OmniBoundingVolume, WindowProperties
)
 
# ============================================================================================
#  SETTINGS
# ============================================================================================
# The simulation is self-contained, but the 3D view needs the models and textures in the same folder.
MODEL_DIR = HERE
CHIEF_MODEL = "chief"            # chief.bam  (falls back to chief.egg)
DEPUTY_MODEL = "deputy"          # deputy.bam (falls back to deputy.egg)
 
# Models are authored in metres, Z-up.  The simulation's approach axis is the chief body +x (b1),
# and the deputy points its +x at the chief.  The chief's docking face is +Y in the model file and
# the deputy's docking probe is on -Y, so we rotate each model about Z to put those on body +x.
CHIEF_H_OFFSET = -90.0           # model +Y  -> body +x
DEPUTY_H_OFFSET = +90.0          # model -Y  -> body +x
CHIEF_AXIS_LEN_M = 3.4           # length of the drawn body axes (metres)
DEPUTY_AXIS_LEN_M = 2.0
 
MIN_SAT_SCREEN_FRAC = 0.02       # A satellite is drawn at least this fraction of its camera distance
MARKER_SCREEN_FRAC = 0.018       # Radius of the constant-screen-size marker ring
VFOV_DEG = 40.0  # Vertical field of view of all cameras
 
TRANSITION_SECONDS = 5.0         # Real-time duration of a (main view) camera mode change
 
R_EARTH = 6378.137               # km
SIDEREAL_DAY_S = 86164.0905      # Earth rotation period relative to inertial space
SUN_DIR_INERTIAL = np.array([1.0, 0.35, 0.25]) / np.linalg.norm([1.0, 0.35, 0.25])  # arbitrary sun direction, only used for shading
 
SPEEDS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # speed multipliers for Up / Down keys
DEFAULT_SPEED_IDX = 6            # x100
 
# Mission configuration
# Mission parameters. Distances in km, mu in km^3/s^2, ddrift = along-track drift rate (km/s),
# yoffset = along-track offset (km) the deputy should end up at after the transfer.
CFG = dict(
    ac=7000.0, ad=6800.0, mu=398600.0, yoffset=-5.0, ddrift=-0.001, inc_deg=10.0, dt=1.0,
    inertia=(700.0, 600.0, 400.0), w_BN0_deg=(0.0, 0.005, 0.01),
    # Chief attitude is integrated from Stage 2 onward (before that it stays
    # aligned with LVLH).  Set True to let the chief tumble from t = 0 (this changes the Stage 2+ results).
    attitude_from_start=False,
)
 
# ============================================================================================
#  1.  SIMULATION  
# ============================================================================================
# Plain fixed-step RK4. f(t, x, *args).
def rk4_step(f, t, x, h, *args):
    k1 = f(t, x, *args)
    k2 = f(t + h / 2, x + h / 2 * k1, *args)
    k3 = f(t + h / 2, x + h / 2 * k2, *args)
    k4 = f(t + h, x + h * k3, *args)
    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
 
 
def _prop_attitude(att, t, h, I, w_HN_H):
    # one RK4 step of [MRP, body rates]
    att = rk4_step(AD.chief_attitude_eom, t, att, h, I, w_HN_H)
    s = att[:3]
    s2 = s @ s
    if s2 > 1.0:                         # MRP shadow-set switch, keeps |sigma| <= 1
        att[:3] = -s / s2
    return att
 
 
def _body_fixed_reference(att, w_HN_H, d_vec, d_dot_vec=None):
    """Reference (rho*, rho*_dot) in H, for a point fixed (or sliding) along the chief body axes."""
    # w_BH_B = body rate relative to LVLH. The offset d is fixed in the body, so it rotates with it
    C_BH = AD.mrp_to_dcm(att[:3])
    w_BH_B = att[3:] - C_BH @ w_HN_H
    C_HB = C_BH.T
    rho_star = C_HB @ d_vec
    # d_dot_vec is only needed when the point also slides along the body axis (approach ramps)
    if d_dot_vec is None:
        rho_dot = C_HB @ np.cross(w_BH_B, d_vec)
    else:
        rho_dot = C_HB @ (np.cross(w_BH_B, d_vec) + d_dot_vec)
    return np.concatenate((rho_star, rho_dot))
 
 
def _ramp(t, d0, dfinal, rate):
    """Same as Brho_time() in Simulation.py (NOTE: it is a function of ABSOLUTE time t)."""
    # time at which the ramp reaches its final distance
    t_end = (d0 - dfinal) / rate
    if t <= t_end:
        # still moving: distance shrinks linearly
        return np.array([d0 - rate * t, 0.0, 0.0]), np.array([-rate, 0.0, 0.0])
    # ramp finished: hold the final distance, zero velocity
    return np.array([dfinal, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
 
 
def run_simulation(cfg=CFG):
    ac, ad, mu = cfg["ac"], cfg["ad"], cfg["mu"]
    yoffset, ddrift, dt = cfg["yoffset"], cfg["ddrift"], cfg["dt"]
 
    # chief mean motion (circular orbit)
    nc = np.sqrt(mu / ac ** 3)
    # radial offset that gives the requested drift rate (CWH drift term: ydot = -3/2 n x)
    xoffset = -(2 * ddrift) / (3 * nc)
    # semi-major axis of the drift orbit
    a_drift = ac + xoffset
    # Hohmann transfer from the deputy's start orbit (ad) to the drift orbit: dv1 first...
    dv1 = np.sqrt(mu / ad) * (np.sqrt(2 * a_drift / (ad + a_drift)) - 1)
    # ...and dv2 to circularise at the end
    dv2 = np.sqrt(mu / a_drift) * (1 - np.sqrt(2 * ad / (ad + a_drift)))
    a_transfer = (ad + a_drift) / 2
    # time of flight = half the transfer ellipse period
    tof = np.pi * np.sqrt(a_transfer ** 3 / mu)
    # initial deputy true anomaly: sets the phase so it arrives yoffset away from the chief
    aof = tof * nc - np.pi + (yoffset / a_drift)
    inc = np.deg2rad(cfg["inc_deg"])
 
    # initial inertial states from orbital elements [a, e, i, raan, argp, ta]
    X0, Xdot0 = OM.oe2cc([ac, 0, inc, 0, 0, 0], mu)
    Xd0, Xdot0d = OM.oe2cc([ad, 0, inc, 0, 0, aof], mu)
 
    rad = np.pi / 180.0
    # inertia tensor (diagonal = principal axes)
    I = np.diag(cfg["inertia"])
    # attitude state [MRP (3), body rates (3)]; zero MRP = body aligned with LVLH
    att = np.concatenate((np.zeros(3), np.array(cfg["w_BN0_deg"]) * rad))
    # LVLH angular velocity wrt inertial, in LVLH components: just n about the orbit normal
    w_HN_H = np.array([0.0, 0.0, nc])
 
    # T_chief is chief orbital period; the run lasts 7 * T_chief
    T_chief = 2 * np.pi * np.sqrt(ac ** 3 / mu)
    total_time = T_chief * 7
 
    Xc = np.concatenate((X0.ravel(), Xdot0.ravel()))
    vd = Xdot0d.ravel()
    # first burn, along the current velocity direction
    burn_dv1 = dv1 * vd / np.linalg.norm(vd)
    burn_t = [0.0]
    burn_r = [Xd0.ravel().copy()]
    burn_dv = [burn_dv1.copy()]
    vd = vd + burn_dv1
    Xd = np.concatenate((Xd0.ravel(), vd))
 
    # --- Time Loop Bookkeeping ---
    t = 0.0
    burn2_done = False
    rho_state = None
    stage = 1
    no_time_for_caution = 0  # steps spent inside 5 m of the reference in stage 2
    n_final = 0  # same, for stage 3
    cwh_idx = None  # history index where the CWH controller takes over
 
    # Stage-1 reference
    ref_params_r1 = [0.6, 0.6, 10 * np.pi / 180, -80 * np.pi / 180, -0.06, -2.0]   # A0,B0,alpha,beta,xoff,yoff
    K1, P1 = 0.00001, 0.004
 
    # History buffers, turned into arrays at the end
    T = [0.0]; Ch = [Xc.copy()]; Dh = [Xd.copy()]; Ah = [att.copy()]; Sh = [stage]
    Uh = [np.zeros(3)]; Eh = [np.zeros(3)]
    s1_u = np.zeros(3)
    s1_dx_err = np.zeros(3)
 
    # Main loop
    while t < total_time:
        h = dt
        # shorten the step so we land exactly at tof, where the 2nd burn happens
        land = (not burn2_done) and (t + h >= tof)
        if land:
            h = tof - t
 
        Xc = rk4_step(OM.inertial_eom, t, Xc, h, mu)
 
        # Chief attitude only runs from t = 0 if enabled in CFG (otherwise from stage 2)
        if cfg["attitude_from_start"] and (rho_state is None or stage == 1):
            att = _prop_attitude(att, t, h, I, w_HN_H)
 
        newly_cwh = False
        # rho_state is None -> still on the inertial two-body propagation (transfer + drift)
        if rho_state is None:
            Xd = rk4_step(OM.inertial_eom, t, Xd, h, mu)
            # 100 s after the second burn, switch to the relative (CWH) model
            if t + h > tof + 100:                       # switch to CWH here
                rho_state = OM.inertial_to_lvlh(Xc, Xd)
                newly_cwh = True
        else:
            # stage 1: track the analytical CWH spiral reference
            if stage == 1:
                ref = OM.cwh_analytical(t, ref_params_r1, nc)
                rho_state = rk4_step(OM.cwh_equations_controlled, t, rho_state, h, nc, ref, K1, P1, stage)
                s1_u, s1_dx_err, _ = OM.cwh_control(rho_state, nc, ref, K1, P1, stage)
 
            else:
                # stages 2-4: the reference is a point fixed to the chief body, so the chief attitude is needed
                att = _prop_attitude(att, t, h, I, w_HN_H)
 
                if stage == 2:
                    # stage 2: hold 25 m along the chief +x axis
                    ref = _body_fixed_reference(att, w_HN_H, np.array([25.0, 0.0, 0.0]) / 1000.0)
                elif stage == 3:
                    # stage 3: approach from 25 m to 5 m at 2 mm/s
                    # NOTE: _ramp uses absolute t, double check the ramp really starts when the stage does
                    d_vec, d_dot = _ramp(t, 25.0 / 1000.0, 5.0 / 1000.0, 0.002 / 1000.0)
                    ref = _body_fixed_reference(att, w_HN_H, d_vec, d_dot)
                else:
                    # stage 4: 5 m -> 2 m at 1 mm/s (same absolute-time caveat)
                    d_vec, d_dot = _ramp(t, 5.0 / 1000.0, 2.0 / 1000.0, 0.0005 / 1000.0)
                    ref = _body_fixed_reference(att, w_HN_H, d_vec, d_dot)
 
                rho_state = rk4_step(OM.cwh_equations_controlled, t, rho_state, h, nc, ref, K1, P1, stage)
                s1_u, s1_dx_err, _ = OM.cwh_control(rho_state, nc, ref, K1, P1, stage)

                # Count how long the error stays under 5 m, used to trigger the next stage
                err = np.linalg.norm(s1_dx_err)
                if stage == 2 and err < 0.005:
                    no_time_for_caution += 1
                if stage == 3 and err < 0.005:
                    n_final += 1
 
            # keep the inertial deputy state updated, the 3D view needs it
            Xd = OM.lvlh_to_inertial(Xc, rho_state)
 
            # stage switching: 1 -> 2 on time, 2 -> 3 and 3 -> 4 when the counters reach their thresholds
            if t > T_chief * 4:
                stage = 2
            if no_time_for_caution > 1800:
                stage = 3
            if n_final > 1800:
                stage = 4
 
        t += h
 
        if land:
            t = tof
            # Second Hohmann burn, again along the velocity direction
            vel = Xd[3:]
            burn_dv2 = dv2 * vel / np.linalg.norm(vel)
            burn_t.append(t)
            burn_r.append(Xd[:3].copy())
            burn_dv.append(burn_dv2.copy())
            Xd[3:] = vel + burn_dv2
            burn2_done = True
 
        # Store this step
        T.append(t); Ch.append(Xc.copy()); Dh.append(Xd.copy()); Ah.append(att.copy()); Sh.append(stage)
        Uh.append(np.array(s1_u)); Eh.append(np.array(s1_dx_err))
        if newly_cwh:
            cwh_idx = len(T) - 1
 
    class R:  # Simple namespace
        pass
    r = R()
    r.T = np.array(T); r.C = np.array(Ch); r.D = np.array(Dh); r.att = np.array(Ah)
    r.stage = np.array(Sh); r.u = np.array(Uh); r.err = np.array(Eh)
    r.burn_t = np.array(burn_t); r.burn_r = np.array(burn_r); r.burn_dv = np.array(burn_dv)
    r.tof = float(tof); r.T_chief = float(T_chief); r.nc = float(nc)
    r.ac = float(ac); r.ad = float(ad); r.cwh_idx = int(cwh_idx if cwh_idx is not None else 0)
    return r
 
 
def _cache_key():
    # Hash of the helper files, the sim code and CFG: the cache is rebuilt automatically when any of them change
    h = hashlib.sha1()
    for fn in ("Orbital_Mechanics.py", "Attitude_Dynamics.py"):
        with open(os.path.join(HERE, fn), "rb") as f:
            h.update(f.read())
    h.update(inspect.getsource(run_simulation).encode())
    h.update(inspect.getsource(_prop_attitude).encode())
    h.update(inspect.getsource(_body_fixed_reference).encode())
    h.update(inspect.getsource(_ramp).encode())
    h.update(repr(sorted(CFG.items())).encode())
    return h.hexdigest()
 
 
def load_or_run(use_cache=True):
    # propagation takes a few seconds, so the result is cached in sim_cache.npz
    path = os.path.join(HERE, "sim_cache.npz")
    key = _cache_key()
    if use_cache and os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["key"]) == key:
                class R: pass
                r = R()
                for k in ("T", "C", "D", "att", "stage", "u", "err", "burn_t", "burn_r", "burn_dv"):
                    setattr(r, k, z[k])
                for k in ("tof", "T_chief", "nc", "ac", "ad"):
                    setattr(r, k, float(z[k]))
                r.cwh_idx = int(z["cwh_idx"])
                print("[sim] loaded cached simulation (%d samples)" % len(r.T))
                return r
        except Exception as e:                                       # noqa
            print("[sim] cache unreadable (%s) - recomputing" % e)
    print("[sim] propagating the mission (about 10 s, cached afterwards)...")
    t0 = time.time()
    r = run_simulation()
    print("[sim] done in %.1f s, %d samples, T_end = %.0f s" % (time.time() - t0, len(r.T), r.T[-1]))
    try:
        np.savez(path, key=key, T=r.T, C=r.C, D=r.D, att=r.att, stage=r.stage, u=r.u, err=r.err,
             burn_t=r.burn_t, burn_r=r.burn_r, burn_dv=r.burn_dv,
                 tof=r.tof, T_chief=r.T_chief, nc=r.nc, ac=r.ac, ad=r.ad, cwh_idx=r.cwh_idx)
    except Exception as e:                                           # noqa
        print("[sim] could not write cache:", e)
    return r
 
 
# ============================================================================================
#  2.  MATH HELPERS & DERIVED DATA
# ============================================================================================
def mat_to_quat(R):
    """Rotation matrix (active, column vectors) -> quaternion (w, x, y, z)."""
    # trace-based conversion; the branches pick the largest term to avoid dividing by something small
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return q / np.linalg.norm(q)
 
 
def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
 
 
# Converts the whole attitude history and flips signs so neighbouring quaternions stay on the same
# side (q and -q are the same rotation; interpolating across a flip would spin the long way round)
def rots_to_quats(Rs):
    q = np.empty((len(Rs), 4))
    prev = None
    for i, R in enumerate(Rs):
        qi = mat_to_quat(R)
        if prev is not None and np.dot(prev, qi) < 0:
            qi = -qi
        q[i] = qi
        prev = qi
    return q
 
 
def normalize(v):
    return v / np.linalg.norm(v)
 
 
# Rodrigues' formula, used for the orbit-camera yaw/pitch
def rot_axis(axis, ang):
    a = normalize(np.asarray(axis, float))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
 
 
# ---- helpers for the smooth camera transition ---------------------------------------------
def smoothstep(x):
    # ease-in / ease-out: 0 -> 1 with zero slope at both ends
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)
 
 
def slerp_vec(a, b, s):
    """Spherical interpolation between two directions."""
    a, b = normalize(a), normalize(b)
    ang = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    if ang < 1e-6:
        return a
    if math.pi - ang < 1e-3:   # nearly opposite: swing around a perpendicular axis
        perp = normalize(np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0]))
        return rot_axis(perp, math.pi * s) @ a
    return (math.sin((1 - s) * ang) * a + math.sin(s * ang) * b) / math.sin(ang)
 
 
def slerp_mat(Qa, Qb, s):
    """Interpolate two rotation matrices through their quaternions (normalised lerp)."""
    qa, qb = mat_to_quat(Qa), mat_to_quat(Qb)
    if qa @ qb < 0:
        qb = -qb
    return quat_to_mat(qa * (1 - s) + qb * s)
 
 
def hermite(x0, x1, h, s):
    """Cubic Hermite interpolation of a 6-state (pos, vel) between two samples; returns pos, vel."""
    p0, v0, p1, v1 = x0[:3], x0[3:], x1[:3], x1[3:]
    if h <= 0:
        return p1.copy(), v1.copy()
    # Hermite basis for the position; the second expression is its derivative (velocity)
    s2, s3 = s * s, s * s * s
    p = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * v1
    v = ((6 * s2 - 6 * s) * p0 / h + (3 * s2 - 4 * s + 1) * v0 + (-6 * s2 + 6 * s) * p1 / h + (3 * s2 - 2 * s) * v1)
    return p, v
 
 
def fmt_time(t):
    t = int(round(t))
    return "%02d:%02d:%02d" % (t // 3600, (t // 60) % 60, t % 60)
 
 
# km in, human-readable string out (km or m)
def fmt_len(km):
    if km >= 10:
        return "%.1f km" % km
    if km >= 1:
        return "%.2f km" % km
    return "%.2f m" % (km * 1000) if km * 1000 < 100 else "%.0f m" % (km * 1000)
 
 
class SimData:
    """History + derived quantities + interpolated sampling."""
    PHASE_NAMES = {
        0: "HOHMANN TRANSFER",
        1: "STAGE 1  -  SAFETY SPIRAL",
        2: "STAGE 2  -  HOLD AT 25 m",
        3: "STAGE 3  -  HOLD AT 5 m",
        4: "STAGE 4  -  HOLD AT 2 m",
    }
 
    def __init__(self, r):
        self.__dict__.update(r.__dict__)
        T, C, D = self.T, self.C, self.D
        N = len(T)
        self.N = N
        # Phase = stage, except everything before the CWH controller starts is phase 0
        self.phase = self.stage.copy()
        self.phase[:self.cwh_idx] = 0
 
        rc, vc = C[:, :3], C[:, 3:]
        # LVLH axes for every sample (same construction as OM.lvlh_dcm, just vectorised)
        o_r = rc / np.linalg.norm(rc, axis=1, keepdims=True)
        h = np.cross(rc, vc)
        o_h = h / np.linalg.norm(h, axis=1, keepdims=True)
        o_t = np.cross(o_h, o_r)
        self.NO = np.stack([o_r, o_t, o_h], axis=1)                      # inertial -> LVLH (rows)
        self.rho_H = np.einsum('nij,nj->ni', self.NO, D[:, :3] - rc)     # km, LVLH components
        C_BH = np.array([AD.mrp_to_dcm(s) for s in self.att[:, :3]])
        self.rho_B = np.einsum('nij,nj->ni', C_BH, self.rho_H)           # km, chief-body components
        R_NB = self.NO.transpose(0, 2, 1) @ C_BH.transpose(0, 2, 1)      # body -> inertial
 
        # Deputy attitude (not simulated in Simulation.py): +x points at the chief, as in your animations
        x = -self.rho_B / np.linalg.norm(self.rho_B, axis=1, keepdims=True)
        y = np.empty_like(x)
        for i, xi in enumerate(x):
            if i == 0:
                # Start with the cardinal axis least aligned with x.
                ref = np.eye(3)[np.argmin(np.abs(xi))]
                yi = np.cross(ref, xi)
            else:
                # Project the previous y-axis onto the plane perpendicular to x.
                yi = y[i - 1] - np.dot(y[i - 1], xi) * xi
                if np.linalg.norm(yi) < 1e-8:
                    z_prev = np.cross(x[i - 1], y[i - 1])
                    yi = np.cross(z_prev, xi)
            y[i] = yi / np.linalg.norm(yi)
        z = np.cross(x, y)
        R_BD = np.stack([x, y, z], axis=2)
        R_ND = R_NB @ R_BD
        # Body -> inertial quaternions for both spacecraft
        self.q_c = rots_to_quats(R_NB)
        self.q_d = rots_to_quats(R_ND)
 
        self.err_m = self.err * 1000.0                                   # m
        self.u_mm = self.u * 1.0e6                                       # mm/s^2
        self.err_norm_m = np.linalg.norm(self.err_m, axis=1)
        self.u_norm_mm = np.linalg.norm(self.u_mm, axis=1)
        self.burn_dv_m_s = np.linalg.norm(self.burn_dv, axis=1) * 1000.0
 
        # Start time of each phase, used by the N/B jump keys and the plot markers
        self.phase_start = {}
        for p in range(5):
            idx = np.where(self.phase == p)[0]
            if len(idx):
                self.phase_start[p] = float(T[idx[0]])
 
        # Decimated trail sources (<= ~6000 points)
        stride = max(1, int(math.ceil(N / 6000.0)))
        ids = np.arange(0, N, stride)
        if ids[-1] != N - 1:
            ids = np.append(ids, N - 1)
        self.tr_idx = ids
        self.tr_T = T[ids]
        self.tr_C = C[ids, :3]
        self.tr_D = D[ids, :3]
        # Relative-motion trails only exist once the controller is running
        rel = ids[ids >= self.cwh_idx]
        self.rel_T = T[rel]
        self.rel_H = self.rho_H[rel]
        self.rel_B = self.rho_B[rel]
 
    class Snap:
        pass
 
    def sample(self, t):
        T = self.T
        t = float(np.clip(t, T[0], T[-1]))
        # find the interval and the fraction s inside it. Positions use Hermite (smooth even at x5000 speed),
        # attitude uses a normalised linear blend (nlerp), good enough for steps this small
        i = int(np.clip(np.searchsorted(T, t, 'right') - 1, 0, self.N - 2))
        h = T[i + 1] - T[i]
        s = 0.0 if h <= 0 else min(1.0, max(0.0, (t - T[i]) / h))
        S = SimData.Snap()
        S.t, S.i, S.s = t, i, s
        S.rc, S.vc = hermite(self.C[i], self.C[i + 1], h, s)
        S.rd, S.vd = hermite(self.D[i], self.D[i + 1], h, s)
        qc = self.q_c[i] * (1 - s) + self.q_c[i + 1] * s
        qd = self.q_d[i] * (1 - s) + self.q_d[i + 1] * s
        S.qc = qc / np.linalg.norm(qc)
        S.qd = qd / np.linalg.norm(qd)
        S.NOc = OM.lvlh_dcm(S.rc, S.vc)[0]
        S.phase = int(self.phase[i])
        return S
 
 
def check_continuity(sim, idx, label, k=4):
    """Prints the finite-difference relative velocity (LVLH, m/s) around a history index."""
    sl = slice(max(0, idx - k), idx + k + 1)
    rel = sim.rho_H[sl] * 1000.0                                  # m, LVLH
    vrel = np.diff(rel, axis=0) / np.diff(sim.T[sl])[:, None]     # m/s
    print("[check]", label, "(history index %d, t = %.1f s)" % (idx, sim.T[idx]))
    print(np.round(vrel, 3))
 
 
# ============================================================================================
#  3.  PANDA3D GEOMETRY HELPERS
# ============================================================================================
class DynLine:
    """Line strip / line-segment set with a FIXED vertex count.
 
    The primitive is built once; every update only overwrites the vertex array.  Unused tail
    vertices repeat the last point (zero-length segments), so nothing is drawn for them and Panda's
    geometry caches never see the primitive change."""
 
    def __init__(self, parent, color, thickness=1.5, segments=False, capacity=64, name="dynline"):
        self.capacity = max(2, int(capacity) + (int(capacity) % 2 if segments else 0))
        self.segments = segments
        # fixed-size dynamic vertex buffer (see class docstring)
        self.vdata = GeomVertexData(name, GeomVertexFormat.getV3(), Geom.UHDynamic)
        self.vdata.setNumRows(self.capacity)
        self.vdata.modifyArray(0).modifyHandle().copyDataFrom(np.zeros((self.capacity, 3), np.float32))
        self.prim = GeomLines(Geom.UHDynamic) if segments else GeomLinestrips(Geom.UHDynamic)
        self.prim.addConsecutiveVertices(0, self.capacity)
        self.prim.closePrimitive()
        geom = Geom(self.vdata)
        geom.addPrimitive(self.prim)
        node = GeomNode(name)
        node.addGeom(geom)
        node.setBounds(OmniBoundingVolume())  # vertices change every frame, don't cull on stale bounds
        node.setFinal(True)
        self.np = parent.attachNewNode(node)
        self.np.setColor(*color)
        self.np.setRenderModeThickness(thickness)
        self.np.setLightOff()
        self.np.setTextureOff()
        if color[3] < 1.0:
            self.np.setTransparency(TransparencyAttrib.MAlpha)
        self._wanted = True          # caller's visibility request
        self._empty = True           # nothing to draw yet
        self._buf = np.zeros((self.capacity, 3), np.float32)
        self._apply()
 
    def _apply(self):
        if self._wanted and not self._empty:
            self.np.show()
        else:
            self.np.hide()
 
    def set_points(self, pts):
        n = min(len(pts), self.capacity)
        if self.segments:
            n -= n % 2
        if n < 2:
            self.clear()
            return
        self._buf[:n] = pts[:n]
        self._buf[n:] = self._buf[n - 1]  # pad the unused tail with the last point
        self.vdata.modifyArray(0).modifyHandle().copyDataFrom(self._buf)
        self._empty = False
        self._apply()
 
    def clear(self):
        self._empty = True
        self._apply()
 
    def show(self, flag=True):
        self._wanted = bool(flag)
        self._apply()
 
 
def make_text(parent, text, color, scale, align=TextNode.ACenter, pos=(0, 0, 0)):
    tn = TextNode("txt")
    tn.setText(text)
    tn.setTextColor(*color)
    tn.setAlign(align)
    tn.setShadow(0.0, 0.0)
    tn.setShadowColor(0, 0, 0, 0.0)
    np_ = parent.attachNewNode(tn)
    np_.setScale(scale)
    np_.setPos(*pos)
    np_.setLightOff()
    np_.setTwoSided(True)
    return np_, tn
 
 
def make_sphere(radius, nlat=96, nlon=192):
    # lat/lon grid -> vertices (position, normal, uv); the last column repeats the seam so the texture wraps cleanly
    lat = np.linspace(np.pi / 2, -np.pi / 2, nlat + 1)
    lon = np.linspace(0, 2 * np.pi, nlon + 1)
    LON, LAT = np.meshgrid(lon, lat)
    x = np.cos(LAT) * np.cos(LON)
    y = np.cos(LAT) * np.sin(LON)
    z = np.sin(LAT)
    u = LON / (2 * np.pi)
    v = (LAT + np.pi / 2) / np.pi
    verts = np.column_stack([x.ravel() * radius, y.ravel() * radius, z.ravel() * radius,
                             x.ravel(), y.ravel(), z.ravel(), u.ravel(), v.ravel()]).astype(np.float32)
    # two triangles per grid cell
    ii, jj = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing='ij')
    a = ii * (nlon + 1) + jj
    b = (ii + 1) * (nlon + 1) + jj
    c = (ii + 1) * (nlon + 1) + jj + 1
    d = ii * (nlon + 1) + jj + 1
    idx = np.stack([a, b, c, a, c, d], axis=-1).reshape(-1).astype(np.uint32)
 
    vdata = GeomVertexData("earth", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
    vdata.setNumRows(len(verts))
    vdata.modifyArray(0).modifyHandle().copyDataFrom(verts)
    tri = GeomTriangles(Geom.UHStatic)
    tri.setIndexType(Geom.NTUint32)
    tri.modifyVertices().modifyHandle().copyDataFrom(idx)
    geom = Geom(vdata)
    geom.addPrimitive(tri)
    node = GeomNode("earth")
    node.addGeom(geom)
    return NodePath(node)
 
def make_stars(n=2500, seed=3):
    rng = np.random.default_rng(seed)
    # normalised gaussians = uniform random directions on the sphere
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    br = rng.random(n) ** 3  # cubed: mostly dim stars, a few bright ones
    vdata = GeomVertexData("stars", GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vdata.setNumRows(n)
    wv, wc = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "color")
    for p, b in zip(v, br):
        wv.addData3(*p)
        g = 0.35 + 0.65 * b
        wc.addData4(g, g, min(1.0, g * 1.05), 1.0)
    pts = GeomPoints(Geom.UHStatic)
    pts.addConsecutiveVertices(0, n)
    pts.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(pts)
    node = GeomNode("stars")
    node.addGeom(geom)
    node.setBounds(OmniBoundingVolume())
    node.setFinal(True)
    npath = NodePath(node)
    npath.setRenderModeThickness(1.8)
    npath.setLightOff()
    npath.setTextureOff()
    npath.setBin("background", 0)  # draw first, behind everything
    npath.setDepthWrite(False)
    npath.setDepthTest(False)
    return npath
 
 
FALLBACK_COLORS = [   # used only when the model's texture files (tex/*.png) are not found
    ("solar", (0.08, 0.16, 0.42)), ("antenna_dish", (0.92, 0.92, 0.94)), ("antenna", (0.70, 0.70, 0.74)),
    ("dock_marker", (0.98, 0.95, 0.25)), ("dock_collar", (0.95, 0.50, 0.10)), ("dock_funnel", (0.35, 0.36, 0.42)),
    ("dock_port", (0.95, 0.50, 0.10)), ("startracker", (0.12, 0.12, 0.14)), ("thruster", (0.28, 0.28, 0.31)),
    ("yoke", (0.55, 0.55, 0.60)), ("probe_target", (0.90, 0.15, 0.15)), ("probe_head", (0.85, 0.85, 0.88)),
    ("probe", (0.65, 0.65, 0.70)),
]
 
 
def load_sat_model(loader, name, bus_color):
    """Loads <name>.bam (or .egg) from MODEL_DIR; returns (nodepath, max extent in metres)."""
    getModelPath().appendDirectory(Filename.fromOsSpecific(MODEL_DIR))
    # the models expect their textures in tex/; if it's missing we colour the parts by name instead
    have_tex = os.path.isdir(os.path.join(MODEL_DIR, "tex"))
    page = None
    if not have_tex:
        page = loadPrcFileData("", "notify-level-gobj fatal")           # silence 'texture not found' spam
    model = None
    for ext in (".bam", ".egg"):
        p = os.path.join(MODEL_DIR, name + ext)
        if os.path.exists(p):
            try:
                model = loader.loadModel(Filename.fromOsSpecific(p), noCache=True)
                break
            except Exception as e:                                       # noqa
                print("[model] could not load %s: %s" % (p, e))
    if page is not None:
        unloadPrcFile(page)
    if model is None or model.isEmpty():
        raise SystemExit("Could not load %s.bam / %s.egg from %s" % (name, name, MODEL_DIR))
    if not have_tex or not model.findAllTextures():
        # pick a colour from keywords in the geom name
        for gn in model.findAllMatches("**/+GeomNode"):
            nm = gn.getName().lower()
            col = bus_color if "bus" in nm else None
            for key, c in FALLBACK_COLORS:
                if key in nm:
                    col = c
                    break
            if col is None:
                col = (0.7, 0.7, 0.7)
            gn.setMaterialOff(1)
            gn.setColor(col[0], col[1], col[2], 1, 1)
    # largest dimension in metres, used for the screen-size exaggeration
    a, b = model.getTightBounds()
    ext = float(max(b[0] - a[0], b[1] - a[1], b[2] - a[2]))
    return model, ext
 
 
# ============================================================================================
#  4.  SATELLITE VISUAL (model + body axes + constant-screen-size marker & label)
# ============================================================================================
AXIS_COLORS = [(1.0, 0.18, 0.18, 1), (0.22, 1.0, 0.22, 1), (0.30, 0.55, 1.0, 1)]  # x = red, y = green, z = blue
 
 
class SatVisual:
    def __init__(self, root, template, extent_m, name, color, h_offset, axis_len, label, label_above):
        self.extent_km = extent_m * 1e-3
        self.name = name
        self.label = label
        self.root = root.attachNewNode(name + "_body")             # pos / quat / scale set every frame
        self.model = template.copyTo(self.root)
        self.model.setH(h_offset)
        self.axes = self.root.attachNewNode(name + "_axes")
        self._build_axes(axis_len)
 
        # marker ring + label keep a constant size on screen and always face the camera (billboard)
        self.marker = root.attachNewNode(name + "_marker")         # world pos only
        self.marker_scale = self.marker.attachNewNode("s")         # uniform scale = screen constant
        bb = self.marker_scale.attachNewNode("bb")
        bb.setBillboardPointEye()
        ls = LineSegs()
        ls.setThickness(2.0)
        ls.setColor(*color)
        n = 28
        for k in range(n + 1):
            a = 2 * math.pi * k / n
            (ls.moveTo if k == 0 else ls.drawTo)(math.cos(a), 0, math.sin(a))
        self.ring = bb.attachNewNode(ls.create())
        self.ring.setLightOff()
        self.label_np, self.label_tn = make_text(
            bb, label, color, 0.95, pos=(0, 0, 1.45 if label_above else -1.5))
        self.marker_scale.setBin("fixed", 50)
        self.marker_scale.setDepthWrite(False)
        self._last_label = label
        self._last_t = -1.0
 
    def _build_axes(self, L):
        ls = LineSegs()
        ls.setThickness(3.0)
        for k in range(3):
            e = np.zeros(3); e[k] = L
            ls.setColor(*AXIS_COLORS[k])
            ls.moveTo(0, 0, 0); ls.drawTo(*e)
            others = [j for j in range(3) if j != k]
            for j in others:                                  # arrow head barbs
                for sgn in (-1, 1):
                    p = e * 0.88
                    p[j] += sgn * 0.05 * L
                    ls.moveTo(*e); ls.drawTo(*p)
        np_ = self.axes.attachNewNode(ls.create())
        np_.setLightOff()
        for k, nm in enumerate("xyz"):
            pos = np.zeros(3); pos[k] = L * 1.13
            holder = self.axes.attachNewNode("lbl")
            holder.setPos(*pos)
            holder.setBillboardPointEye()
            make_text(holder, nm, AXIS_COLORS[k], 0.30 * L / 2.0)
 
    def update(self, p, R_view, dcam, sat_label_extra=None, now=0.0):
        # far from the camera the real model would be sub-pixel, so scale it up to MIN_SAT_SCREEN_FRAC of the distance
        exag = max(1.0, MIN_SAT_SCREEN_FRAC * dcam / self.extent_km)
        self.root.setPos(*p)
        self.root.setQuat(LQuaternionf(*mat_to_quat(R_view)))
        self.root.setScale(1e-3 * exag)  # model is in metres, the scene is in km
        self.marker.setPos(*p)
        self.marker_scale.setScale(max(dcam, 1e-7) * MARKER_SCREEN_FRAC)
        # update the range text at most 10 times per second
        if sat_label_extra is not None and now - self._last_t > 0.1:
            txt = self.label + "\n" + sat_label_extra
            if txt != self._last_label:
                self.label_tn.setText(txt)
                self._last_label = txt
            self._last_t = now
 
 
# ============================================================================================
#  5.  WORLD VIEW  (own scene graph + camera + display region)
# ============================================================================================
EARTH, DEPUTY_RADIAL, CHIEF_LVLH, CHIEF_BODY = range(4)
MODE_NAMES = {EARTH: "EARTH / ORBITS", DEPUTY_RADIAL: "DEPUTY - RADIAL (NADIR)",
              CHIEF_LVLH: "CHIEF - LVLH FRAME", CHIEF_BODY: "CHIEF - BODY FRAME"}
MODE_CYCLE = [EARTH, CHIEF_LVLH, CHIEF_BODY, DEPUTY_RADIAL]
 
# per-mode camera defaults: start direction and up vector (in the view frame), min/max distance in km
MODE_CFG = {
    EARTH:         dict(base_dir=(0.80, -0.85, 0.55), base_up=(0, 0, 1), dmin=7500.0, dmax=250000.0),
    DEPUTY_RADIAL: dict(base_dir=(1.0, 0.0, 0.0),     base_up=(0, 1, 0), dmin=0.005,  dmax=30000.0),
    CHIEF_LVLH:    dict(base_dir=(0.30, -0.45, -0.85), base_up=(1, 0, 0), dmin=0.0008, dmax=3000.0),
    CHIEF_BODY:    dict(base_dir=(0.30, -0.90, 0.45), base_up=(0, 0, 1), dmin=0.0008, dmax=3000.0),
}
 
# phase -> (main camera, inset camera)
AUTO_LAYOUT = {
    0: (EARTH, DEPUTY_RADIAL),
    1: (CHIEF_LVLH, EARTH),
    2: (CHIEF_LVLH, EARTH),
    3: (CHIEF_LVLH, EARTH),
    4: (CHIEF_LVLH, EARTH),
}
 
CHIEF_COLOR = (1.0, 0.45, 0.25, 1.0)
DEPUTY_COLOR = (0.35, 0.85, 1.0, 1.0)
 
 
# One WorldView = one display region with its own scene graph and camera. The main view and the
# inset are two instances, so each one can show a different mode at the same time.
class WorldView:
    def __init__(self, app, name, bounds, sort, assets, sim):
        self.app, self.name, self.sim = app, name, sim
        self.bounds = bounds
        self.root = NodePath(name)
        self.root.setAntialias(AntialiasAttrib.MAuto)
 
        # camera & display region ------------------------------------------------------------
        self.lens = PerspectiveLens()
        self.lens.setNearFar(1.0, 1e5)
        self.cam = Camera(name + "_cam")
        self.cam.setLens(self.lens)
        self.cam_np = self.root.attachNewNode(self.cam)
        self.dr = app.win.makeDisplayRegion(*bounds)
        self.dr.setSort(sort)  # higher sort draws on top (inset over main)
        self.dr.setCamera(self.cam_np)
        self.dr.setClearColorActive(True)
        self.dr.setClearColor((0.0, 0.0, 0.012, 1.0))
        self.dr.setClearDepthActive(True)
 
        # lights -----------------------------------------------------------------------------
        amb = AmbientLight(name + "_amb")
        amb.setColor((0.20, 0.22, 0.28, 1))
        self.root.setLight(self.root.attachNewNode(amb))
        sun = DirectionalLight(name + "_sun")
        sun.setColor((1.20, 1.15, 1.05, 1))
        self.sun_np = self.root.attachNewNode(sun)
        self.root.setLight(self.sun_np)
 
        # scene content ----------------------------------------------------------------------
        self.stars = assets.stars.copyTo(self.root)
        self.earth = assets.earth.copyTo(self.root)
        self.chief = SatVisual(self.root, assets.chief, assets.chief_ext, "chief", CHIEF_COLOR,
                               CHIEF_H_OFFSET, CHIEF_AXIS_LEN_M, "CHIEF", True)
        self.deputy = SatVisual(self.root, assets.deputy, assets.deputy_ext, "deputy", DEPUTY_COLOR,
                                DEPUTY_H_OFFSET, DEPUTY_AXIS_LEN_M, "DEPUTY", False)
 
        cap = len(sim.tr_C) + 2  # room for every trail sample plus the current position
        self.trail_c = DynLine(self.root, (1.0, 0.40, 0.22, 1.0), 2.2, capacity=cap, name="trail_c")
        self.trail_d = DynLine(self.root, (0.30, 0.85, 1.0, 1.0), 2.2, capacity=cap, name="trail_d")
        self.burn_markers = []
        self.burn_vectors = []
        for i in range(len(sim.burn_t)):
            marker = self.root.attachNewNode("burn_marker_%d" % (i + 1))
            scale = marker.attachNewNode("burn_marker_scale")
            billboard = scale.attachNewNode("burn_marker_billboard")
            billboard.setBillboardPointEye()
            ring = LineSegs()
            ring.setThickness(2.5)
            ring.setColor(1.0, 0.68, 0.18, 1.0)
            for k in range(33):
                angle = 2 * math.pi * k / 32
                (ring.moveTo if k == 0 else ring.drawTo)(0.78 * math.cos(angle), 0, 0.78 * math.sin(angle))
            ring_np = billboard.attachNewNode(ring.create())
            ring_np.setLightOff()
            make_text(billboard, "BURN %d" % (i + 1), (1.0, 0.75, 0.30, 1.0), 0.5,
                      pos=(0, 0, -2.0))
            self.burn_markers.append((marker, scale))
            self.burn_vectors.append(DynLine(self.root, (1.0, 0.68, 0.18, 1.0), 2.5,
                                             segments=True, capacity=6, name="burn_vector_%d" % (i + 1)))
        # faint full-orbit trails (static, inertial coordinates), shown in the Earth view so you can see where the satellites are going
        self.ghost_c = DynLine(self.root, (1.0, 0.40, 0.22, 0.22), 1.2, capacity=cap, name="ghost_c")
        self.ghost_d = DynLine(self.root, (0.30, 0.85, 1.0, 0.22), 1.2, capacity=cap, name="ghost_d")
        self.ghost_c.set_points(sim.tr_C)
        self.ghost_d.set_points(sim.tr_D)
 
        # LVLH guides (R-bar / V-bar / H-bar) through the chief
        self.guides = self.root.attachNewNode("guides")
        self.guides_scale = self.guides.attachNewNode("gs")
        gls = LineSegs()
        gls.setThickness(1.2)
        gls.setColor(0.75, 0.80, 0.9, 0.55)
        for k in range(3):
            a = np.zeros(3); a[k] = -1
            b = np.zeros(3); b[k] = 1
            gls.moveTo(*a); gls.drawTo(*b)
        gnp = self.guides_scale.attachNewNode(gls.create())
        gnp.setTransparency(TransparencyAttrib.MAlpha)
        gnp.setLightOff()
        for k, nm in enumerate(("R-bar", "V-bar", "H-bar")):
            pos = np.zeros(3); pos[k] = 1.06
            holder = self.guides_scale.attachNewNode("g")
            holder.setPos(*pos)
            holder.setBillboardPointEye()
            make_text(holder, nm + " +", (0.75, 0.80, 0.9, 0.9), 0.045)
 
        # camera state ------------------------------------------------------------------------
        self.mode = EARTH
        # per-mode camera state: yaw/pitch from mouse drag, zoom multiplier, manual distance
        self.state = {m: dict(yaw=0.0, pitch=0.0, mult=1.0, manual=None) for m in MODE_CFG}
        self.auto_fit = True
        self.dist = None  # None -> snap to the target distance on the next frame
        self.trans = None   # active camera transition: {'frm': mode, 's': 0..1}
        self.show_axes = True
        self.show_trails = True
        self.show_labels = True
        self.show_guides = True
        self.set_aspect(1.6)
 
    # ------------------------------------------------------------------------------------------
    def set_aspect(self, aspect):
        # keep the vertical FOV fixed and derive the horizontal one from the window aspect
        v = math.radians(VFOV_DEG)
        h = 2 * math.atan(aspect * math.tan(v / 2))
        self.lens.setAspectRatio(aspect)
        self.lens.setFov(math.degrees(h), VFOV_DEG)
 
    def set_mode(self, mode, animate=True):
        if mode == self.mode:
            return
        if animate and self.dist is not None:
            # glide from the mode we are leaving to the new one
            self.trans = dict(frm=self.mode, s=0.0)
        else:
            self.trans = None
            self.dist = None            # snap on the next frame
        self.mode = mode
 
    def contains(self, mx, my):
        # mouse coords are in [-1, 1], display region bounds in [0, 1]
        l, r, b, t = self.bounds
        u, v = (mx + 1) / 2, (my + 1) / 2
        return l <= u <= r and b <= v <= t
 
    def zoom(self, factor):
        # auto-fit: zoom scales the fitted distance; manual: it scales the absolute distance
        st = self.state[self.mode]
        if self.auto_fit or self.mode == EARTH:
            st['mult'] *= factor
        else:
            st['manual'] = (st['manual'] or self.dist or 1.0) * factor
 
    def orbit(self, dx, dy):
        st = self.state[self.mode]
        # mouse drag -> yaw/pitch; pitch is clamped so the camera can't flip over the pole
        st['yaw'] -= math.radians(dx * 110.0)
        st['pitch'] = float(np.clip(st['pitch'] + math.radians(dy * 90.0), -math.radians(89), math.radians(89)))
 
    def toggle_autofit(self):
        self.auto_fit = not self.auto_fit
        st = self.state[self.mode]
        st['manual'] = self.dist if not self.auto_fit else None
 
    def reset_orbit(self):
        for st in self.state.values():
            st.update(yaw=0.0, pitch=0.0, mult=1.0, manual=None)
        self.auto_fit = True
 
    # ------------------------------------------------------------------------------------------
    def _frame(self, mode, S):
        """(focus point, rotation inertial -> view axes, auto-fit distance) of a camera mode."""
        rc, rd, vd = S.rc, S.rd, S.vd
        rng = float(np.linalg.norm(rd - rc))
        if mode == EARTH:
            return np.zeros(3), np.eye(3), 3.7 * self.sim.ac
        if mode == DEPUTY_RADIAL:
            return rd, OM.lvlh_dcm(rd, vd)[0], max(3.0 * rng, 0.02)
        if mode == CHIEF_LVLH:
            return rc, S.NOc, max(2.8 * rng, 0.012)
        return rc, quat_to_mat(S.qc).T, max(2.8 * rng, 0.012)
 
    def _pose(self, mode, S):
        """Camera pose of a mode: focus, view frame Q, wanted distance, and the camera direction / up in INERTIAL axes."""
        cfg, st = MODE_CFG[mode], self.state[mode]
        focus, Q, fit = self._frame(mode, S)
        # wanted camera distance (follows the range in auto-fit, manual zoom otherwise), limited per mode
        if self.auto_fit or mode == EARTH or st['manual'] is None:
            target = st['mult'] * fit
        else:
            target = st['manual']
        target = float(np.clip(target, cfg['dmin'], cfg['dmax']))
        # orbit camera: yaw around the up vector, then pitch around the right vector
        base_dir = normalize(np.array(cfg['base_dir'], float))
        base_up = normalize(np.array(cfg['base_up'], float))
        right = normalize(np.cross(base_up, base_dir))
        Rcam = rot_axis(base_up, st['yaw']) @ rot_axis(right, st['pitch'])
        cdir, cup = Rcam @ base_dir, Rcam @ base_up
        return dict(focus=focus, Q=Q, dist=target, dir=Q.T @ cdir, up=Q.T @ cup)
 
    # ------------------------------------------------------------------------------------------
    def update(self, S, dt, now):
        sim = self.sim
        rc, rd, vd = S.rc, S.rd, S.vd
        rng = float(np.linalg.norm(rd - rc))
        Rn_c = quat_to_mat(S.qc)
        Rn_d = quat_to_mat(S.qd)
 
        mode = self.mode
        pb = self._pose(mode, S)                      # pose of the (new) target mode
 
        # ---- camera transition bookkeeping ----------------------------------------------------
        trans_active = False
        if self.trans is not None:
            self.trans['s'] += dt / TRANSITION_SECONDS
            if self.trans['s'] >= 1.0:
                self.trans = None
                self.dist = pb['dist']
            else:
                trans_active = True
 
        if trans_active:
            pa = self._pose(self.trans['frm'], S)     # pose of the mode we are leaving
            e = smoothstep(self.trans['s'])
            # distance interpolated in log space (works from thousands of km down to metres)
            d = math.exp((1 - e) * math.log(pa['dist']) + e * math.log(pb['dist']))
            # focus is anchored to the closer-in pose and slides to the farther one as the distance grows,
            # so the satellites stay in frame during the whole move
            near, far_ = (pa, pb) if pa['dist'] < pb['dist'] else (pb, pa)
            w = (d - near['dist']) / max(far_['dist'] - near['dist'], 1e-12)
            focus = near['focus'] + (far_['focus'] - near['focus']) * w
            Q = slerp_mat(pa['Q'], pb['Q'], e)
            dir_i = slerp_vec(pa['dir'], pb['dir'], e)
            up_i = normalize((1 - e) * pa['up'] + e * pb['up'])
            self.dist = d
        else:
            focus, Q, dir_i, up_i = pb['focus'], pb['Q'], pb['dir'], pb['up']
            if self.dist is None:
                self.dist = pb['dist']
            else:
                # ease towards the target in log space: feels the same from thousands of km down to metres, and doesn't depend on fps
                k = 1.0 - math.exp(-4.0 * dt)
                self.dist = math.exp(math.log(self.dist) + (math.log(pb['dist']) - math.log(self.dist)) * k)
 
        # camera sits at dist along dir (inertial dir rotated into the view frame), looking at the view origin (the focus point)
        cam_pos = Q @ (dir_i * self.dist)
        cup = Q @ up_i
        self.cam_np.setPos(*cam_pos)
        self.cam_np.lookAt(Point3(0, 0, 0), Vec3(*cup))
 
        # ---- everything in view coordinates (float64 -> small numbers) ------------------------
        pc = Q @ (rc - focus)
        pd = Q @ (rd - focus)
        pe = Q @ (-focus)
        qQ = LQuaternionf(*mat_to_quat(Q))

        self.earth.setPos(*pe)
        earth_angle = (2.0 * math.pi * S.t / SIDEREAL_DAY_S) % (2.0 * math.pi)
        earth_rotation = rot_axis((0.0, 0.0, 1.0), earth_angle)
        self.earth.setQuat(LQuaternionf(*mat_to_quat(Q @ earth_rotation)))
        # clip planes follow the camera distance (avoids z-fighting at metre scale but still reaches the Earth)
        far = (np.linalg.norm(pe - cam_pos) + R_EARTH/2000) * 1.03 + 4.0 * self.dist
        near = max(self.dist * 0.002, 1e-3)
        self.lens.setNearFar(near, far)
        # stars move with the camera so they stay infinitely far away
        self.stars.setPos(*cam_pos)
        self.stars.setQuat(qQ)
        self.stars.setScale(0.5 * far)
        self.sun_np.lookAt(Point3(*(-(Q @ SUN_DIR_INERTIAL))))  # sun direction rotated into the view frame
 
        dcam_c = float(np.linalg.norm(pc - cam_pos))
        dcam_d = float(np.linalg.norm(pd - cam_pos))
        self.chief.update(pc, Q @ Rn_c, dcam_c, None, now)
        self.deputy.update(pd, Q @ Rn_d, dcam_d, fmt_len(rng), now)
        self.chief.axes.show() if self.show_axes else self.chief.axes.hide()
        self.deputy.axes.show() if self.show_axes else self.deputy.axes.hide()
        for sv in (self.chief, self.deputy):
            sv.marker.show() if self.show_labels else sv.marker.hide()
 
        # Burn locations stay on the deputy's orbit trail; their arrows show inertial delta-v in view coordinates.
        # During a transition they are shown if either the old or the new mode shows them.
        modes_now = {mode} | ({self.trans['frm']} if trans_active else set())
        show_burns = bool(modes_now & {EARTH, DEPUTY_RADIAL})
        for i, (marker, marker_scale) in enumerate(self.burn_markers):
            visible = show_burns and S.t >= sim.burn_t[i]
            marker.show() if visible else marker.hide()
            self.burn_vectors[i].show(visible)
            if not visible:
                continue
            burn_pos = Q @ (sim.burn_r[i] - focus)
            marker.setPos(*burn_pos)
            marker_scale.setScale(max(self.dist, 1e-7) * MARKER_SCREEN_FRAC * 1.5)
            direction = normalize(Q @ sim.burn_dv[i])
            arrow_len = max(self.dist * 0.025, 0.003)
            side = normalize(np.cross(direction, [0.0, 0.0, 1.0]))
            if np.linalg.norm(side) < 1e-8:
                side = normalize(np.cross(direction, [0.0, 1.0, 0.0]))
            tip = burn_pos + direction * arrow_len
            barb = arrow_len * 0.18
            spread = arrow_len * 0.08
            self.burn_vectors[i].set_points(np.array([
                burn_pos, tip,
                tip, tip - direction * barb + side * spread,
                tip, tip - direction * barb - side * spread,
            ]))
 
        # ---- LVLH guides (only once the transition has finished) ------------------------------
        if self.show_guides and not trans_active and mode in (CHIEF_LVLH, CHIEF_BODY):
            self.guides.show()
            self.guides.setPos(*pc)
            self.guides.setQuat(LQuaternionf(*mat_to_quat(Q @ S.NOc.T)))
            self.guides_scale.setScale(self.dist * 0.75)
        else:
            self.guides.hide()
 
        # ---- trails -------------------------------------------------------------------------------
        tr_on = self.show_trails
        # ghost orbits are stored in inertial coordinates, so transform them like the Earth
        ghost_on = tr_on and ((mode == EARTH and not trans_active) or (trans_active and self.dist > 50.0))
        for g in (self.ghost_c, self.ghost_d):
            g.show(ghost_on)
            g.np.setPos(*pe)
            g.np.setQuat(qQ)
 
        inertial_trails = trans_active or mode in (EARTH, DEPUTY_RADIAL)
        if tr_on and inertial_trails:
            # only draw the trail up to the current time
            k = int(np.searchsorted(sim.tr_T, S.t, 'right'))
            pts = np.vstack([(sim.tr_C[:k] - focus) @ Q.T, pc[None, :]])
            self.trail_c.set_points(pts)
            pts = np.vstack([(sim.tr_D[:k] - focus) @ Q.T, pd[None, :]])
            self.trail_d.set_points(pts)
            self.trail_c.show(True); self.trail_d.show(True)
        # chief-centred views: draw the deputy's relative path instead (LVLH or body components)
        elif tr_on and mode in (CHIEF_LVLH, CHIEF_BODY):
            k = int(np.searchsorted(sim.rel_T, S.t, 'right'))
            src = sim.rel_H if mode == CHIEF_LVLH else sim.rel_B
            if k >= 1:
                self.trail_d.set_points(np.vstack([src[:k], pd[None, :]]))
            else:
                self.trail_d.clear()
            self.trail_c.show(False); self.trail_d.show(True)
        else:
            self.trail_c.show(False); self.trail_d.show(False)
 
 
# ============================================================================================
#  6.  PLOT PANELS (drawn natively with Panda3D, rolling time window)
# ============================================================================================
# Small rolling-window plot drawn with Panda3D lines/text (no matplotlib), cheap enough to update every few frames.
class PlotPanel:
    ML, MR, MB, MT = 0.165, 0.03, 0.07, 0.085  # margins: left, right, bottom, top
 
    def __init__(self, parent, x, z, w, h, title, unit, names, colors, T, data, valid_idx, stage_marks):
        from direct.gui.OnscreenText import OnscreenText
        self.w, self.h = w, h
        self.T, self.data, self.valid = T, data, valid_idx
        self.stage_marks = stage_marks              # {label: time}
        self.root = parent.attachNewNode("plot_" + title)
        self.root.setPos(x, 0, z)
        cm = CardMaker("bg")
        cm.setFrame(0, w, 0, h)
        bg = self.root.attachNewNode(cm.generate())
        bg.setColor(0.02, 0.03, 0.07, 0.74)
        bg.setTransparency(TransparencyAttrib.MAlpha)
        ls = LineSegs()
        ls.setThickness(1.0)
        ls.setColor(0.55, 0.60, 0.70, 0.9)
        ls.moveTo(0, 0, 0); ls.drawTo(w, 0, 0); ls.drawTo(w, 0, h); ls.drawTo(0, 0, h); ls.drawTo(0, 0, 0)
        self.root.attachNewNode(ls.create())
 
        self.px0, self.px1 = self.ML, w - self.MR
        self.pz0, self.pz1 = self.MB, h - self.MT
        mk = lambda txt, pos, sc=0.030, al=TextNode.ALeft, fg=(0.85, 0.9, 1, 1): OnscreenText(
            text=txt, parent=self.root, pos=pos, scale=sc, fg=fg, align=al, mayChange=True)
        self.title = mk("%s  [%s]" % (title, unit), (0.012, h - 0.05), 0.034)
        self.lbl_hi = mk("", (self.px0 - 0.008, self.pz1 - 0.02), 0.026, TextNode.ARight)
        self.lbl_lo = mk("", (self.px0 - 0.008, self.pz0 - 0.005), 0.026, TextNode.ARight)
        self.lbl_t = mk("", (self.px1, 0.012), 0.024, TextNode.ARight, (0.7, 0.75, 0.85, 1))
        self.note = mk("", ((self.px0 + self.px1) / 2, (self.pz0 + self.pz1) / 2), 0.032, TextNode.ACenter,
                       (0.9, 0.8, 0.5, 1))
        x0 = w - 0.012
        self.legend = []
        for nm, c in reversed(list(zip(names, colors))):
            t = OnscreenText(text=nm, parent=self.root, pos=(x0, h - 0.05), scale=0.030, fg=c,
                             align=TextNode.ARight, mayChange=False)
            self.legend.append(t)
            x0 -= 0.034 + 0.021 * len(nm)
 
        self.lines = [DynLine(self.root, c, 1.6, capacity=480, name="pl%d" % i) for i, c in enumerate(colors)]
        self.zero = DynLine(self.root, (0.6, 0.65, 0.75, 0.55), 1.0, segments=True, capacity=2, name="zero")
        self.marks = DynLine(self.root, (0.9, 0.9, 0.6, 0.45), 1.0, segments=True, capacity=8, name="marks")
        self.mark_txt = {lab: mk(lab, (0, 0), 0.024, TextNode.ACenter, (0.9, 0.9, 0.6, 0.9)) for lab in stage_marks}
        for t in self.mark_txt.values():
            t.hide()
        self.ylo, self.yhi = -1.0, 1.0
        self._last_labels = ("", "", "")
 
    def show(self, f=True):
        self.root.show() if f else self.root.hide()
 
    def update(self, t_now, window):
        T = self.T
        i1 = int(np.searchsorted(T, t_now, 'right'))
        # nothing to plot until the controller has started
        if i1 - self.valid < 2:
            for ln in self.lines:
                ln.clear()
            self.zero.clear(); self.marks.clear()
            for t in self.mark_txt.values():
                t.hide()
            self.note.setText("waiting for the controller (Stage 1 starts at t = %s)" % fmt_time(T[self.valid]))
            return
        self.note.setText("")
        t0v = T[self.valid]
        t_left = max(t0v, t_now - window)
        t_right = t_left + window
        i0 = max(self.valid, int(np.searchsorted(T, t_left)))
        n = i1 - i0
        step = max(1, -(-n // 450))  # ceil division: at most ~450 points per line
        idx = np.arange(i0, i1, step)
        if idx[-1] != i1 - 1:
            idx = np.append(idx, i1 - 1)
        tt = T[idx]
        Y = self.data[idx]
        # y-range always includes zero
        lo, hi = min(0.0, float(Y.min())), max(0.0, float(Y.max()))
        if hi - lo < 1e-12:
            hi, lo = hi + 1e-6, lo - 1e-6
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        # expand immediately, shrink slowly
        self.yhi = hi if hi > self.yhi else self.yhi + (hi - self.yhi) * 0.2
        self.ylo = lo if lo < self.ylo else self.ylo + (lo - self.ylo) * 0.2
        ylo, yhi = self.ylo, self.yhi
        sx = (self.px1 - self.px0) / (t_right - t_left)
        sz = (self.pz1 - self.pz0) / (yhi - ylo)
        x = self.px0 + (tt - t_left) * sx
        for c, ln in enumerate(self.lines):
            ln.set_points(np.column_stack((x, np.zeros_like(x), self.pz0 + (Y[:, c] - ylo) * sz)))
        zz = self.pz0 + (0.0 - ylo) * sz
        self.zero.set_points(np.array([[self.px0, 0, zz], [self.px1, 0, zz]]))
        seg = []
        for lab, tm in self.stage_marks.items():
            if t_left < tm < t_now:
                xm = self.px0 + (tm - t_left) * sx
                seg += [[xm, 0, self.pz0], [xm, 0, self.pz1]]
                self.mark_txt[lab].setPos(xm, self.pz1 + 0.008)
                self.mark_txt[lab].show()
            else:
                self.mark_txt[lab].hide()
        self.marks.set_points(np.array(seg) if seg else np.zeros((0, 3)))
        labels = ("%.3g" % yhi, "%.3g" % ylo, "window %d s  (t = %s)" % (window, fmt_time(t_now)))
        if labels != self._last_labels:
            self.lbl_hi.setText(labels[0]); self.lbl_lo.setText(labels[1]); self.lbl_t.setText(labels[2])
            self._last_labels = labels
 
 
# ============================================================================================
#  7.  APPLICATION
# ============================================================================================
class Assets:
    pass
 
 
# The ShowBase subclass is built inside a function so configure_panda() has already set the
# window options before ShowBase gets imported.
def build_app_class():
    from direct.showbase.ShowBase import ShowBase
    from direct.gui.OnscreenText import OnscreenText
 
    class RPOApp(ShowBase):
        INSET = (0.672, 0.995, 0.615, 0.985)          # left, right, bottom, top (window fractions)
 
        def __init__(self, sim, start_t=0.0, speed_idx=DEFAULT_SPEED_IDX, paused=False):
            ShowBase.__init__(self)
            self.sim = sim
            self.disableMouse()  # disable the default mouse camera, we orbit by hand
            self.setBackgroundColor(0, 0, 0)
            self.camNode.setActive(False)  # default camera not used, the two WorldViews do the rendering

            props = WindowProperties()      # Disable the mouse cursor (it is used for orbiting the camera)
            props.setCursorHidden(True)
            self.win.requestProperties(props)
 
            self.t = float(start_t)
            self.playing = not paused
            self.speed_idx = speed_idx
            self.plot_window = 3600.0
            self.show_plots = True        # user wish (P key); plots are only really shown after stage 0
            self.plots_active = False     # plots currently occupy the inset area
            self.show_help = False
            self.drag_view = None
            self.last_mouse = None
            self.last_phase = None
            self._hud_t = -1.0
 
            # ---- shared assets ----------------------------------------------------------------
            A = Assets()
            A.chief, A.chief_ext = load_sat_model(self.loader, CHIEF_MODEL, (0.80, 0.66, 0.25))
            A.deputy, A.deputy_ext = load_sat_model(self.loader, DEPUTY_MODEL, (0.72, 0.74, 0.80))
            A.stars = make_stars()
            A.earth = make_sphere(R_EARTH)
            p = os.path.join(MODEL_DIR, "earth_texture.jpg")
        
            tex = self.loader.loadTexture(Filename.fromOsSpecific(p))
            tex.setMinfilter(Texture.FTLinearMipmapLinear)
            tex.setMagfilter(Texture.FTLinear)
            tex.setAnisotropicDegree(4)
            tex.setWrapU(Texture.WMRepeat)  # wrap around in longitude, clamp at the poles
            tex.setWrapV(Texture.WMClamp)
            A.earth.setTexture(tex)
            self.assets = A
 
            # ---- the two views --------------------------------------------------------------------
            self.main = WorldView(self, "main", (0.0, 1.0, 0.0, 1.0), 0, A, sim)
            self.inset = WorldView(self, "inset", self.INSET, 5, A, sim)
 
            # ---- 2-D overlay ---------------------------------------------------------------------------
            self.hud = OnscreenText(text="", parent=self.a2dTopLeft, pos=(0.04, -0.075), scale=0.043,
                                    fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True)
            self.banner = OnscreenText(text="", parent=self.a2dTopCenter, pos=(0, -0.085), scale=0.06,
                                       fg=(1.0, 0.85, 0.35, 1), shadow=(0, 0, 0, 1), align=TextNode.ACenter,
                                       mayChange=True)
            self.hint = OnscreenText(text="H : help", parent=self.a2dBottomCenter, pos=(0, 0.03), scale=0.036,
                                     fg=(0.7, 0.75, 0.85, 0.9), align=TextNode.ACenter, mayChange=False)
            self.inset_title = OnscreenText(text="", parent=self.aspect2d, pos=(0, 0), scale=0.036,
                                            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1), align=TextNode.ALeft,
                                            mayChange=True)
            self.main_title = OnscreenText(text="", parent=self.a2dBottomRight, pos=(-0.04, 0.05), scale=0.036,
                                           fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1), align=TextNode.ARight,
                                           mayChange=True)
            ls = LineSegs()
            ls.setThickness(2.0)
            ls.setColor(0.8, 0.85, 1.0, 0.9)
            l, r, b, t = [2 * v - 1 for v in self.INSET]
            ls.moveTo(l, 0, b); ls.drawTo(r, 0, b); ls.drawTo(r, 0, t); ls.drawTo(l, 0, t); ls.drawTo(l, 0, b)
            self.frame = self.render2d.attachNewNode(ls.create())
 
            # opaque backdrop for the inset area: when the plots replace the inset camera they sit on this
            # card, so the main view doesn't show through the translucent plot backgrounds
            cm = CardMaker("inset_bg")
            cm.setFrame(l, r, b, t)
            self.inset_bg = self.render2d.attachNewNode(cm.generate())
            self.inset_bg.setColor(0.0, 0.0, 0.012, 1.0)
            self.inset_bg.setBin("background", 0)
            self.inset_bg.hide()
 
            # help text is cut out of the module docstring (Controls section)
            self.help_text = OnscreenText(text=__doc__.split("Controls\n--------\n")[1].split("Camera modes")[0],
                                          parent=self.aspect2d, pos=(0, 0.35), scale=0.040, fg=(1, 1, 1, 1),
                                          shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True)
            self.help_text.hide()
 
            # ---- plots -------------------------------------------------------------------------------------
            # vertical lines in the plots where each stage starts
            marks = {}
            for p, lab in ((2, "S2"), (3, "S3"), (4, "S4")):
                if p in sim.phase_start:
                    marks[lab] = sim.phase_start[p]
            # both panels live in a holder that on_resize() scales / centres inside the inset area
            pw, ph, gap = 1.15, 0.345, 0.02
            self.plot_box = (pw, 2 * ph + gap)
            self.plot_holder = self.aspect2d.attachNewNode("plots")
            self.plot_holder.hide()
            self.plot_err = PlotPanel(self.plot_holder, 0.0, 0.0, pw, ph,
                                      "Position error vs reference", "m",
                                      ["ex", "ey", "ez", "|e|"],
                                      [(1, .3, .3, 1), (.3, 1, .3, 1), (.4, .6, 1, 1), (1, 1, 1, 1)],
                                      sim.T, np.column_stack((sim.err_m, sim.err_norm_m)), sim.cwh_idx, marks)
            self.plot_u = PlotPanel(self.plot_holder, 0.0, ph + gap, pw, ph,
                                    "Controller input u", "mm/s^2",
                                    ["ux", "uy", "uz", "|u|"],
                                    [(1, .3, .3, 1), (.3, 1, .3, 1), (.4, .6, 1, 1), (1, 1, 1, 1)],
                                    sim.T, np.column_stack((sim.u_mm, sim.u_norm_mm)), sim.cwh_idx, marks)
 
            self.on_resize()
            self.accept("aspectRatioChanged", self.on_resize)
            self._bind_keys()
            self.taskMgr.add(self.update_task, "rpo_update")  # runs every frame
 
        # ------------------------------------------------------------------------------------------------
        def on_resize(self):
            aspect = self.getAspectRatio()
            # keep the camera aspect ratios in sync with the window (the inset only covers part of it)
            W, H = max(1, self.win.getXSize()), max(1, self.win.getYSize())
            l, r, b, t = self.INSET
            self.main.set_aspect(W / float(H))
            self.inset.set_aspect((r - l) * W / float((t - b) * H))
            self.inset_title.setPos((2 * l - 1) * aspect + 0.012, (2 * t - 1) - 0.05)
            self.help_text.setPos(0, 0.45)
 
            # fit the plot stack inside the inset area (aspect2d units), keeping its proportions
            box_w, box_h = (r - l) * 2 * aspect, (t - b) * 2
            sc = 0.96 * min(box_w / self.plot_box[0], box_h / self.plot_box[1])
            cx, cz = (l + r - 1) * aspect, (b + t - 1)
            self.plot_holder.setScale(sc)
            self.plot_holder.setPos(cx - sc * self.plot_box[0] / 2, 0, cz - sc * self.plot_box[1] / 2)
 
        # ------------------------------------------------------------------------------------------------
        def _bind_keys(self):
            a = self.accept
            a("escape", sys.exit)
            a("space", self.toggle_play)
            for k in ("arrow_up",):
                a(k, self.change_speed, [+1])
            for k in ("arrow_down",):
                a(k, self.change_speed, [-1])
            for k in ("arrow_right", "arrow_right-repeat"):
                a(k, self.seek, [60.0])
            for k in ("arrow_left", "arrow_left-repeat"):
                a(k, self.seek, [-60.0])
            for k in ("shift-arrow_right", "shift-arrow_right-repeat"):
                a(k, self.seek, [600.0])
            for k in ("shift-arrow_left", "shift-arrow_left-repeat"):
                a(k, self.seek, [-600.0])
            a("home", self.set_time, [0.0])
            a("n", self.jump_phase, [+1])
            a("b", self.jump_phase, [-1])
            a("c", self.cycle_camera, ["main"])
            a("v", self.cycle_camera, ["inset"])
            a("x", self.swap_cameras)
            a("r", self.reset_views)
            a("p", self.toggle_plots)
            a("[", self.change_window, [0.5])
            a("]", self.change_window, [2.0])
            a("a", self.toggle, ["show_axes"])
            a("t", self.toggle, ["show_trails"])
            a("l", self.toggle, ["show_labels"])
            a("g", self.toggle, ["show_guides"])
            a("f", self.toggle_autofit)
            a("h", self.toggle_help)
            a("f2", self.toggle_fps)
            # several key names per action to cover different keyboard layouts
            for k in ("wheel_up", "=", "+", "shift-=", "shift-+", "num_add"):
                a(k, self.zoom_cursor, [1 / 1.12])
            for k in ("wheel_down", "-", "shift--", "num_subtract", "_"):
                a(k, self.zoom_cursor, [1.12])
            a("mouse1", self.mouse_down)
            a("mouse1-up", self.mouse_up)
 
        # ---- actions -----------------------------------------------------------------------------------
        def toggle_play(self):
            # at the end of the run, play restarts from the beginning
            if self.t >= self.sim.T[-1] - 1e-6:
                self.t = 0.0
            self.playing = not self.playing
 
        def change_speed(self, d):
            self.speed_idx = int(np.clip(self.speed_idx + d, 0, len(SPEEDS) - 1))
 
        def set_time(self, t):
            self.t = float(np.clip(t, self.sim.T[0], self.sim.T[-1]))
 
        def seek(self, dt):
            self.set_time(self.t + dt)
 
        def jump_phase(self, d):
            # jump to just after the next/previous phase start (the 5 s margin stops 'previous' from sticking to the current one)
            starts = sorted(self.sim.phase_start.values())
            if d > 0:
                nxt = [s for s in starts if s > self.t + 1.0]
                self.set_time(nxt[0] + 1.0 if nxt else starts[-1] + 1.0)
            else:
                prv = [s for s in starts if s < self.t - 5.0]
                self.set_time(prv[-1] + 1.0 if prv else 0.0)
 
        # the view under the mouse is the one that gets zoom / drag
        # (None while the cursor is over the plots, there is no 3D view there)
        def view_at_cursor(self):
            if self.mouseWatcherNode.hasMouse():
                m = self.mouseWatcherNode.getMouse()
                if self.inset.contains(m.getX(), m.getY()):
                    return None if self.plots_active else self.inset
            return self.main
 
        def zoom_cursor(self, factor):
            v = self.view_at_cursor()
            if v is not None:
                v.zoom(factor)
 
        def toggle_autofit(self):
            v = self.view_at_cursor()
            if v is not None:
                v.toggle_autofit()
 
        def cycle_camera(self, which):
            if which == "inset":
                self.show_plots = False       # asking for an inset camera -> bring the 3D view back
            v = self.main if which == "main" else self.inset
            i = MODE_CYCLE.index(v.mode)
            v.set_mode(MODE_CYCLE[(i + 1) % len(MODE_CYCLE)])
 
        def swap_cameras(self):
            self.show_plots = False           # swapping only makes sense with a 3D view in the inset
            m, i = self.main.mode, self.inset.mode
            self.main.set_mode(i)
            self.inset.set_mode(m)
 
        def reset_views(self):
            for v in (self.main, self.inset):
                v.reset_orbit()
            self.show_plots = True            # R = back to the automatic layout (plots in the inset after stage 0)
            self.apply_layout(self.cur_phase)
 
        def apply_layout(self, phase):
            m, i = AUTO_LAYOUT[phase]
            self.main.set_mode(m)                    # animated glide
            self.inset.set_mode(i, animate=False)    # hidden behind the plots anyway
 
        def toggle_plots(self):
            self.show_plots = not self.show_plots
 
        def set_plots_active(self, flag):
            """Plots take over the inset area (and the inset camera is switched off), or give it back."""
            if flag == self.plots_active:
                return
            self.plots_active = flag
            if flag:
                self.plot_holder.show()
                self.inset_bg.show()
                self.inset_title.hide()
            else:
                self.plot_holder.hide()
                self.inset_bg.hide()
                self.inset_title.show()
            self.inset.dr.setActive(not flag)     # don't render a view nobody can see
 
        def change_window(self, f):
            self.plot_window = float(np.clip(self.plot_window * f, 150.0, 40000.0))  # between 150 s and ~11 h
 
        def toggle(self, attr):
            for v in (self.main, self.inset):
                setattr(v, attr, not getattr(v, attr))
 
        def toggle_help(self):
            self.show_help = not self.show_help
            self.help_text.show() if self.show_help else self.help_text.hide()
 
        def toggle_fps(self):
            self.setFrameRateMeter(not self.frameRateMeter)
 
        # ---- mouse -------------------------------------------------------------------------------------
        def mouse_down(self):
            if self.mouseWatcherNode.hasMouse():
                m = self.mouseWatcherNode.getMouse()
                self.drag_view = self.view_at_cursor()
                self.last_mouse = (m.getX(), m.getY())
 
        def mouse_up(self):
            self.drag_view = None
 
        # ---- main loop ---------------------------------------------------------------------------------
        cur_phase = 0
 
        def update_task(self, task):
            # cap dt so a frame hiccup doesn't make the simulation jump
            dt = min(globalClock.getDt(), 0.1)
            if self.playing:
                # simulation time advances `speed` times faster than real time
                self.t += dt * SPEEDS[self.speed_idx]
                if self.t >= self.sim.T[-1]:
                    self.t = float(self.sim.T[-1])
                    self.playing = False
 
            # orbit the view while the left button is held
            if self.drag_view is not None and self.mouseWatcherNode.hasMouse():
                m = self.mouseWatcherNode.getMouse()
                x, y = m.getX(), m.getY()
                self.drag_view.orbit(x - self.last_mouse[0], y - self.last_mouse[1])
                self.last_mouse = (x, y)
 
            # interpolated state at the current time
            S = self.sim.sample(self.t)
            self.cur_phase = S.phase
            # only touch the cameras when the automatic layout changes, otherwise manual choices would be overwritten every frame
            if self.last_phase is None or AUTO_LAYOUT[S.phase] != AUTO_LAYOUT[self.last_phase]:
                self.apply_layout(S.phase)
            self.last_phase = S.phase
 
            # plots only exist once the controller runs (phase >= 1); in phase 0 the inset keeps its camera view
            self.set_plots_active(self.show_plots and S.phase >= 1)
 
            now = time.time()
            self.main.update(S, dt, now)
            if not self.plots_active:
                self.inset.update(S, dt, now)
            self.update_hud(S, now)
            if self.plots_active and now - getattr(self, "_plot_t", 0.0) > 0.05:  # plots refresh at ~20 Hz
                self._plot_t = now
                self.plot_err.update(S.t, self.plot_window)
                self.plot_u.update(S.t, self.plot_window)
            return task.cont
 
        def update_hud(self, S, now):
            if now - self._hud_t < 0.066:  # HUD text refreshes at ~15 Hz
                return
            self._hud_t = now
            sim = self.sim
            rng = float(np.linalg.norm(S.rd - S.rc))
            chief_orbits = sim.nc * S.t / (2.0 * math.pi)
            i = S.i
            if S.phase == 0:
                ctrl = "Controller:  off (free flight)"
            else:
                ctrl = "|Pos Error| %8.3f m     |u| %8.4f mm/s^2" % (sim.err_norm_m[i], sim.u_norm_mm[i])
            burn_idx = int(np.searchsorted(sim.burn_t, S.t, side="right") - 1)
            burn_status = ""
            self.banner.setText(SimData.PHASE_NAMES[S.phase])
            self.hud.setText(
                "Chief orbits: %.2f\nt = %s   (%.0f s)\n%s   Speed x%d\nRange Chief-Deputy: %s\n%s%s" % (
                    chief_orbits, fmt_time(S.t), S.t,
                    "Playing" if self.playing else "Paused", SPEEDS[self.speed_idx],
                    fmt_len(rng), ctrl, burn_status))
            self.main_title.setText("MAIN: " + MODE_NAMES[self.main.mode] + ("" if self.main.auto_fit else "  [manual zoom]"))
            self.inset_title.setText(MODE_NAMES[self.inset.mode] + ("" if self.inset.auto_fit else "  [manual zoom]"))
 
    return RPOApp
 
 
# ============================================================================================
#  MAIN
# ============================================================================================
def configure_panda(width=1600, height=900):
    loadPrcFileData("", """
window-title Chief / Deputy RPO - 3D Simulation
win-size %d %d
sync-video 1
framebuffer-multisample 1
multisamples 4
audio-library-name null
""" % (width, height))
 
 
def main():
    ap = argparse.ArgumentParser(description="3D Panda3D visualisation of the chief/deputy RPO simulation")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--speed", type=int, default=SPEEDS[DEFAULT_SPEED_IDX], help="initial speed multiplier")
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="print relative-velocity continuity at burn 2 and at the CWH hand-over")
    args = ap.parse_args()
 
    w, h = [int(v) for v in args.size.lower().split("x")]
    # run (or load) the simulation before opening any window
    sim = SimData(load_or_run(use_cache=not args.no_cache))
    print("[sim] phases start at:", {k: round(v) for k, v in sim.phase_start.items()})
 
    if args.check:
        check_continuity(sim, int(np.searchsorted(sim.T, sim.tof)), "burn 2 (expect a ~54 m/s step along-track)")
        check_continuity(sim, sim.cwh_idx, "CWH hand-over (expect NO step)")
 
    configure_panda(w, h)
    App = build_app_class()
    # snap --speed to the nearest allowed value
    idx = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - args.speed))
    app = App(sim, start_t=args.start, speed_idx=idx, paused=args.paused)
    app.run()
 
 
if __name__ == "__main__":
    main()