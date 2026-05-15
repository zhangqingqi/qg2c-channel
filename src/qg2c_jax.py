"""
JAX/GPU port of the 2-layer QG channel model.

Mirrors qg2c.py (the scipy/numpy reference). Same equations, same time
stepping, same staggering. Only the implementations are JAX-traced so the
inner step can be jit-compiled and dispatched to GPU.

scipy.fft DCT-II and DST-I are replaced with jnp.fft.fft tricks below,
validated against scipy in qg2c_jax_validate.py.

Public API mirrors qg2c.py:
  setup_params(...)      -> dict P (numpy arrays); host-side build.
  default_init(P)        -> (q1, q2) numpy.
  run(...)               -> dict with axes, final fields, wall.
                            Accepts diag_hook / field_hook called from Python
                            outside the jitted step.
"""

from __future__ import annotations

import os
# Limit thread pollution from numpy-on-the-side
os.environ.setdefault("XLA_FLAGS",
                      "--xla_gpu_deterministic_ops=false")

import time as _time
import numpy as np

import jax
# Match scipy double precision; cheap on A100. Must be set before any
# jax computation runs.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ===========================================================================
# DCT-II and DST-I via direct matrix multiplies, norm='ortho'.
#
# At our sizes (ny <= 256) these are tiny matmuls (<=256x256). XLA fuses
# them with surrounding ops on GPU. Avoiding the FFT-trick gymnastics
# also sidesteps the kind of off-by-phase bugs that plague hand-rolled
# DCT/DST implementations.
# ===========================================================================

def _dct_ii_matrix(N, dtype=np.float64):
    """M[k,n] such that y = M @ x is the orthonormal DCT-II of x."""
    n = np.arange(N); k = np.arange(N)
    M = np.cos(np.pi * (2 * n[None, :] + 1) * k[:, None] / (2 * N))
    f = np.where(k == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))
    return (f[:, None] * M).astype(dtype)


def _dst_i_matrix(N, dtype=np.float64):
    """M[k,n] such that y = M @ x is the orthonormal DST-I of x."""
    n = np.arange(N); k = np.arange(N)
    M = np.sin(np.pi * (n[None, :] + 1) * (k[:, None] + 1) / (N + 1))
    return (np.sqrt(2.0 / (N + 1)) * M).astype(dtype)


# Standalone helpers (used by the validation script). They build the matrix
# on-demand from numpy, so each call re-materializes — fine for one-off
# correctness checks but slow if you call inside a hot loop. The hot loop
# uses pre-built matrices passed through make_step's closure (below).

def dct_ii_ortho(x: jnp.ndarray) -> jnp.ndarray:
    """DCT-II norm='ortho' along the last axis."""
    M = jnp.asarray(_dct_ii_matrix(x.shape[-1]))
    return x @ M.T


def idct_ii_ortho(X: jnp.ndarray) -> jnp.ndarray:
    """Inverse DCT-II norm='ortho' along the last axis."""
    M = jnp.asarray(_dct_ii_matrix(X.shape[-1]))
    return X @ M


def dst_i_ortho_axis0(x: jnp.ndarray) -> jnp.ndarray:
    """DST-I norm='ortho' along axis 0."""
    M = jnp.asarray(_dst_i_matrix(x.shape[0]))
    return M @ x


def idst_i_ortho_axis0(X: jnp.ndarray) -> jnp.ndarray:
    return dst_i_ortho_axis0(X)


# ===========================================================================
# Parameter setup (host-side, returns dict of numpy arrays + JAX-ready)
# ===========================================================================

def setup_params(amp=4e4, r=0.1, delta=0.2, nx=128, ny=64,
                 f0=8.64, beta=1.728e-3, W=500.0, Rd=40.0,
                 r1=None, r2=None):
    if r1 is None: r1 = r
    if r2 is None: r2 = r
    P = dict(f0=f0, beta=beta, W=W, L=2.0 * W, r=r, r1=r1, r2=r2, Rd=Rd)
    P['del'] = delta
    P['nx'] = nx; P['ny'] = ny
    L = P['L']

    k0x = 2.0 * np.pi / L
    k0y = np.pi / W
    kx_vec = np.concatenate([np.arange(0, nx // 2 + 1),
                             np.arange(-nx // 2 + 1, 0)]) * k0x
    ly_vec = np.arange(1, ny) * k0y
    k, l = np.meshgrid(kx_vec, ly_vec)
    l0 = np.arange(0, ny) * k0y

    F1 = 1.0 / Rd ** 2 / (1.0 + delta)
    F2 = delta * F1
    P['F1'], P['F2'] = F1, F2

    wv2 = k * k + l * l
    det = wv2 * (wv2 + F1 + F2)
    a11 = -(wv2 + F2) / det
    a12 = -F1 / det
    a21 = -F2 / det
    a22 = -(wv2 + F1) / det
    a11[:, 0] = 0; a12[:, 0] = 0; a21[:, 0] = 0; a22[:, 0] = 0
    P.update(a11=a11, a12=a12, a21=a21, a22=a22)

    wv20 = l0 * l0
    det0 = wv20 * (wv20 + F1 + F2)
    det0[0] = 1.0
    b11 = -(wv20 + F2) / det0
    b12 = -F1 / det0
    b21 = -F2 / det0
    b22 = -(wv20 + F1) / det0
    b11[0] = 0; b12[0] = 0; b21[0] = 0; b22[0] = 0
    P.update(b11=b11, b12=b12, b21=b21, b22=b22)

    x_c = (np.arange(nx) + 0.5) / nx * L
    y_c = (np.arange(ny) + 0.5) / ny * W
    x, y = np.meshgrid(x_c, y_c)
    P['x'], P['y'] = x, y
    P['by'] = beta * (y - W / 2.0)

    P['amp'] = amp
    P['qforce'] = amp * np.cos(np.pi * y / W)

    y0 = np.arange(0, ny + 1) * W / ny
    Hfree = np.cosh((y0 - W / 2.0) / Rd)
    massfac = np.concatenate([[0.5], np.ones(ny - 1), [0.5]])
    Hfree = Hfree / (massfac @ Hfree) / (1.0 + delta)
    P['Hfree'] = Hfree
    P['massfac'] = massfac

    return P


def default_init(P):
    x, y = P['x'], P['y']
    q1 = 0.01 * np.sin(2 * np.pi * x / P['L']) * np.sin(np.pi * y / P['W'])
    return q1, q1.copy()


# ===========================================================================
# Spectral inversion (JAX)
# ===========================================================================

def invert_jax(q1, q2,
               a11, a12, a21, a22, b11, b12, b21, b22,
               massfac, Hfree, delv,
               M_dct=None, M_dst=None):
    """JAX port of qg2c.invert.

    M_dct: (ny, ny) DCT-II ortho matrix; M_dst: (ny-1, ny-1) DST-I ortho matrix.
    When None, they're built on the fly (slow path used by standalone tests).
    """
    ny, nx = q1.shape
    if M_dct is None: M_dct = jnp.asarray(_dct_ii_matrix(ny))
    if M_dst is None: M_dst = jnp.asarray(_dst_i_matrix(ny - 1))

    z1b = q1.mean(axis=1); z2b = q2.mean(axis=1)
    q1e = q1 - z1b[:, None]; q2e = q2 - z2b[:, None]

    qe1_left = jnp.concatenate([q1e[:, -1:], q1e[:, :-1]], axis=1)
    qe2_left = jnp.concatenate([q2e[:, -1:], q2e[:, :-1]], axis=1)
    z1 = 0.25 * (q1e[:ny - 1, :] + qe1_left[:ny - 1, :]
                 + q1e[1:ny, :] + qe1_left[1:ny, :])
    z2 = 0.25 * (q2e[:ny - 1, :] + qe2_left[:ny - 1, :]
                 + q2e[1:ny, :] + qe2_left[1:ny, :])

    # Barotropic-mean: DCT in y via matmul.
    Z1b = z1b @ M_dct.T
    Z2b = z2b @ M_dct.T
    P1b = (b11 * Z1b + b12 * Z2b) @ M_dct
    P2b = (b21 * Z1b + b22 * Z2b) @ M_dct

    # to corners (length ny+1)
    P1b = jnp.concatenate([
        ((0.5 * P1b[0] + 0.5 * P1b[1])[None]),
        0.5 * (P1b[:ny - 1] + P1b[1:ny]),
        ((0.5 * P1b[ny - 2] + 0.5 * P1b[ny - 1])[None]),
    ], axis=0)
    P2b = jnp.concatenate([
        ((0.5 * P2b[0] + 0.5 * P2b[1])[None]),
        0.5 * (P2b[:ny - 1] + P2b[1:ny]),
        ((0.5 * P2b[ny - 2] + 0.5 * P2b[ny - 1])[None]),
    ], axis=0)

    delmass = jnp.dot(massfac, P1b - P2b)
    P1b = P1b - delmass * Hfree
    P2b = P2b + delv * delmass * Hfree

    # Eddy: DST in y (matmul along axis 0), FFT in x.
    Z1 = jnp.fft.fft(M_dst @ z1, axis=1)
    Z2 = jnp.fft.fft(M_dst @ z2, axis=1)
    P1 = a11 * Z1 + a12 * Z2
    P2 = a21 * Z1 + a22 * Z2
    p1_int = jnp.real(M_dst @ jnp.fft.ifft(P1, axis=1))
    p2_int = jnp.real(M_dst @ jnp.fft.ifft(P2, axis=1))

    p1 = jnp.concatenate([jnp.zeros((1, nx), dtype=p1_int.dtype), p1_int,
                          jnp.zeros((1, nx), dtype=p1_int.dtype)], axis=0)
    p2 = jnp.concatenate([jnp.zeros((1, nx), dtype=p2_int.dtype), p2_int,
                          jnp.zeros((1, nx), dtype=p2_int.dtype)], axis=0)
    p1 = jnp.concatenate([p1, p1[:, 0:1]], axis=1) + P1b[:, None]
    p2 = jnp.concatenate([p2, p2[:, 0:1]], axis=1) + P2b[:, None]
    return p1, p2


# ===========================================================================
# Flux-limited van-Leer advection (JAX)
# ===========================================================================

def flux_jax(f, v, dy, dt, bcf):
    """JAX port of qg2c.flux. Returns (ny+1, nx) face flux along axis 0.

    bcf is a static int (0 = wall, 1 = periodic) — DO NOT pass as a Tracer.
    """
    if bcf == 0:
        f_ext = jnp.concatenate([-f[0:1, :], f, -f[-1:, :]], axis=0)
    else:
        f_ext = jnp.concatenate([f[-1:, :], f, f[0:1, :]], axis=0)
    n = f_ext.shape[0]
    fbar = 0.5 * (f_ext[:n - 1, :] + f_ext[1:n, :])
    delf = f_ext[1:n, :] - f_ext[:n - 1, :]
    absv = jnp.abs(v)
    fup = v * fbar - 0.5 * absv * delf
    flw = v * fbar - 0.5 * (dt / dy) * absv ** 2 * delf

    delfp = jnp.roll(delf, 1, axis=0)
    delfm = jnp.roll(delf, -1, axis=0)

    # r = ((v>=0)*delfp + (v<0)*delfm) / delf, safely
    num = jnp.where(v >= 0, delfp, delfm)
    safe_denom = jnp.where(delf == 0, 1.0, delf)
    r = jnp.where(delf == 0, 0.0, num / safe_denom)
    psi = (r + jnp.abs(r)) / (1.0 + jnp.abs(r))
    return fup + psi * (flw - fup)


# ===========================================================================
# One time step (jit-compiled)
# ===========================================================================

def _p_axis(f, dy, axis):
    """Forward difference along axis (jax equivalent of qg2c.p)."""
    if axis == 0:
        return (f[1:, :] - f[:-1, :]) / dy
    else:
        return (f[:, 1:] - f[:, :-1]) / dy


def make_step(P, dt):
    """Build a jitted step function bound to P's static arrays + dt.

    The step takes (q1, q2, q1_p, q2_p) -> (q1_new, q2_new, q1_p_new, q2_p_new,
                                            p1, p2, u1, v1, u2, v2).
    p1..v2 returned for diag/field hooks (cheap since already on GPU).
    """
    # Move static arrays to device once
    a11 = jnp.asarray(P['a11']); a12 = jnp.asarray(P['a12'])
    a21 = jnp.asarray(P['a21']); a22 = jnp.asarray(P['a22'])
    b11 = jnp.asarray(P['b11']); b12 = jnp.asarray(P['b12'])
    b21 = jnp.asarray(P['b21']); b22 = jnp.asarray(P['b22'])
    massfac = jnp.asarray(P['massfac']); Hfree = jnp.asarray(P['Hfree'])
    by = jnp.asarray(P['by']); qforce = jnp.asarray(P['qforce'])

    nx, ny = P['nx'], P['ny']
    M_dct = jnp.asarray(_dct_ii_matrix(ny))
    M_dst = jnp.asarray(_dst_i_matrix(ny - 1))
    L, W = P['L'], P['W']
    dx = L / nx; dy = W / ny
    delv = P['del']
    # r  = thermal-relaxation rate (drives the qforce term, symmetric)
    # r1 = Ekman friction on layer 1; r2 = friction on layer 2.
    r = float(P['r'])
    r1 = float(P['r1']); r2 = float(P['r2'])
    F1 = float(P['F1']); F2 = float(P['F2'])

    @jax.jit
    def step(q1, q2, q1_p, q2_p):
        # AB-like extrapolation
        q1_ab = 1.5 * q1 - 0.5 * q1_p
        q2_ab = 1.5 * q2 - 0.5 * q2_p
        q1_p_new = q1
        q2_p_new = q2

        p1, p2 = invert_jax(q1_ab, q2_ab,
                            a11, a12, a21, a22, b11, b12, b21, b22,
                            massfac, Hfree, delv,
                            M_dct=M_dct, M_dst=M_dst)
        u1 = -_p_axis(p1, dy, axis=0); v1 = _p_axis(p1, dx, axis=1)
        u2 = -_p_axis(p2, dy, axis=0); v2 = _p_axis(p2, dx, axis=1)

        qt1 = q1_ab + by
        qt2 = q2_ab + by

        Fy1 = flux_jax(qt1, v1, dy, dt, 0)
        Fy1 = Fy1.at[0, :].set(0.0).at[-1, :].set(0.0)
        Fx1 = flux_jax(qt1.T, u1.T, dx, dt, 1).T
        dq1dt = (-_p_axis(Fx1, dx, axis=1) - _p_axis(Fy1, dy, axis=0)
                 - r1 * q1_ab - r * F1 * qforce)

        Fy2 = flux_jax(qt2, v2, dy, dt, 0)
        Fy2 = Fy2.at[0, :].set(0.0).at[-1, :].set(0.0)
        Fx2 = flux_jax(qt2.T, u2.T, dx, dt, 1).T
        dq2dt = (-_p_axis(Fx2, dx, axis=1) - _p_axis(Fy2, dy, axis=0)
                 - r2 * q2_ab + r * F2 * qforce)

        q1_new = q1_p_new + dt * dq1dt
        q2_new = q2_p_new + dt * dq2dt
        return q1_new, q2_new, q1_p_new, q2_p_new, p1, p2, u1, v1, u2, v2

    return step


# ===========================================================================
# Run loop (Python loop, jitted step)
# ===========================================================================

def run(P=None, q1_init=None, q2_init=None,
        tmax=50.0, dt=1.0 / 128.0,
        diag_hook=None, dt_diag=None,
        field_hook=None, dt_field=None,
        verbose=False, progress_every=None):
    if P is None:
        P = setup_params()
    if q1_init is None or q2_init is None:
        q1_init, q2_init = default_init(P)

    step = make_step(P, dt)

    q1 = jnp.asarray(q1_init)
    q2 = jnp.asarray(q2_init)
    q1_p = q1; q2_p = q2

    diag_every  = max(1, int(round(dt_diag  / dt))) if dt_diag  else None
    field_every = max(1, int(round(dt_field / dt))) if dt_field else None
    progress_every_steps = (max(1, int(round(progress_every / dt)))
                            if progress_every else None)
    nsteps = int(np.round((tmax + dt / 2) / dt)) + 1

    p1 = p2 = u1 = v1 = u2 = v2 = None

    t0 = _time.time()
    for tc in range(nsteps + 1):
        t = tc * dt
        if t > tmax + dt / 2:
            break

        q1, q2, q1_p, q2_p, p1, p2, u1, v1, u2, v2 = step(q1, q2, q1_p, q2_p)

        # Hooks at requested cadences. Materialize only what's needed.
        if (diag_every  is not None) and (tc % diag_every  == 0) and (diag_hook is not None):
            state = dict(t=t,
                         q1=np.asarray(q1_p), q2=np.asarray(q2_p),
                         p1=np.asarray(p1), p2=np.asarray(p2),
                         u1=np.asarray(u1), v1=np.asarray(v1),
                         u2=np.asarray(u2), v2=np.asarray(v2))
            diag_hook(state, P)
        if (field_every is not None) and (tc % field_every == 0) and (field_hook is not None):
            state = dict(t=t,
                         q1=np.asarray(q1_p), q2=np.asarray(q2_p),
                         p1=np.asarray(p1), p2=np.asarray(p2),
                         u1=np.asarray(u1), v1=np.asarray(v1),
                         u2=np.asarray(u2), v2=np.asarray(v2))
            field_hook(state, P)

        if progress_every_steps and (tc % progress_every_steps == 0) and tc > 0:
            elapsed = _time.time() - t0
            frac = t / tmax if tmax > 0 else 0
            eta = elapsed * (1 / frac - 1) if frac > 0 else float('nan')
            qp_max = float(jnp.max(jnp.abs(q1_p - q1_p.mean(axis=1, keepdims=True))))
            print(f"[jax progress] t={t:7.2f}/{tmax:.0f}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min  "
                  f"max|q1p|={qp_max:.3e}", flush=True)

    # Block for any pending GPU work, then return host-side numpy
    q1.block_until_ready(); q2.block_until_ready()
    return dict(P=P,
                q1_final=np.asarray(q1_p), q2_final=np.asarray(q2_p),
                p1=np.asarray(p1), p2=np.asarray(p2),
                u1=np.asarray(u1), v1=np.asarray(v1),
                u2=np.asarray(u2), v2=np.asarray(v2),
                wall=_time.time() - t0)
