"""
Python port of the 2-layer QG channel model (originally MATLAB, in ../qg2c/*.m).

This module is the *single* place that implements the PDE. Other scripts
(run_one_case.py, run_sweep_driver.py) should call into here rather than
duplicate the time-stepping or parameter setup.

Conventions match the MATLAB code exactly:
  - q is laid out as (ny, nx), row = y (south-to-north), col = x.
  - x is periodic, y is bounded (no-flux walls at y=0 and y=W).
  - q lives at cell centers y = (i+0.5)*dy, x = (j+0.5)*dx.
  - psi (p1, p2) lives at corners y = i*dy, i=0..ny  -- shape (ny+1, nx+1).
  - u = -d psi/dy at v-edges, v = d psi/dx at u-edges (staggered like in invert.m).

The MATLAB code mixes orthonormal DCT (type-II) and (non-orthonormal) DST-I.
We use scipy.fft with norm='ortho' for both, which is sufficient because each
transform is paired with its inverse around a diagonal spectral operation.

Public API:
  setup_params(amp=, r=, delta=, nx=, ny=, ...)
      Build the parameter dictionary P. Does NOT set the initial q (caller
      sets q1_init/q2_init when calling run()).

  run(P=None, q1_init=None, q2_init=None, tmax=, dt=,
      diag_hook=None, dt_diag=None, field_hook=None, dt_field=None,
      verbose=False)
      Run the AB-extrapolation + van-Leer flux-limited upwind scheme.
      Hooks are called with a state dict
        {'t', 'q1', 'q2', 'p1', 'p2', 'u1', 'v1', 'u2', 'v2'}
      at the requested cadences. q1, q2 in the hook are the q AT TIME t
      (not the AB-extrapolated half-step).

  default_init(P) -> (q1, q2)
      Returns the original small sin*sin barotropic seed (validation).
"""

import numpy as np
from scipy.fft import fft, ifft, dct, idct, dst, idst


# ---------- small helpers (translations of p.m, rs.m) ----------

def p(f, dy, axis=0):
    """diff/dy along given axis. Matches MATLAB p.m which is diff along dim 1."""
    return np.diff(f, axis=axis) / dy


def rs(q):
    """Rescale to [0,1] for plotting."""
    qmax = q.max(); qmin = q.min()
    d = qmax - qmin
    return q if d == 0 else (q - qmin) / d


# ---------- flux-limited advection (flux.m) ----------

def flux(f, v, dy, dt, bcf):
    """
    Van-Leer flux-limited upwind advection along axis 0.
      f  : (ny, nx) cell-centered tracer
      v  : (ny+1, nx) face-normal velocity (one extra row vs f along axis 0)
      bcf: 0 = no-flux wall (antisymmetric extension), 1 = periodic
    Returns face flux of shape (ny+1, nx).
    """
    if bcf == 0:
        f = np.vstack([-f[0:1, :], f, -f[-1:, :]])
    else:
        f = np.vstack([f[-1:, :], f, f[0:1, :]])
    n = f.shape[0]
    nm1 = n - 1
    fbar = 0.5 * (f[:nm1, :] + f[1:n, :])
    delf = f[1:n, :] - f[:nm1, :]
    absv = np.abs(v)
    fup = v * fbar - 0.5 * absv * delf
    flw = v * fbar - 0.5 * (dt / dy) * absv ** 2 * delf

    delfp = np.roll(delf, 1, axis=0)
    delfm = np.roll(delf, -1, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        r = ((v >= 0) * delfp + (v < 0) * delfm) / delf
    r = np.where(np.isnan(r), 0.0, r)
    r = np.where(np.isinf(r), 1e20 * np.sign(r), r)

    psi = (r + np.abs(r)) / (1.0 + np.abs(r))      # van Leer limiter
    return fup + psi * (flw - fup)


# ---------- spectral inversion of q->psi (invert.m) ----------

def invert(q1, q2, a11, a12, a21, a22, b11, b12, b21, b22, massfac, Hfree, delv):
    """Inverts the 2-layer PV equations to get (p1, p2) of shape (ny+1, nx+1).
    Mirrors invert.m bit-for-bit."""
    ny, nx = q1.shape

    z1b = q1.mean(axis=1); z2b = q2.mean(axis=1)
    q1e = q1 - z1b[:, None]; q2e = q2 - z2b[:, None]

    qe1_left = np.concatenate([q1e[:, -1:], q1e[:, :-1]], axis=1)
    qe2_left = np.concatenate([q2e[:, -1:], q2e[:, :-1]], axis=1)
    z1 = 0.25 * (q1e[:ny - 1, :] + qe1_left[:ny - 1, :]
                 + q1e[1:ny, :] + qe1_left[1:ny, :])
    z2 = 0.25 * (q2e[:ny - 1, :] + qe2_left[:ny - 1, :]
                 + q2e[1:ny, :] + qe2_left[1:ny, :])

    Z1b = dct(z1b, type=2, norm='ortho')
    Z2b = dct(z2b, type=2, norm='ortho')
    P1b = idct(b11 * Z1b + b12 * Z2b, type=2, norm='ortho')
    P2b = idct(b21 * Z1b + b22 * Z2b, type=2, norm='ortho')

    def to_corners(pb):
        out = np.empty(ny + 1)
        out[0]      = 0.5 * pb[0]   + 0.5 * pb[1]
        out[1:ny]   = 0.5 * (pb[:ny - 1] + pb[1:ny])
        out[ny]     = 0.5 * pb[ny - 2] + 0.5 * pb[ny - 1]
        return out
    P1b = to_corners(P1b); P2b = to_corners(P2b)

    delmass = massfac @ (P1b - P2b)
    P1b = P1b - delmass * Hfree
    P2b = P2b + delv * delmass * Hfree

    Z1 = fft(dst(z1, type=1, norm='ortho', axis=0), axis=1)
    Z2 = fft(dst(z2, type=1, norm='ortho', axis=0), axis=1)
    P1 = a11 * Z1 + a12 * Z2
    P2 = a21 * Z1 + a22 * Z2
    p1_int = np.real(idst(ifft(P1, axis=1), type=1, norm='ortho', axis=0))
    p2_int = np.real(idst(ifft(P2, axis=1), type=1, norm='ortho', axis=0))

    p1 = np.vstack([np.zeros((1, nx)), p1_int, np.zeros((1, nx))])
    p2 = np.vstack([np.zeros((1, nx)), p2_int, np.zeros((1, nx))])
    p1 = np.concatenate([p1, p1[:, 0:1]], axis=1) + P1b[:, None]
    p2 = np.concatenate([p2, p2[:, 0:1]], axis=1) + P2b[:, None]
    return p1, p2


# ---------- parameterized setup ----------

def setup_params(amp=4e4, r=0.1, delta=0.2, nx=128, ny=64,
                 f0=8.64, beta=1.728e-3, W=500.0, Rd=40.0,
                 r1=None, r2=None):
    """Build the parameter dictionary P. Caller supplies the initial q.

    Returns a dict containing grid, parameters, inversion matrices,
    forcing field qforce(y), and the free-mode (Hfree, massfac) for mass
    adjustment. The default values match the original MATLAB qg2c_par.m.

    r1 / r2 override the per-layer damping (and the associated relaxation
    forcing -r*F*qforce). If None they fall back to the scalar r.
    """
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
    """Original small sin*sin barotropic seed used for validation."""
    x, y = P['x'], P['y']
    q1 = 0.01 * np.sin(2 * np.pi * x / P['L']) * np.sin(np.pi * y / P['W'])
    return q1, q1.copy()


# ---------- time stepping ----------

def run(P=None, q1_init=None, q2_init=None,
        tmax=50.0, dt=1.0 / 128.0,
        diag_hook=None, dt_diag=None,
        field_hook=None, dt_field=None,
        verbose=False, progress_every=None):
    """Run the AB-extrapolation + van-Leer flux-limited upwind scheme.

    Hooks are called with a state dict whose q-values are at time t:
        state = {'t', 'q1', 'q2', 'p1', 'p2', 'u1', 'v1', 'u2', 'v2'}
    Velocities and psi correspond to the AB-extrapolated half-step
    (offset by dt/2 from q); this matches the original MATLAB log.

    Returns dict with axes, final fields (q1, q2, p1..v2), and 'wall'.
    """
    if P is None:
        P = setup_params()
    if q1_init is None or q2_init is None:
        q1_init, q2_init = default_init(P)

    nx, ny = P['nx'], P['ny']
    L, W = P['L'], P['W']
    # r  = thermal-relaxation rate (drives -r*F*qforce, symmetric on both layers)
    # r1 = friction (Ekman drag) on layer 1
    # r2 = friction (Ekman drag) on layer 2
    # In the original symmetric model r1 = r2 = r so the two are entangled;
    # splitting them lets us turn off layer-1 friction while keeping thermal forcing.
    r, r1, r2 = P['r'], P['r1'], P['r2']
    F1, F2 = P['F1'], P['F2']
    by = P['by']
    qforce = P['qforce']
    dx = L / nx
    dy = W / ny

    q1 = q1_init.copy(); q2 = q2_init.copy()
    q1_p = q1.copy();    q2_p = q2.copy()

    diag_every  = max(1, int(round(dt_diag  / dt))) if dt_diag  else None
    field_every = max(1, int(round(dt_field / dt))) if dt_field else None
    nsteps = int(np.round((tmax + dt / 2) / dt)) + 1

    import time as _time
    t0 = _time.time()
    p1 = p2 = u1 = v1 = u2 = v2 = None

    progress_every_steps = (max(1, int(round(progress_every / dt)))
                            if progress_every else None)

    for tc in range(nsteps + 1):
        t = tc * dt
        if t > tmax + dt / 2:
            break

        # AB-like extrapolation: q1 <- 1.5 q1 - 0.5 q1_p; q1_p <- old q1.
        tmp = q1.copy(); q1 = 1.5 * q1 - 0.5 * q1_p; q1_p = tmp
        tmp = q2.copy(); q2 = 1.5 * q2 - 0.5 * q2_p; q2_p = tmp

        p1, p2 = invert(q1, q2,
                        P['a11'], P['a12'], P['a21'], P['a22'],
                        P['b11'], P['b12'], P['b21'], P['b22'],
                        P['massfac'], P['Hfree'], P['del'])
        u1 = -p(p1, dy, axis=0); v1 =  p(p1, dx, axis=1)
        u2 = -p(p2, dy, axis=0); v2 =  p(p2, dx, axis=1)

        if (diag_every  is not None) and (tc % diag_every  == 0) and (diag_hook is not None):
            diag_hook(dict(t=t, q1=q1_p, q2=q2_p, p1=p1, p2=p2,
                           u1=u1, v1=v1, u2=u2, v2=v2), P)
        if (field_every is not None) and (tc % field_every == 0) and (field_hook is not None):
            field_hook(dict(t=t, q1=q1_p, q2=q2_p, p1=p1, p2=p2,
                            u1=u1, v1=v1, u2=u2, v2=v2), P)

        if verbose and (tc % max(1, int(round(1.0 / dt))) == 0):
            qp = q1_p - q1_p.mean(axis=1, keepdims=True)
            print(f"t={t:6.2f}  max|u1bar|={np.max(np.abs(u1.mean(axis=1))):8.4f}  "
                  f"max(qp1)={qp.max():10.3e}", flush=True)
        if progress_every_steps and (tc % progress_every_steps == 0) and tc > 0:
            elapsed = _time.time() - t0
            frac = t / tmax if tmax > 0 else 0
            eta = elapsed * (1 / frac - 1) if frac > 0 else float('nan')
            qp = q1_p - q1_p.mean(axis=1, keepdims=True)
            print(f"[progress] t={t:7.2f}/{tmax:.0f}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min  "
                  f"max|q1p|={np.max(np.abs(qp)):.3e}", flush=True)

        qt1 = q1 + by
        qt2 = q2 + by

        Fy = flux(qt1, v1, dy, dt, 0); Fy[0, :] = 0; Fy[-1, :] = 0
        Fx = flux(qt1.T, u1.T, dx, dt, 1).T
        dq1dt = -p(Fx, dx, axis=1) - p(Fy, dy, axis=0) - r1 * q1 - r * F1 * qforce

        Fy = flux(qt2, v2, dy, dt, 0); Fy[0, :] = 0; Fy[-1, :] = 0
        Fx = flux(qt2.T, u2.T, dx, dt, 1).T
        dq2dt = -p(Fx, dx, axis=1) - p(Fy, dy, axis=0) - r2 * q2 + r * F2 * qforce

        q1 = q1_p + dt * dq1dt
        q2 = q2_p + dt * dq2dt

    return dict(P=P,
                q1_final=q1_p, q2_final=q2_p,
                p1=p1, p2=p2, u1=u1, v1=v1, u2=u2, v2=v2,
                wall=_time.time() - t0)


if __name__ == '__main__':
    # Validation: run the default IC and save final fields + zonal-mean u1.
    import time
    ts, ub_max, qp_max = [], [], []

    def diag(s, P):
        u1bar = s['u1'].mean(axis=1)
        ts.append(s['t']); ub_max.append(float(np.max(np.abs(u1bar))))
        qp = s['q1'] - s['q1'].mean(axis=1, keepdims=True)
        qp_max.append(float(qp.max()))

    t0 = time.time()
    out = run(tmax=50.0, dt=1.0 / 128.0, diag_hook=diag, dt_diag=1.0, verbose=True)
    print(f"Elapsed: {time.time()-t0:.1f}s")

    np.savez('legacy_validation/qg2c_python_out.npz',
             ts=np.array(ts), ub_max=np.array(ub_max), qp_max=np.array(qp_max),
             q1=out['q1_final'], q2=out['q2_final'])
