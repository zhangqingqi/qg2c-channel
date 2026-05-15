"""
Diagnostics for the 2-layer QG channel model.

All functions follow the conventions in qg2c.py:
  - q is (ny, nx), cell-centered.
  - psi is (ny+1, nx+1), corner-located.
  - u = -d psi/dy has shape (ny, nx+1) on vertical edges of corner grid.
  - v =  d psi/dx has shape (ny+1, nx) on horizontal edges of corner grid.

Layer weights from layer-height ratio delta:
    h1 = delta / (1 + delta)
    h2 = 1     / (1 + delta)
"""

import numpy as np


# ---------- staggered -> cell-center ----------

def cell_center_velocity(u, v):
    """u: (ny, nx+1), v: (ny+1, nx). Returns (u_cc, v_cc) of shape (ny, nx)."""
    u_cc = u[:, :-1]
    v_cc = 0.5 * (v[:-1, :] + v[1:, :])
    return u_cc, v_cc


def psi_to_cell_center(psi):
    """psi: (ny+1, nx+1). Returns (ny, nx) via 4-corner average."""
    return 0.25 * (psi[:-1, :-1] + psi[1:, :-1] + psi[:-1, 1:] + psi[1:, 1:])


# ---------- perturbations ----------

def perturb_x(f):
    """Remove zonal (x) mean. f shape (ny, nx). Returns f'."""
    return f - f.mean(axis=1, keepdims=True)


# ---------- bulk diagnostics ----------

def compute_EKE(u1, v1, u2, v2, delta):
    """Bulk eddy KE with layer weights from delta."""
    u1c, v1c = cell_center_velocity(u1, v1)
    u2c, v2c = cell_center_velocity(u2, v2)
    u1p = perturb_x(u1c); v1p = perturb_x(v1c)
    u2p = perturb_x(u2c); v2p = perturb_x(v2c)
    h1 = delta / (1.0 + delta)
    h2 = 1.0   / (1.0 + delta)
    return 0.5 * (h1 * np.mean(u1p**2 + v1p**2) + h2 * np.mean(u2p**2 + v2p**2))


def compute_Zq(q1, q2, delta):
    """Bulk eddy PV enstrophy. Returns (Zq, Zq1, Zq2)."""
    q1p = perturb_x(q1)
    q2p = perturb_x(q2)
    h1 = delta / (1.0 + delta)
    h2 = 1.0   / (1.0 + delta)
    Zq1 = 0.5 * np.mean(q1p**2)
    Zq2 = 0.5 * np.mean(q2p**2)
    Zq = h1 * Zq1 + h2 * Zq2
    return Zq, Zq1, Zq2


# ---------- spectral diagnostics ----------

def zonal_power_spectrum(qp):
    """rFFT along x, then average |.|^2 over y. qp shape (ny, nx).
    Returns power(kx_nonneg)."""
    qhat = np.fft.rfft(qp, axis=1)
    return (np.abs(qhat) ** 2).mean(axis=0)


def spectral_signature(power):
    """Return (f_dom, S_k_normalized) for a 1-D power spectrum."""
    p = np.asarray(power, dtype=float)
    total = p.sum()
    if total <= 0:
        return np.nan, np.nan
    p = p / total
    p_nz = p[p > 0]
    f_dom = float(p.max())
    S = -float(np.sum(p_nz * np.log(p_nz))) / np.log(len(p))
    return f_dom, S


# ---------- y-resolved statistics ----------

def variance_x(fp):
    """<f'^2>_x along x. fp shape (ny, nx). Returns (ny,)."""
    return np.mean(fp ** 2, axis=1)


def skewness_x(fp):
    """Skewness of f' along x at each y."""
    v = np.mean(fp ** 2, axis=1)
    m3 = np.mean(fp ** 3, axis=1)
    return m3 / np.maximum(v, 1e-30) ** 1.5


def kurtosis_x(fp):
    """Kurtosis of f' along x at each y (Gaussian -> 3)."""
    v = np.mean(fp ** 2, axis=1)
    m4 = np.mean(fp ** 4, axis=1)
    return m4 / np.maximum(v, 1e-30) ** 2


def pooled_skewness_y(fp_seq):
    """Skewness at each y, pooled over (t, x). fp_seq shape (nt, ny, nx)."""
    # Pool x and t at each y: reshape (nt, ny, nx) -> (ny, nt*nx)
    arr = np.transpose(fp_seq, (1, 0, 2)).reshape(fp_seq.shape[1], -1)
    arr = arr - arr.mean(axis=1, keepdims=True)
    v = np.mean(arr ** 2, axis=1)
    m3 = np.mean(arr ** 3, axis=1)
    return m3 / np.maximum(v, 1e-30) ** 1.5


def pooled_kurtosis_y(fp_seq):
    """Kurtosis at each y, pooled over (t, x). fp_seq shape (nt, ny, nx)."""
    arr = np.transpose(fp_seq, (1, 0, 2)).reshape(fp_seq.shape[1], -1)
    arr = arr - arr.mean(axis=1, keepdims=True)
    v = np.mean(arr ** 2, axis=1)
    m4 = np.mean(arr ** 4, axis=1)
    return m4 / np.maximum(v, 1e-30) ** 2


def pooled_moments_box(fp_seq, y_lo, y_hi):
    """Scalar (mean, std, skew, kurt) over the interior box.

    fp_seq shape (nt, ny, nx); y_lo:y_hi selects y-slice (cell indices).
    Pools over (t, y_lo:y_hi, x). Returns dict.
    """
    sub = fp_seq[:, y_lo:y_hi, :].ravel()
    sub = sub[np.isfinite(sub)]
    n = sub.size
    if n < 4:
        return dict(n=n, mean=np.nan, std=np.nan, skew=np.nan, kurt=np.nan)
    mu = float(sub.mean())
    chi = sub - mu
    var = float(np.mean(chi ** 2))
    if var <= 0:
        return dict(n=n, mean=mu, std=0.0, skew=np.nan, kurt=np.nan)
    sd = var ** 0.5
    sk = float(np.mean(chi ** 3) / sd ** 3)
    ku = float(np.mean(chi ** 4) / sd ** 4)
    return dict(n=int(n), mean=mu, std=float(sd), skew=sk, kurt=ku)


# ---------- growth-rate fitter ----------

def fit_growth_rate(t, y, tmin=None, tmax=None, auto=True):
    """
    Fit y(t) ~ exp(2 sigma t) -> log(y) = a + 2 sigma t.

    If auto=True (and tmin/tmax not given) selects window where
        10 * percentile_5(y) < y < 0.1 * percentile_95(y)
    which targets the linear-growth regime.

    Returns dict with sigma, intercept, r2, mask, tfit, yfit.
    """
    t = np.asarray(t); y = np.asarray(y)
    mask = np.isfinite(y) & (y > 0)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax
    if auto and tmin is None and tmax is None:
        if mask.sum() > 5:
            ypos = y[mask]
            lo = np.percentile(ypos, 5)
            hi = np.percentile(ypos, 95)
            mask &= (y > 10 * lo) & (y < 0.1 * hi)

    if mask.sum() < 5:
        return dict(sigma=np.nan, intercept=np.nan, r2=np.nan,
                    mask=mask, tfit=np.array([]), yfit=np.array([]))

    tt = t[mask]; yy = y[mask]
    logy = np.log(yy)
    slope, intercept = np.polyfit(tt, logy, 1)
    sigma = 0.5 * slope
    pred = intercept + slope * tt
    resid = logy - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(sigma=float(sigma), intercept=float(intercept), r2=float(r2),
                mask=mask, tfit=tt, yfit=np.exp(pred))


# ---------- baroclinic noise IC ----------

def make_baroclinic_noise(x, y, L, W, seed, kmax=4, nmax=4, rms=1e-3):
    """Wall-compatible (sin n*pi*y/W) low-wavenumber random eddy with zero zonal mean.

    Inputs x, y are 2-D meshgrids of shape (ny, nx) like P['x'], P['y'].
    Output shape (ny, nx), normalized to RMS = rms.
    """
    rng = np.random.default_rng(seed)
    eta = np.zeros_like(x)
    for k in range(1, kmax + 1):
        for n in range(1, nmax + 1):
            a = rng.normal()
            phi = rng.uniform(0.0, 2.0 * np.pi)
            eta += a * np.sin(n * np.pi * y / W) * np.cos(2.0 * np.pi * k * x / L + phi)
    eta -= eta.mean(axis=1, keepdims=True)        # enforce zero zonal mean
    rms_now = np.sqrt(np.mean(eta ** 2))
    if rms_now > 0:
        eta *= rms / rms_now
    return eta


# ---------- regime label ----------

def regime_label(sigma_EKE, f_dom_late, threshold=0.5):
    if not np.isfinite(sigma_EKE) or sigma_EKE <= 0:
        return "stable_or_damped"
    if np.isfinite(f_dom_late) and f_dom_late > threshold:
        return "coherent_baroclinic_wave"
    return "broadband_or_turbulent"
