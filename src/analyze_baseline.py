"""
Detailed B+C+D analysis for the baseline run (loads its npz).

Generates (under outputs/figures/baseline/):
  01_EKE_Zq.png          - EKE(t), Zq(t) log-y, with fitted exp(2*sigma*t).
  02_growth_summary.txt  - sigma_EKE, sigma_Zq, R^2, sat levels.
  03_q_psi_panels.png    - 4-stage panel snapshots of q1', q2' and psi1, psi2.
  04_kx_hovmoller.png    - P(kx, t) time-wavenumber diagram.
  05_kx_spectra.png      - P(kx) at four stages.
  06_fdom_Sk.png         - f_dom(t), spectral_entropy(t).
  07_Zq_y_hov.png        - Zq(y,t) Hovmoller + late-time mean.
  08_skew_y_hov.png      - skewness(y,t) Hovmoller (layer 1 & 2).
  09_kurt_y_hov.png      - kurtosis(y,t) Hovmoller (layer 1 & 2).
  10_pdfs_qq.png         - standardized-PDF + Q-Q at selected y, late stage.

Movies (under outputs/movies/baseline/):
  q_prime.mp4   - q1' and q2' side by side
  psi.mp4       - psi1 and psi2 side by side
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from diagnostics import (
    fit_growth_rate, zonal_power_spectrum,
    pooled_skewness_y, pooled_kurtosis_y,
)


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)   # repo root, parent of src/
FIG_DIR = ""   # set by main()
MOV_DIR = ""


def load_run(amp, r1, r2, delta, seed, run_dir):
    if r1 == r2:
        tag = f"run_amp{amp:.4g}_r{r1:.3g}_d{delta:.3g}_s{seed}"
    else:
        tag = (f"run_amp{amp:.4g}_r1_{r1:.3g}_r2_{r2:.3g}"
               f"_d{delta:.3g}_s{seed}")
    path = os.path.join(run_dir, tag + ".npz")
    return path, np.load(path, allow_pickle=False), tag


def fmt_growth(sigma, r2, label):
    if not np.isfinite(sigma):
        return f"{label}: fit failed"
    return f"{label}: sigma = {sigma:.4f}  (R^2 = {r2:.3f})"


def auto_stages(t, EKE):
    """Return dict of (tlo, thi) for initial / growth / saturation / late."""
    fit = fit_growth_rate(t, EKE, auto=True)
    growth_mask = fit['mask']
    if growth_mask.any():
        t_g_lo = float(t[growth_mask].min())
        t_g_hi = float(t[growth_mask].max())
    else:
        t_g_lo = float(t[0]); t_g_hi = float(t[-1] * 0.3)

    t_last = float(t[-1])
    late_lo = max(t_g_hi, 0.65 * t_last)
    late_hi = t_last
    sat_lo = t_g_hi
    sat_hi = late_lo

    initial_lo = float(t[0])
    initial_hi = min(t_g_lo, 0.05 * t_last)
    return dict(
        initial=(initial_lo, initial_hi),
        growth=(t_g_lo, t_g_hi),
        saturation=(sat_lo, sat_hi),
        late=(late_lo, late_hi),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=4e4)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--r1", type=float, default=None)
    ap.add_argument("--r2", type=float, default=None)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir",
                    default=os.path.join(PROJECT_ROOT, "outputs", "runs"))
    ap.add_argument("--label", default=None,
                    help="figure/movie subdir; defaults to the run tag.")
    args = ap.parse_args()

    r1 = args.r if args.r1 is None else args.r1
    r2 = args.r if args.r2 is None else args.r2

    path, d, tag = load_run(args.amp, r1, r2, args.delta, args.seed, args.run_dir)
    label = args.label or tag
    fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures", label)
    mov_dir = os.path.join(PROJECT_ROOT, "outputs", "movies", label)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(mov_dir, exist_ok=True)
    global FIG_DIR, MOV_DIR
    FIG_DIR = fig_dir; MOV_DIR = mov_dir
    print(f"Loaded {path}")
    t   = d['t']
    EKE = d['EKE']; Zq = d['Zq']
    Zq1 = d['Zq1']; Zq2 = d['Zq2']
    fdom = d['f_dom']; Sk = d['spectral_entropy']

    Zq_y = d['Zq_y']
    skew1_y = d['skew1_y']; skew2_y = d['skew2_y']
    kurt1_y = d['kurt1_y']; kurt2_y = d['kurt2_y']
    y_axis = d['y']

    if 't_fields' not in d.files:
        print("This run has no saved fields; baseline analysis needs save_fields=True.")
        sys.exit(1)
    tf = d['t_fields']
    q1f = d['q1_fields']; q2f = d['q2_fields']
    psi1f = d['psi1_fields']; psi2f = d['psi2_fields']
    x_axis = d['x']

    h1 = float(d['h1']); h2 = float(d['h2'])
    amp = float(d['amp']); rcfg = float(d['r']); delta = float(d['delta'])
    # r1/r2 may not be present in legacy npz files.
    if 'r1' in d.files and 'r2' in d.files:
        r1n = float(d['r1']); r2n = float(d['r2'])
    else:
        r1n = r2n = rcfg
    r_label = (f"r={rcfg}" if r1n == r2n
               else f"$r_1$={r1n}, $r_2$={r2n}")

    stages = auto_stages(t, EKE)
    print("Stages:", stages)

    # -------- 01: EKE(t), Zq(t) with growth-rate fits --------
    fit_EKE = fit_growth_rate(t, EKE, auto=True)
    fit_Zq  = fit_growth_rate(t, Zq,  auto=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(t, EKE, label="EKE")
    if np.isfinite(fit_EKE['sigma']):
        axes[0].semilogy(fit_EKE['tfit'], fit_EKE['yfit'], 'r--',
                         label=f"fit: $\\sigma={fit_EKE['sigma']:.3f}$")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("EKE"); axes[0].legend()
    axes[0].set_title(f"EKE(t), amp={amp:g}, {r_label}, $\\delta$={delta}")
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].semilogy(t, Zq, label="$Z_q$")
    if np.isfinite(fit_Zq['sigma']):
        axes[1].semilogy(fit_Zq['tfit'], fit_Zq['yfit'], 'r--',
                         label=f"fit: $\\sigma={fit_Zq['sigma']:.3f}$")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("$Z_q$"); axes[1].legend()
    axes[1].set_title("Eddy PV enstrophy")
    axes[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_EKE_Zq.png"), dpi=120)
    plt.close()

    # -------- 02: growth-rate summary --------
    sat_lo, sat_hi = stages['late']
    sat_mask = (t >= sat_lo) & (t <= sat_hi)
    EKE_sat = float(np.mean(EKE[sat_mask])) if sat_mask.any() else np.nan
    Zq_sat  = float(np.mean(Zq[sat_mask]))  if sat_mask.any() else np.nan
    fdom_late = float(np.mean(fdom[sat_mask])) if sat_mask.any() else np.nan
    Sk_late = float(np.mean(Sk[sat_mask])) if sat_mask.any() else np.nan

    summary_lines = [
        f"Baseline: amp={amp}, r1={r1n}, r2={r2n}, delta={delta}, "
        f"seed={int(d['seed'])}, tmax={t[-1]}",
        fmt_growth(fit_EKE['sigma'], fit_EKE['r2'], "EKE"),
        fmt_growth(fit_Zq['sigma'],  fit_Zq['r2'],  "Zq"),
        f"Late window: t in [{sat_lo:.1f}, {sat_hi:.1f}]",
        f"EKE_sat       = {EKE_sat:.4e}",
        f"Zq_sat        = {Zq_sat:.4e}",
        f"f_dom_late    = {fdom_late:.3f}",
        f"S_k_late      = {Sk_late:.3f}",
    ]
    with open(os.path.join(FIG_DIR, "02_growth_summary.txt"), "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print("\n".join(summary_lines))

    # -------- 03: 4-stage panels of q' and psi --------
    stage_keys = ['initial', 'growth', 'saturation', 'late']
    stage_times = [0.5 * (stages[k][0] + stages[k][1]) for k in stage_keys]
    field_idx = [int(np.argmin(np.abs(tf - st))) for st in stage_times]

    # Color limits.
    q1p_all = np.array([q1f[i] - q1f[i].mean(axis=1, keepdims=True) for i in field_idx])
    q2p_all = np.array([q2f[i] - q2f[i].mean(axis=1, keepdims=True) for i in field_idx])
    qvmax = float(np.percentile(np.abs(np.concatenate([q1p_all.ravel(), q2p_all.ravel()])), 99))
    qvmax = qvmax if qvmax > 0 else 1.0
    psi_all = np.array([psi1f[i] for i in field_idx] + [psi2f[i] for i in field_idx])
    psivmax = float(np.percentile(np.abs(psi_all), 99))
    psivmax = psivmax if psivmax > 0 else 1.0

    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    for col, (key, st_t, fi) in enumerate(zip(stage_keys, stage_times, field_idx)):
        q1p = q1f[fi] - q1f[fi].mean(axis=1, keepdims=True)
        q2p = q2f[fi] - q2f[fi].mean(axis=1, keepdims=True)
        axes[0, col].imshow(q1p, origin='lower', aspect='auto', extent=extent,
                            vmin=-qvmax, vmax=qvmax, cmap='RdBu_r')
        axes[0, col].set_title(f"{key}\nt={st_t:.1f}, $q_1'$")
        axes[1, col].imshow(q2p, origin='lower', aspect='auto', extent=extent,
                            vmin=-qvmax, vmax=qvmax, cmap='RdBu_r')
        axes[1, col].set_title("$q_2'$")
        axes[2, col].imshow(psi1f[fi], origin='lower', aspect='auto', extent=extent,
                            vmin=-psivmax, vmax=psivmax, cmap='RdBu_r')
        axes[2, col].set_title("$\\psi_1$")
        axes[3, col].imshow(psi2f[fi], origin='lower', aspect='auto', extent=extent,
                            vmin=-psivmax, vmax=psivmax, cmap='RdBu_r')
        axes[3, col].set_title("$\\psi_2$")
        for row in range(4):
            axes[row, col].set_xlabel("x")
            if col == 0:
                axes[row, col].set_ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_q_psi_panels.png"), dpi=110)
    plt.close()

    # -------- 05: P(kx) at four stages, with -3 and -5/3 reference lines ----
    nframes = len(tf)
    P_kt = np.zeros((nframes, q1f.shape[2] // 2 + 1))
    for i in range(nframes):
        q1p = q1f[i] - q1f[i].mean(axis=1, keepdims=True)
        q2p = q2f[i] - q2f[i].mean(axis=1, keepdims=True)
        P_kt[i] = h1 * zonal_power_spectrum(q1p) + h2 * zonal_power_spectrum(q2p)
    fig, ax = plt.subplots(figsize=(7, 5))
    kx = np.arange(1, P_kt.shape[1])
    for key, fi in zip(stage_keys, field_idx):
        ax.loglog(kx, P_kt[fi, 1:] + 1e-30, label=key, lw=1.4)
    # Reference slopes anchored at low kx, extending across the full spectrum.
    if len(kx) > 4:
        sat_fi = field_idx[2]
        # anchor at kx ~ peak of the saturation spectrum so the reference
        # lines pass through the data and span the rest visibly.
        k0_idx = max(2, int(np.argmax(P_kt[sat_fi, 1:])) + 1)
        kref = kx[k0_idx - 1:]
        P_anchor = P_kt[sat_fi, k0_idx]
        ax.loglog(kref, P_anchor * (kref / kref[0]) ** (-3.0),
                  'k--', lw=1.2, label='$k^{-3}$ (enstrophy cascade)')
        ax.loglog(kref, P_anchor * (kref / kref[0]) ** (-5.0 / 3.0),
                  'k:',  lw=1.2, label='$k^{-5/3}$ (inverse cascade)')
    ax.set_xlabel("$k_x$ index"); ax.set_ylabel("$P(k_x)$")
    ax.set_title("Layer-averaged zonal spectra at four stages")
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_kx_spectra.png"), dpi=120)
    plt.close()

    # -------- 07: Zq(y,t) 200d running-mean Hovmoller + last-window mean -----
    win_len = 200.0
    # Build a 200d running-mean Zq_y_rm[i, y] centered on a trailing window
    # ending at t[i]: average Zq_y over t in (t[i] - win_len, t[i]].
    Zq_y_rm = np.empty_like(Zq_y)
    for i in range(len(t)):
        m = (t > t[i] - win_len) & (t <= t[i])
        if m.any():
            Zq_y_rm[i] = Zq_y[m].mean(axis=0)
        else:
            Zq_y_rm[i] = Zq_y[i]

    last_win_lo = max(t[0], t[-1] - win_len)
    last_win_mask = (t >= last_win_lo) & (t <= t[-1])
    Zq_y_win = (Zq_y[last_win_mask].mean(axis=0)
                if last_win_mask.any() else Zq_y[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={'width_ratios': [3, 1]})
    pm = axes[0].pcolormesh(t, y_axis, Zq_y_rm.T, cmap='viridis', shading='auto',
                            norm=matplotlib.colors.LogNorm(
                                vmin=max(np.percentile(Zq_y_rm[Zq_y_rm > 0], 5), 1e-20),
                                vmax=Zq_y_rm.max()))
    plt.colorbar(pm, ax=axes[0], label='$\\langle Z_q\\rangle_{200d}(y,t)$')
    axes[0].axvline(win_len, color='w', lw=0.8, ls='--')
    axes[0].set_xlabel("t (end of trailing 200d window)"); axes[0].set_ylabel("y")
    axes[0].set_title("Eddy PV enstrophy, 200d trailing-window mean")
    axes[1].plot(Zq_y_win, y_axis)
    axes[1].set_xlabel(f"$\\overline{{Z_q}}(y)$ over t$\\in$[{last_win_lo:.0f},{t[-1]:.0f}]")
    axes[1].set_ylabel("y"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_Zq_y_hov.png"), dpi=120)
    plt.close()

    # -------- 10: pooled PDF + Q-Q over the LAST 200d window --------
    # Pool (interior y, x, t) within the window into one big sample per layer.
    ny_full = q1f.shape[1]
    y_lo10 = int(round(0.10 * ny_full))
    y_hi10 = int(round(0.90 * ny_full))
    win_field_mask = (tf >= tf[-1] - win_len) & (tf <= tf[-1])
    if not win_field_mask.any():
        win_field_mask[-1] = True
    qw1 = q1f[win_field_mask] - q1f[win_field_mask].mean(axis=2, keepdims=True)
    qw2 = q2f[win_field_mask] - q2f[win_field_mask].mean(axis=2, keepdims=True)
    s1 = qw1[:, y_lo10:y_hi10, :].ravel()
    s2 = qw2[:, y_lo10:y_hi10, :].ravel()
    s1 = s1[np.isfinite(s1)]; s2 = s2[np.isfinite(s2)]

    def _std_skew_kurt(s):
        s = s - s.mean()
        sd = s.std()
        chi = s / sd if sd > 0 else s
        return chi, float(np.mean(chi ** 3)), float(np.mean(chi ** 4))

    chi1, sk1, ku1 = _std_skew_kurt(s1)
    chi2, sk2, ku2 = _std_skew_kurt(s2)
    win_lo_t = tf[win_field_mask][0]; win_hi_t = tf[win_field_mask][-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Wide explicit range so tails out to ±xmax sigma are visible. Bin width
    # ~0.05 sigma => 240 bins across [-6, 6].
    pdf_xmax = max(6.0, float(np.max(np.abs(np.concatenate([chi1, chi2])))) * 1.05)
    nbins_pdf = max(200, int(round(2 * pdf_xmax / 0.05)))
    bin_edges = np.linspace(-pdf_xmax, pdf_xmax, nbins_pdf + 1)
    xx = np.linspace(-pdf_xmax, pdf_xmax, 600)
    gauss = np.exp(-0.5 * xx ** 2) / np.sqrt(2 * np.pi)
    for row, (chi, sk, ku, lab) in enumerate([
            (chi1, sk1, ku1, "layer 1"),
            (chi2, sk2, ku2, "layer 2")]):
        ax = axes[row, 0]
        ax.hist(chi, bins=bin_edges, density=True, color='steelblue', alpha=0.75)
        ax.plot(xx, gauss, 'r--', lw=1.2, label='Gaussian')
        ax.set_yscale('log'); ax.set_ylim(1e-7, 1)
        ax.set_xlim(-pdf_xmax, pdf_xmax)
        ax.set_xlabel("$\\chi$"); ax.set_ylabel("PDF")
        ax.set_title(f"{lab}: N={chi.size}  skew={sk:.2f}  kurt={ku:.2f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax2 = axes[row, 1]
        # Subsample to keep the Q-Q lightweight when N is large.
        if chi.size > 20000:
            idx = np.linspace(0, chi.size - 1, 20000).astype(int)
            q_data = np.sort(chi)[idx]
        else:
            q_data = np.sort(chi)
        n = q_data.size
        q_theory = np.sqrt(2) * _erfinv(2 * (np.arange(1, n + 1) - 0.5) / n - 1)
        ax2.plot(q_theory, q_data, '.', ms=2)
        lim = max(abs(q_theory.min()), q_theory.max(),
                  abs(q_data.min()), q_data.max())
        ax2.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
        ax2.set_xlabel("Gaussian quantile"); ax2.set_ylabel("data quantile")
        ax2.set_title(f"Q-Q  {lab}")
        ax2.grid(alpha=0.3)
    fig.suptitle(f"Pooled stats over t$\\in$[{win_lo_t:.0f},{win_hi_t:.0f}],  "
                 f"y$\\in$[{y_axis[y_lo10]:.1f},{y_axis[y_hi10-1]:.1f}]")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_pdfs_qq.png"), dpi=110)
    plt.close()

    # -------- 11: sliding-window pooled skew/(kurt-3)(y) ----
    # 200-day window, 20-day step. Spin-up = 100 days. At each y, pool
    # samples over (t in window, x); the pooled (t,x)-mean is removed per y
    # inside pooled_skewness_y / pooled_kurtosis_y.
    spinup = 100.0
    win_len_sw = 200.0
    win_step = 20.0

    t_lo_arr, t_hi_arr, t_c_arr = [], [], []
    skew1_sw, skew2_sw, kurt1_sw, kurt2_sw = [], [], [], []
    t_lo = spinup
    while t_lo + win_len_sw <= tf[-1] + 1e-9:
        t_hi = t_lo + win_len_sw
        wm = (tf >= t_lo) & (tf < t_hi)
        if wm.sum() >= 5:
            q1w = q1f[wm]
            q2w = q2f[wm]
            skew1_sw.append(pooled_skewness_y(q1w))
            skew2_sw.append(pooled_skewness_y(q2w))
            kurt1_sw.append(pooled_kurtosis_y(q1w))
            kurt2_sw.append(pooled_kurtosis_y(q2w))
            t_lo_arr.append(t_lo); t_hi_arr.append(t_hi)
            t_c_arr.append(0.5 * (t_lo + t_hi))
        t_lo += win_step

    if len(t_c_arr) >= 1:
        skew1_sw = np.array(skew1_sw); skew2_sw = np.array(skew2_sw)
        kurt1_sw = np.array(kurt1_sw); kurt2_sw = np.array(kurt2_sw)
        t_c_arr = np.array(t_c_arr)

        # Excess kurtosis: K − 3 (Gaussian = 0).
        ek1_sw = kurt1_sw - 3.0
        ek2_sw = kurt2_sw - 3.0

        smax = float(np.nanpercentile(
            np.abs(np.concatenate([skew1_sw.ravel(), skew2_sw.ravel()])), 99))
        smax = smax if smax > 0 else 1.0
        kmax = float(np.nanpercentile(
            np.abs(np.concatenate([ek1_sw.ravel(), ek2_sw.ravel()])), 99))
        kmax = kmax if kmax > 0 else 1.0

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
        for ax, S, ttl in zip(axes[0], [skew1_sw, skew2_sw],
                              ['Skew, layer 1', 'Skew, layer 2']):
            pm = ax.pcolormesh(t_c_arr, y_axis, S.T, cmap='RdBu_r',
                               vmin=-smax, vmax=smax, shading='auto')
            plt.colorbar(pm, ax=ax)
            ax.set_title(f"{ttl}  (200d window, 20d step)")
            ax.set_xlabel("window center [days]"); ax.set_ylabel("y")
        for ax, K, ttl in zip(axes[1], [ek1_sw, ek2_sw],
                              ['Kurt$-3$, layer 1', 'Kurt$-3$, layer 2']):
            pm = ax.pcolormesh(t_c_arr, y_axis, K.T, cmap='RdBu_r',
                               vmin=-kmax, vmax=kmax, shading='auto')
            plt.colorbar(pm, ax=ax)
            ax.set_title(f"{ttl}  (Gaussian = 0)")
            ax.set_xlabel("window center [days]"); ax.set_ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "11_skew_kurt_sw.png"), dpi=120)
        plt.close()

    summary_lines += [
        "",
        f"Sliding window: {len(t_c_arr)} windows  win=200d, step=20d, "
        f"spin-up=100d",
        f"Pooled stats over last 200d window  "
        f"t in [{win_lo_t:.0f}, {win_hi_t:.0f}], "
        f"y in [{y_axis[y_lo10]:.1f}, {y_axis[y_hi10-1]:.1f}]:",
        f"  layer 1: N={chi1.size}  skew={sk1:.3f}  kurt={ku1:.3f}",
        f"  layer 2: N={chi2.size}  skew={sk2:.3f}  kurt={ku2:.3f}",
    ]
    with open(os.path.join(FIG_DIR, "02_growth_summary.txt"), "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print("\n".join(summary_lines[-5:]))

    # Save sliding-window arrays for later cross-amp comparison.
    np.savez(os.path.join(FIG_DIR, "sliding_window_stats.npz"),
             t_c=t_c_arr, y=y_axis,
             skew1_sw=skew1_sw, skew2_sw=skew2_sw,
             kurt1_sw=kurt1_sw, kurt2_sw=kurt2_sw)

    # -------- movies (q' and psi), GIF via PillowWriter --------
    print("Writing movies (this is the slow part)...", flush=True)
    write_movie(tf, q1f, q2f, mode='q',
                path=os.path.join(MOV_DIR, "q_prime.gif"),
                x_axis=x_axis, y_axis=y_axis)
    write_movie(tf, psi1f, psi2f, mode='psi',
                path=os.path.join(MOV_DIR, "psi.gif"),
                x_axis=x_axis, y_axis=y_axis)
    print("Done.")


def _erfinv(x):
    # Use scipy.special.erfinv if available.
    try:
        from scipy.special import erfinv
        return erfinv(x)
    except Exception:
        # Beasley-Springer-Moro fallback.
        x = np.clip(x, -1 + 1e-12, 1 - 1e-12)
        return np.sign(x) * np.sqrt(-np.log(1 - np.abs(x)))


def write_movie(t, f1, f2, mode, path, x_axis, y_axis, fps=15):
    """Animate two side-by-side fields and save as GIF (PillowWriter).

    We write GIF directly to avoid an ffmpeg dependency. For .mp4 output,
    swap the writer for animation.FFMpegWriter and change the extension.
    """
    if mode == 'q':
        a = f1 - f1.mean(axis=2, keepdims=True)
        b = f2 - f2.mean(axis=2, keepdims=True)
        labels = ("$q_1'$", "$q_2'$"); cmap = 'RdBu_r'
    else:
        a = f1; b = f2
        labels = ("$\\psi_1$", "$\\psi_2$"); cmap = 'RdBu_r'
    vmax = float(np.percentile(np.abs(np.concatenate([a.ravel(), b.ravel()])), 99))
    if vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    im1 = axes[0].imshow(a[0], origin='lower', aspect='auto', extent=extent,
                         vmin=-vmax, vmax=vmax, cmap=cmap)
    im2 = axes[1].imshow(b[0], origin='lower', aspect='auto', extent=extent,
                         vmin=-vmax, vmax=vmax, cmap=cmap)
    title = fig.suptitle(f"{labels[0]} | {labels[1]}     t = {t[0]:.2f}")
    for ax, lab in zip(axes, labels):
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lab)
    plt.colorbar(im1, ax=axes[0]); plt.colorbar(im2, ax=axes[1])

    def update(i):
        im1.set_data(a[i]); im2.set_data(b[i])
        title.set_text(f"{labels[0]} | {labels[1]}     t = {t[i]:.2f}")
        return im1, im2, title

    anim = animation.FuncAnimation(fig, update, frames=len(t),
                                   interval=1000 / fps, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    print(f"  wrote {path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
