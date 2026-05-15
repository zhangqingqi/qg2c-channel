"""
Single-case wrapper around qg2c.run.

Adds:
  - Zero-base + low-wavenumber baroclinic-noise initial condition.
  - Diagnostic hook that accumulates scalar series + y-resolved series.
  - Optional field-snapshot hook for movies / Hovmollers / PDFs.
  - npz save helper.

The PDE itself is solved by qg2c.run; nothing here re-implements the
time-stepping or inversion.
"""

import os
import numpy as np

import qg2c
from diagnostics import (
    cell_center_velocity, psi_to_cell_center, perturb_x,
    variance_x, skewness_x, kurtosis_x,
    zonal_power_spectrum, spectral_signature,
    make_baroclinic_noise,
)


def _get_backend(name):
    """Return the solver module: 'scipy' -> qg2c, 'jax' -> qg2c_jax."""
    if name == 'scipy':
        return qg2c
    if name == 'jax':
        import qg2c_jax
        return qg2c_jax
    raise ValueError(f"unknown backend: {name}")


def run_one_case(amp=4e4, r=0.1, delta=0.2, seed=0,
                 r1=None, r2=None,
                 nx=128, ny=64,
                 Rd=40.0, beta=1.728e-3,
                 init_style='baroclinic-noise',
                 noise_amp=1e-3,
                 eps_noise=1e-3, kmax=4, nmax=4,
                 tmax=150.0, dt=1.0 / 128.0,
                 dt_save_scalars=0.25,
                 dt_save_fields=1.0,
                 save_fields=False,
                 verbose=False,
                 progress_every=None,
                 backend='scipy'):
    """Run one case. Returns a dict of diagnostics and (optionally) fields.

    backend: 'scipy' uses the reference qg2c.py; 'jax' uses qg2c_jax.py
    (jit-compiled, GPU-friendly). Both produce numerically equivalent
    results to float64 precision; hooks see numpy arrays either way.

    init_style:
      'baroclinic-noise' (default): low-wavenumber wall-compatible eta
        with q1 = +eta, q2 = -eta (RMS=eps_noise).
      'matlab-sin-noise': matches the matlab seed setup
        q1 = 0.01*sin(2*pi*x/L)*sin(pi*y/W) + noise_amp*randn
        q2 = q1 + noise_amp*randn (independent draw).
    """
    solver = _get_backend(backend)
    P = solver.setup_params(amp=amp, r=r, delta=delta, r1=r1, r2=r2,
                            nx=nx, ny=ny, Rd=Rd, beta=beta)

    if init_style == 'baroclinic-noise':
        eta = make_baroclinic_noise(P['x'], P['y'], P['L'], P['W'], seed,
                                    kmax=kmax, nmax=nmax, rms=eps_noise)
        q1_init = +eta.copy()
        q2_init = -eta.copy()
    elif init_style == 'matlab-sin-noise':
        rng = np.random.default_rng(seed)
        smooth = (0.01 * np.sin(2 * np.pi * P['x'] / P['L'])
                       * np.sin(np.pi * P['y'] / P['W']))
        q1_init = smooth + noise_amp * rng.standard_normal(P['x'].shape)
        q2_init = q1_init + noise_amp * rng.standard_normal(P['x'].shape)
    else:
        raise ValueError(f"unknown init_style: {init_style}")

    h1 = delta / (1.0 + delta)
    h2 = 1.0   / (1.0 + delta)

    series = dict(
        t=[], EKE=[], Zq=[], Zq1=[], Zq2=[],
        f_dom=[], spectral_entropy=[],
        max_q1p=[], max_q2p=[],
        Zq_y=[], Zq1_y=[], Zq2_y=[],
        skew1_y=[], skew2_y=[], kurt1_y=[], kurt2_y=[],
    )
    fields = dict(t=[], q1=[], q2=[], psi1=[], psi2=[]) if save_fields else None

    def diag(s, P_):
        q1p = perturb_x(s['q1'])
        q2p = perturb_x(s['q2'])
        u1c, v1c = cell_center_velocity(s['u1'], s['v1'])
        u2c, v2c = cell_center_velocity(s['u2'], s['v2'])
        u1p = perturb_x(u1c); v1p = perturb_x(v1c)
        u2p = perturb_x(u2c); v2p = perturb_x(v2c)

        EKE = 0.5 * (h1 * np.mean(u1p ** 2 + v1p ** 2)
                     + h2 * np.mean(u2p ** 2 + v2p ** 2))
        Zq1_v = 0.5 * np.mean(q1p ** 2)
        Zq2_v = 0.5 * np.mean(q2p ** 2)
        Zq = h1 * Zq1_v + h2 * Zq2_v

        Pavg = h1 * zonal_power_spectrum(q1p) + h2 * zonal_power_spectrum(q2p)
        fdom, Sk = spectral_signature(Pavg)

        series['t'].append(s['t'])
        series['EKE'].append(EKE)
        series['Zq'].append(Zq); series['Zq1'].append(Zq1_v); series['Zq2'].append(Zq2_v)
        series['f_dom'].append(fdom); series['spectral_entropy'].append(Sk)
        series['max_q1p'].append(float(np.max(np.abs(q1p))))
        series['max_q2p'].append(float(np.max(np.abs(q2p))))

        Zq1_y = 0.5 * variance_x(q1p)
        Zq2_y = 0.5 * variance_x(q2p)
        series['Zq_y'].append(h1 * Zq1_y + h2 * Zq2_y)
        series['Zq1_y'].append(Zq1_y); series['Zq2_y'].append(Zq2_y)
        series['skew1_y'].append(skewness_x(q1p))
        series['skew2_y'].append(skewness_x(q2p))
        series['kurt1_y'].append(kurtosis_x(q1p))
        series['kurt2_y'].append(kurtosis_x(q2p))

        if verbose:
            print(f"  t={s['t']:7.2f} EKE={EKE:.3e} Zq={Zq:.3e} "
                  f"max|q1p|={series['max_q1p'][-1]:.3e} "
                  f"f_dom={fdom:.3f} Sk={Sk:.3f}", flush=True)

    def snap(s, P_):
        fields['t'].append(s['t'])
        fields['q1'].append(s['q1'].copy())
        fields['q2'].append(s['q2'].copy())
        fields['psi1'].append(psi_to_cell_center(s['p1']))
        fields['psi2'].append(psi_to_cell_center(s['p2']))

    res = solver.run(P=P, q1_init=q1_init, q2_init=q2_init,
                     tmax=tmax, dt=dt,
                     diag_hook=diag, dt_diag=dt_save_scalars,
                     field_hook=snap if save_fields else None,
                     dt_field=dt_save_fields if save_fields else None,
                     verbose=False,
                     progress_every=progress_every)

    out = dict(
        amp=amp, r=r, delta=delta, seed=seed,
        r1=P['r1'], r2=P['r2'],
        Rd=Rd, beta=beta,
        init_style=init_style, noise_amp=noise_amp,
        backend=backend,
        eps_noise=eps_noise,
        kmax=kmax, nmax=nmax, tmax=tmax, dt=dt,
        nx=P['nx'], ny=P['ny'], L=P['L'], W=P['W'],
        F1=P['F1'], F2=P['F2'], h1=h1, h2=h2,
        x=P['x'][0, :], y=P['y'][:, 0],
        q1_final=res['q1_final'], q2_final=res['q2_final'],
        wall_seconds=res['wall'],
    )
    for k, v in series.items():
        out[k] = np.array(v)
    if save_fields:
        out['t_fields'] = np.array(fields['t'])
        out['q1_fields'] = np.array(fields['q1'])
        out['q2_fields'] = np.array(fields['q2'])
        out['psi1_fields'] = np.array(fields['psi1'])
        out['psi2_fields'] = np.array(fields['psi2'])
    return out


def save_npz(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in out.items()
                                 if not isinstance(v, str)})


if __name__ == '__main__':
    import sys
    amp = float(sys.argv[1]) if len(sys.argv) > 1 else 4e4
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"[smoke] amp={amp}, seed={seed}")
    o = run_one_case(amp=amp, seed=seed, tmax=20.0,
                     save_fields=True, dt_save_fields=2.0, verbose=True)
    print(f"[smoke] wall={o['wall_seconds']:.2f}s, "
          f"final EKE={o['EKE'][-1]:.3e}, final Zq={o['Zq'][-1]:.3e}")
