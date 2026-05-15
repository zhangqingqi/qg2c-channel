"""Run one case (signature or detailed) and optionally invoke analyze_baseline."""

import argparse
import os
import subprocess
import sys

from run_one_case import run_one_case, save_npz


HERE = os.path.dirname(os.path.abspath(__file__))


def build_tag(amp, r1, r2, delta, seed):
    """Asymmetric form 'r1_X_r2_Y' if r1!=r2; else fall back to legacy 'r{r}'."""
    if r1 == r2:
        return f"run_amp{amp:.4g}_r{r1:.3g}_d{delta:.3g}_s{seed}"
    return (f"run_amp{amp:.4g}_r1_{r1:.3g}_r2_{r2:.3g}"
            f"_d{delta:.3g}_s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, required=True)
    ap.add_argument("--r", type=float, default=0.1,
                    help="symmetric damping (used for both layers if --r1/--r2 omitted)")
    ap.add_argument("--r1", type=float, default=None,
                    help="upper-layer damping (overrides --r for layer 1)")
    ap.add_argument("--r2", type=float, default=None,
                    help="lower-layer damping (overrides --r for layer 2)")
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=150.0)
    ap.add_argument("--dt", type=float, default=1.0 / 128.0)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--dt-save-fields", type=float, default=None,
                    help="cadence for detailed field snapshots "
                         "(default 1.0 detailed / 5.0 not)")
    ap.add_argument("--run-dir", default="outputs/runs_detailed")
    ap.add_argument("--detailed", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing npz instead of skipping")
    ap.add_argument("--progress-every", type=float, default=25.0,
                    help="print heartbeat every N simulated days (None to disable)")
    ap.add_argument("--backend", choices=["scipy", "jax"], default="scipy",
                    help="numerical backend: 'scipy' (numpy/CPU) or 'jax' (GPU-jit)")
    ap.add_argument("--Rd", type=float, default=40.0,
                    help="Rossby deformation radius")
    ap.add_argument("--beta", type=float, default=1.728e-3,
                    help="beta plane gradient of Coriolis")
    ap.add_argument("--init-style", choices=["baroclinic-noise", "matlab-sin-noise"],
                    default="baroclinic-noise",
                    help="initial condition style")
    ap.add_argument("--noise-amp", type=float, default=1e-3,
                    help="noise amplitude for IC (used by both styles)")
    args = ap.parse_args()

    r1 = args.r if args.r1 is None else args.r1
    r2 = args.r if args.r2 is None else args.r2

    run_dir = os.path.join(HERE, args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    tag = build_tag(args.amp, r1, r2, args.delta, args.seed)
    path = os.path.join(run_dir, tag + ".npz")

    if os.path.exists(path) and not args.force:
        print(f"skip (exists): {path}", flush=True)
    else:
        dt_save = args.dt_save_fields
        if dt_save is None:
            dt_save = 1.0 if args.detailed else 5.0
        print(f"run: {tag}  backend={args.backend}  detailed={args.detailed}  "
              f"dt={args.dt}  r1={r1} r2={r2}  nx={args.nx} ny={args.ny}  "
              f"dt_save_fields={dt_save}", flush=True)
        o = run_one_case(
            amp=args.amp, r=args.r, delta=args.delta, seed=args.seed,
            r1=r1, r2=r2,
            nx=args.nx, ny=args.ny,
            Rd=args.Rd, beta=args.beta,
            init_style=args.init_style, noise_amp=args.noise_amp,
            tmax=args.tmax, dt=args.dt,
            save_fields=args.detailed,
            dt_save_fields=dt_save,
            progress_every=args.progress_every,
            backend=args.backend,
        )
        save_npz(o, path)
        print(f"wrote {path}  wall={o['wall_seconds']:.1f}s  "
              f"final EKE={o['EKE'][-1]:.3e}  Zq={o['Zq'][-1]:.3e}",
              flush=True)

    if args.analyze:
        label = args.label or tag
        rc = subprocess.call([
            sys.executable, os.path.join(HERE, "analyze_baseline.py"),
            "--amp", str(args.amp),
            "--r", str(args.r),
            "--r1", str(r1),
            "--r2", str(r2),
            "--delta", str(args.delta),
            "--seed", str(args.seed),
            "--run-dir", run_dir,
            "--label", label,
        ])
        sys.exit(rc)


if __name__ == "__main__":
    main()
