"""Generate docs/channel_diagram.png — the 2-layer QG channel schematic
used in README.md."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.lines import Line2D


HERE = os.path.dirname(os.path.abspath(__file__))


def isometric(x, y, z, sx=1.0, sy=0.55, sz=1.0):
    """Cabinet-style isometric projection (x to the right, y back, z up)."""
    px = sx * x + sy * 0.7 * y
    py = sz * z + sy * 0.7 * y
    return px, py


def face(ax, pts3d, **kw):
    pts2d = [isometric(*p) for p in pts3d]
    poly = Polygon(pts2d, closed=True, **kw)
    ax.add_patch(poly)
    return pts2d


def line3d(ax, p0, p1, **kw):
    x0, y0 = isometric(*p0)
    x1, y1 = isometric(*p1)
    ax.add_line(Line2D([x0, x1], [y0, y1], **kw))


def annot3d(ax, p, text, **kw):
    x, y = isometric(*p)
    ax.annotate(text, (x, y), **kw)


def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.6, 8.4)
    ax.set_ylim(-0.6, 4.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Channel dimensions (schematic, not to physical scale)
    L  = 5.0          # zonal
    W  = 2.6          # meridional
    H1 = 0.55          # upper layer thickness
    H2 = 1.30          # lower layer thickness
    z_top  = H1 + H2
    z_mid  = H2
    z_bot  = 0.0

    c_upper = "#cfe6ff"
    c_lower = "#d6f0d8"
    edge    = "#2b2b2b"

    # Back wall (y = W) — drawn first so front faces overlay it.
    face(ax, [(0, W, z_bot), (L, W, z_bot), (L, W, z_mid), (0, W, z_mid)],
         facecolor=c_lower, edgecolor=edge, linewidth=1.2, alpha=0.55)
    face(ax, [(0, W, z_mid), (L, W, z_mid), (L, W, z_top), (0, W, z_top)],
         facecolor=c_upper, edgecolor=edge, linewidth=1.2, alpha=0.55)

    # Right end wall (x = L)
    face(ax, [(L, 0, z_bot), (L, W, z_bot), (L, W, z_mid), (L, 0, z_mid)],
         facecolor=c_lower, edgecolor=edge, linewidth=1.4)
    face(ax, [(L, 0, z_mid), (L, W, z_mid), (L, W, z_top), (L, 0, z_top)],
         facecolor=c_upper, edgecolor=edge, linewidth=1.4)

    # Front face (y = 0) — main visible cross-section
    face(ax, [(0, 0, z_bot), (L, 0, z_bot), (L, 0, z_mid), (0, 0, z_mid)],
         facecolor=c_lower, edgecolor=edge, linewidth=1.6)
    face(ax, [(0, 0, z_mid), (L, 0, z_mid), (L, 0, z_top), (0, 0, z_top)],
         facecolor=c_upper, edgecolor=edge, linewidth=1.6)

    # Top lid
    face(ax, [(0, 0, z_top), (L, 0, z_top), (L, W, z_top), (0, W, z_top)],
         facecolor="#f5f5f5", edgecolor=edge, linewidth=1.4, alpha=0.75)

    # Interface (slight wave to suggest baroclinic deformation)
    xs = np.linspace(0, L, 60)
    ys_back = np.full_like(xs, W)
    ys_front = np.zeros_like(xs)
    z_int_front = z_mid + 0.10 * np.sin(2 * np.pi * xs / L) * np.cos(0.3 * xs)
    z_int_back  = z_mid + 0.10 * np.sin(2 * np.pi * xs / L) * np.cos(0.3 * xs)
    for x, yf, yb, zf, zb in zip(xs[:-1], ys_front[:-1], ys_back[:-1],
                                  z_int_front[:-1], z_int_back[:-1]):
        pass
    # Just draw the front-edge interface curve, as a visual cue.
    pts = [isometric(x, 0.0, z) for x, z in zip(xs, z_int_front)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color="#1f5fbf", linewidth=2.0, alpha=0.9)

    # Periodicity arrows on the top lid: x wraps around
    def top_arrow(x0, x1):
        p0 = isometric(x0, W * 0.5, z_top + 0.02)
        p1 = isometric(x1, W * 0.5, z_top + 0.02)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                      mutation_scale=14,
                                      color="#444", linewidth=1.2))
    top_arrow(0.4, 1.5)
    top_arrow(L - 1.5, L - 0.4)
    ax.text(*isometric(L / 2, W * 0.5, z_top + 0.05),
            "periodic in $x$, period $L = 2W$",
            ha="center", va="bottom", fontsize=11)

    # Wall annotations
    line3d(ax, (0, 0, z_bot), (0, W, z_bot), color="#000", linewidth=2.4)
    line3d(ax, (0, W, z_bot), (0, W, z_top), color="#000", linewidth=2.4)
    line3d(ax, (L, 0, z_bot), (L, W, z_bot), color="#000", linewidth=2.4)
    line3d(ax, (L, W, z_bot), (L, W, z_top), color="#000", linewidth=2.4)

    annot3d(ax, (-0.05, 0.0, z_top + 0.05),
            "$y = 0$ wall  (no-flux, $v = 0$)",
            ha="right", va="bottom", fontsize=10, color="#222")
    annot3d(ax, (-0.05, W, z_top + 0.05),
            "$y = W$ wall  (no-flux, $v = 0$)",
            ha="right", va="bottom", fontsize=10, color="#222")

    # Layer labels (on the visible front face)
    annot3d(ax, (L * 0.5, 0, z_mid + (z_top - z_mid) * 0.55),
            r"Layer 1:  $\psi_1$, $q_1$,  thickness $H_1 = \delta/(1+\delta)$",
            ha="center", va="center", fontsize=11, color="#1f3a6b",
            fontweight="bold")
    annot3d(ax, (L * 0.5, 0, z_bot + z_mid * 0.5),
            r"Layer 2:  $\psi_2$, $q_2$,  thickness $H_2 = 1/(1+\delta)$",
            ha="center", va="center", fontsize=11, color="#1f5f30",
            fontweight="bold")

    # Interface label
    annot3d(ax, (L + 0.10, 0, z_mid),
            "interface", ha="left", va="center",
            fontsize=10, color="#1f5fbf")

    # Axis arrows: x, y, z
    p0 = isometric(0, 0, z_bot)
    px = isometric(0.9, 0, z_bot)
    py = isometric(0, 1.0, z_bot)
    pz = isometric(0, 0, z_bot + 0.9)
    for tip, lbl, off in [(px, r"$x$", (0.02, -0.18)),
                          (py, r"$y$", (0.05, 0.04)),
                          (pz, r"$z$", (-0.18, 0.02))]:
        ax.add_patch(FancyArrowPatch(p0, tip, arrowstyle="-|>",
                                      mutation_scale=14,
                                      color="#222", linewidth=1.5))
        ax.text(tip[0] + off[0], tip[1] + off[1], lbl, fontsize=12)

    # Forcing profile sketch, well clear of the channel
    y_prof = np.linspace(0, W, 50)
    qf_y = np.cos(np.pi * y_prof / W)            # cos(pi y/W) in [-1, 1]
    ox, oy = 7.0, 0.4
    pw, ph = 1.0, 2.6
    base_y = np.linspace(0, ph, len(y_prof))
    cx0 = ox + 0.5 * pw
    curve_x = cx0 + 0.5 * pw * qf_y
    # axis line + bracket
    ax.plot([cx0, cx0], [oy, oy + ph], color="#888", linewidth=1.0, linestyle=":")
    ax.plot([ox, ox + pw], [oy, oy], color="#888", linewidth=1.0)
    ax.plot([ox, ox + pw], [oy + ph, oy + ph], color="#888", linewidth=1.0)
    ax.plot(curve_x, oy + base_y, color="#c0392b", linewidth=2.2)
    ax.text(cx0, oy + ph + 0.12,
            r"$q_{\rm force}(y) = A\cos(\pi y/W)$",
            ha="center", va="bottom", fontsize=11, color="#c0392b")
    ax.text(ox - 0.05, oy + ph, "$y=W$", fontsize=10, va="center", ha="right")
    ax.text(ox - 0.05, oy,       "$y=0$", fontsize=10, va="center", ha="right")

    # Title
    ax.text(0.0, 3.7,
            "Two-layer QG beta-plane channel",
            fontsize=15, fontweight="bold")

    out = os.path.join(HERE, "channel_diagram.png")
    fig.savefig(out, dpi=160, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
