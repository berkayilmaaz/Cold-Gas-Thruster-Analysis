"""
Visualization for pneumatic analysis results.

Dark theme consistent with brky.ai project page design system.
All plots use the same color palette derived from the team logos.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .nozzle import NozzleGeometry, NozzlePerformance
from .pipe import PipeFlowResult

# color palette (from STYLE_GUIDE.md)
C = {
    "blue":      "#7EC8E3",
    "red":       "#C2506A",
    "yellow":    "#eab308",
    "green":     "#22c55e",
    "purple":    "#a855f7",
    "orange":    "#f97316",
    "bg":        "#0e1117",
    "grid":      "#21262d",
    "zeroline":  "#30363d",
    "text":      "#c9d1d9",
    "text_sec":  "#8b949e",
}


def _apply_theme(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply dark engineering theme to an axis."""
    ax.set_facecolor(C["bg"])
    ax.figure.patch.set_facecolor(C["bg"])

    if title:
        ax.set_title(title, color=C["text"], fontsize=13,
                     fontfamily="sans-serif", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["text_sec"], fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["text_sec"], fontsize=11)

    ax.tick_params(colors=C["text_sec"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C["zeroline"])
    ax.grid(True, color=C["grid"], linewidth=0.5, alpha=0.7)


def plot_nozzle_profile(geom: NozzleGeometry, ax=None) -> plt.Figure:
    """2D meridional cross-section of the CD nozzle."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    x, r_up, r_lo = geom.profile(n_points=120)

    # convert to mm for display
    x_mm = x * 1e3
    r_up_mm = r_up * 1e3
    r_lo_mm = r_lo * 1e3

    # nozzle walls (outer envelope)
    ax.fill_between(x_mm, r_up_mm * 1.25, r_up_mm, color="#2d3040", alpha=0.8)
    ax.fill_between(x_mm, r_lo_mm, r_lo_mm * 1.25, color="#2d3040", alpha=0.8)

    # inner contour lines
    ax.plot(x_mm, r_up_mm, color=C["blue"], linewidth=2.0)
    ax.plot(x_mm, r_lo_mm, color=C["blue"], linewidth=2.0)

    # Mach gradient fill (blue → red, low → high speed)
    n = len(x_mm)
    for i in range(n - 1):
        frac = i / n
        c = plt.cm.coolwarm(frac * 0.85 + 0.07)
        ax.fill_between(x_mm[i:i+2], r_lo_mm[i:i+2], r_up_mm[i:i+2],
                        color=c, alpha=0.12)

    # throat marker
    r_t_mm = geom.d_throat * 1e3 / 2.0
    r_e_mm = geom.d_exit * 1e3 / 2.0
    ax.axvline(x=0, color=C["yellow"], linewidth=0.8, linestyle="--", alpha=0.5)

    # dimension annotations
    ax.annotate("", xy=(0, r_t_mm + 0.5), xytext=(0, -r_t_mm - 0.5),
                arrowprops=dict(arrowstyle="<->", color=C["yellow"], lw=1.5))
    ax.text(1.5, 0, f"D* = {geom.d_throat*1e3:.2f} mm",
            color=C["text"], fontsize=10, fontfamily="monospace", ha="left", va="center")

    x_exit = x_mm[-1]
    ax.annotate("", xy=(x_exit, r_e_mm + 0.5), xytext=(x_exit, -r_e_mm - 0.5),
                arrowprops=dict(arrowstyle="<->", color=C["green"], lw=1.5))
    ax.text(x_exit + 1, 0, f"De = {geom.d_exit*1e3:.2f} mm",
            color=C["text"], fontsize=10, fontfamily="monospace", ha="left", va="center")

    ax.text(0, max(r_up_mm) * 1.35, "THROAT", color=C["yellow"],
            fontsize=9, ha="center", fontfamily="sans-serif", fontweight="bold")
    ax.text(x_exit * 0.5, max(r_up_mm) * 1.35, "FLOW →", color=C["blue"],
            fontsize=9, ha="center", fontfamily="sans-serif", fontweight="bold")

    _apply_theme(ax, xlabel="Axial position [mm]", ylabel="Radial position [mm]")
    ax.set_aspect("equal")
    ax.set_xlim(x_mm[0] * 1.2, x_mm[-1] * 1.35)
    ax.set_ylim(-max(r_up_mm) * 1.6, max(r_up_mm) * 1.6)

    return fig


def plot_thrust_vs_pressure(pressures, thrusts, ax=None,
                            reference_lines: dict = None) -> plt.Figure:
    """
    Thrust as a function of chamber pressure.
    reference_lines: dict of {label: pressure_value_Pa} for vertical markers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    p_bar = pressures / 1e5
    ax.plot(p_bar, thrusts, color=C["blue"], linewidth=2.5, label="Thrust [N]")
    ax.fill_between(p_bar, thrusts, alpha=0.06, color=C["blue"])

    if reference_lines:
        colors_cycle = [C["yellow"], C["red"], C["green"], C["purple"]]
        for i, (label, p_val) in enumerate(reference_lines.items()):
            c = colors_cycle[i % len(colors_cycle)]
            ax.axvline(x=p_val / 1e5, color=c, linestyle="--", alpha=0.7, linewidth=1)
            ax.text(p_val / 1e5 + 2, max(thrusts) * (0.9 - 0.15 * i),
                    label, color=c, fontsize=9, fontfamily="sans-serif")

    _apply_theme(ax, title="Thrust vs Chamber Pressure",
                 xlabel="Chamber Pressure [bar]", ylabel="Thrust [N]")
    ax.legend(facecolor="#161b22", edgecolor=C["zeroline"], labelcolor=C["text"])

    return fig


def plot_loss_vs_diameter(diameters, losses, p_in: float,
                          ax=None) -> plt.Figure:
    """Pressure loss vs pipe inner diameter."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    d_mm = diameters * 1e3
    dp_bar = losses / 1e5

    ax.plot(d_mm, dp_bar, color=C["purple"], linewidth=2.5)
    ax.fill_between(d_mm, dp_bar, alpha=0.06, color=C["purple"])

    # 10% loss threshold
    threshold = p_in / 1e5 * 0.10
    ax.axhline(y=threshold, color=C["red"], linestyle="--", alpha=0.7)
    ax.text(d_mm[-1] * 0.65, threshold * 1.1,
            f"10% loss threshold ({threshold:.1f} bar)",
            color=C["red"], fontsize=9, fontfamily="sans-serif")

    _apply_theme(ax, title="Pressure Loss vs Pipe Diameter",
                 xlabel="Inner Diameter [mm]", ylabel="Pressure Loss [bar]")

    return fig


def plot_blowdown(blowdown: dict, ax=None) -> plt.Figure:
    """Tank blowdown time-series: pressure, thrust, mass."""
    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    else:
        fig = ax.figure
        axes = fig.axes

    t = blowdown["t"]

    # tank pressure
    axes[0].plot(t, blowdown["p_tank"] / 1e5, color=C["blue"], linewidth=2)
    _apply_theme(axes[0], title="Tank Blowdown", ylabel="Tank Pressure [bar]")

    # thrust
    axes[1].plot(t, blowdown["thrust"], color=C["green"], linewidth=2)
    axes[1].fill_between(t, blowdown["thrust"], alpha=0.08, color=C["green"])
    _apply_theme(axes[1], ylabel="Thrust [N]")

    # gas mass
    axes[2].plot(t, blowdown["mass_gas"] * 1e3, color=C["orange"], linewidth=2)
    _apply_theme(axes[2], xlabel="Time [s]", ylabel="Gas Mass [g]")

    fig.tight_layout()
    return fig
