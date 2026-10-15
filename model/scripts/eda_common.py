"""Shared plotting helpers for the EDA report (one light theme, PNG output)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEME = dict(panel="#ffffff", ink="#1b1d1c", ink2="#585b58", grid="#e7e8e5", ref="#9a9c98",
             s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
             seq=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], band_alpha=0.16)

# diabetes-status groups (CGMacros) take categorical slots 1-3 in fixed order
GROUPS = ["Healthy", "Prediabetes", "T2D"]


def apply_theme():
    T = THEME
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 10.5,
        "figure.facecolor": T["panel"], "axes.facecolor": T["panel"], "savefig.facecolor": T["panel"],
        "axes.edgecolor": T["ink2"], "axes.labelcolor": T["ink2"], "axes.titlecolor": T["ink"],
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "xtick.color": T["ink2"], "ytick.color": T["ink2"], "text.color": T["ink"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": T["grid"], "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2,
        "axes.prop_cycle": matplotlib.cycler(color=T["s"]),
    })
    return T


def save_fig(draw, name, outdir, dpi=130, **kw):
    """draw(T) builds and returns a figure for theme dict T; writes <outdir>/<name>.png"""
    os.makedirs(outdir, exist_ok=True)
    fig = draw(apply_theme(), **kw)
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
