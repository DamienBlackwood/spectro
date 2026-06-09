INK    = "#0e0e12"
PANEL  = "#15151b"
TEXT   = "#c9c9d4"
MUTED  = "#6d6d7e"
GRID   = "#26262f"
ACCENT = "#7ad0ff"
ORANGE = "#ff9a3c"

VERDICT_COLORS = {
    "PASS": "#3ddc84",
    "WARN": "#ffcc4d",
    "FAIL": "#ff5555",
    "INCONCLUSIVE": "#9a9aa8",
}


def lazy_pyplot():
    # defer matplotlib until something actually plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor":  INK,
        "savefig.facecolor": INK,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "text.color":        TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        GRID,
        "grid.alpha":        0.6,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    10,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })
    return plt
