import os, shutil, sys

import numpy as np

RED, YELLOW, GREEN, CYAN, DIM = "31", "33", "32", "36", "2"

_RAMP  = (16, 17, 53, 54, 90, 91, 125, 161, 197, 196, 202, 208, 214, 220, 226, 231)  # rough inferno in xterm-256
_ASCII = " .:-=+*#%@"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


COLOR = supports_color()


def paint(text: str, code: str, bold: bool = False) -> str:
    if not COLOR:
        return text
    b = "1;" if bold else ""
    return f"\033[{b}{code}m{text}\033[0m"


def verdict_color(verdict: str) -> str:
    return {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}.get(verdict, CYAN)


def severity_color(sev: str) -> str:
    return {"high": RED, "medium": YELLOW, "low": CYAN}.get(sev, DIM)


def bar(value: float, width: int = 20, color: str = None) -> str:
    filled = int(round(width * max(0.0, min(100.0, value)) / 100))
    s = "█" * filled + "░" * (width - filled)
    return paint(s, color) if color else s


def render_term_spectrogram(Sxx_db: np.ndarray, freqs: np.ndarray, duration: float,
                            height: int = 18, width: int = None) -> None:
    if width is None:
        cols = shutil.get_terminal_size((100, 30)).columns
        width = max(40, min(cols - 9, 140))

    n_f, n_t = Sxx_db.shape
    height = min(height, n_f)
    width = min(width, n_t)

    # max-pool onto the character grid
    f_edges = np.unique(np.linspace(0, n_f, height + 1).astype(int))[:-1]
    t_edges = np.unique(np.linspace(0, n_t, width + 1).astype(int))[:-1]
    pooled = np.maximum.reduceat(Sxx_db, f_edges, axis=0)
    pooled = np.maximum.reduceat(pooled, t_edges, axis=1)

    vmax = float(pooled.max())
    norm = np.clip((pooled - (vmax - 80.0)) / 80.0, 0.0, 1.0)

    h, w = norm.shape
    lines = []
    for r in range(h - 1, -1, -1):
        f_hi = freqs[min(n_f - 1, int(round((r + 1) / h * (n_f - 1))))] / 1000.0
        label = f"{f_hi:6.1f}k " if (h - 1 - r) % 4 == 0 else "        "
        if COLOR:
            idx = (norm[r] * (len(_RAMP) - 1)).astype(int)
            row = "".join(f"\033[38;5;{_RAMP[i]}m█" for i in idx) + "\033[0m"
        else:
            idx = (norm[r] * (len(_ASCII) - 1)).astype(int)
            row = "".join(_ASCII[i] for i in idx)
        lines.append(label + row)

    print("\n".join(lines))
    left, right = "0s", f"{duration:.0f}s"
    print("        " + left + " " * max(1, w - len(left) - len(right)) + right)
