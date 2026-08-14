import os, shutil, sys

import numpy as np

RED, YELLOW, GREEN, CYAN, DIM = "31", "33", "32", "36", "2"

_RAMP  = (16, 17, 53, 54, 90, 91, 125, 161, 197, 196, 202, 208, 214, 220, 226, 231)  # rough inferno in xterm-256
_ASCII = " .:-=+*#%@"


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def supports_color() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not sys.stdout.isatty():
        return False
    _enable_windows_ansi()
    return True


def blocks_ok() -> bool:
    """Block glyphs are 'ambiguous' width. Terminal.app draws them double-wide and
    every row after slides sideways. ghostty, iTerm and kitty are fine."""
    if os.environ.get("SPECTRO_BLOCKS"):
        return os.environ["SPECTRO_BLOCKS"] != "0"
    return os.environ.get("TERM_PROGRAM") != "Apple_Terminal"


COLOR = supports_color()
BLOCKS = blocks_ok()

FULL = "█" if BLOCKS else "#"
HALF = "▀"
EMPTY = "░" if BLOCKS else "·"
RULE = "─" if BLOCKS else "-"


def spectrogram_width(width: int = None) -> int:
    """Resolve the drawable terminal width in one place."""
    if width is not None:
        return width
    cols = shutil.get_terminal_size((100, 30)).columns
    return max(40, min(cols - 9, 140))


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
    s = FULL * filled + EMPTY * (width - filled)
    return paint(s, color) if color else s


def _pool(Sxx_db: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Max-pool the spectrogram down onto a character grid."""
    n_f, n_t = Sxx_db.shape
    f_edges = np.unique(np.linspace(0, n_f, min(rows, n_f) + 1).astype(int))[:-1]
    t_edges = np.unique(np.linspace(0, n_t, min(cols, n_t) + 1).astype(int))[:-1]
    pooled = np.maximum.reduceat(Sxx_db, f_edges, axis=0)
    return np.maximum.reduceat(pooled, t_edges, axis=1)


def _paint_run(cells) -> str:
    """One SGR per colour change instead of one per cell, keeps the line short."""
    out = []
    last = None
    for fg, bg, glyph in cells:
        if (fg, bg) != last:
            sgr = f"38;5;{_RAMP[fg]}" if bg is None else f"38;5;{_RAMP[fg]};48;5;{_RAMP[bg]}"
            out.append(f"\033[{sgr}m")
            last = (fg, bg)
        out.append(glyph)
    out.append("\033[0m")
    return "".join(out)


def _freq_label(freqs: np.ndarray, frac: float) -> str:
    hz = freqs[min(len(freqs) - 1, int(round(frac * (len(freqs) - 1))))]
    return f"{hz/1000:6.1f}k "


def render_term_spectrogram(Sxx_db: np.ndarray, freqs: np.ndarray, duration: float,
                            height: int = 18, width: int = None,
                            half: bool = True) -> None:
    width = spectrogram_width(width)

    # half-blocks pack two frequency bands into one text row
    half = half and COLOR and BLOCKS
    rows = height * 2 if half else height

    pooled = _pool(Sxx_db, rows, width)
    vmax = float(pooled.max())
    norm = np.clip((pooled - (vmax - 80.0)) / 80.0, 0.0, 1.0)
    h, w = norm.shape

    if half and h % 2:
        norm = norm[:-1]
        h -= 1

    gutter = " " * 8
    lines = [gutter + RULE * w]

    shade = (norm * (len(_RAMP) - 1)).astype(int)
    ramp = (norm * (len(_ASCII) - 1)).astype(int)

    if half:
        for r in range(h - 2, -1, -2):
            row = _paint_run((u, l, HALF) for u, l in zip(shade[r + 1], shade[r]))
            label = _freq_label(freqs, (r + 2) / h) if ((h - 2 - r) // 2) % 4 == 0 else gutter
            lines.append(label + row)
    else:
        for r in range(h - 1, -1, -1):
            if COLOR:
                # colour carries the level, and so does the glyph
                cells = [(s, None, FULL if BLOCKS else _ASCII[i])
                         for s, i in zip(shade[r], ramp[r])]
                row = _paint_run(cells)
            else:
                row = "".join(_ASCII[i] for i in ramp[r])
            label = _freq_label(freqs, (r + 1) / h) if (h - 1 - r) % 4 == 0 else gutter
            lines.append(label + row)

    lines.append(gutter + RULE * w)
    print("\n".join(lines))

    left, right = "0s", f"{duration:.0f}s"
    print(gutter + left + " " * max(1, w - len(left) - len(right)) + right)
