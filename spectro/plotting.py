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


def _seed_ascii_font_cache() -> None:
    """Avoid a full system-font crawl when bundled fonts cover the whole plot."""
    import ast
    import importlib.util
    import json
    import os
    import re
    import sys
    from pathlib import Path

    spec = importlib.util.find_spec("matplotlib")
    if not spec or not spec.submodule_search_locations:
        return
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    source_path = package_dir / "font_manager.py"
    try:
        source = source_path.read_text(encoding="utf-8")
        match = re.search(
            r"class FontManager:.*?__version__\s*=\s*([^\n#]+)",
            source, flags=re.DOTALL)
        if not match:
            return
        version = ast.literal_eval(match.group(1).strip())
    except (OSError, SyntaxError, ValueError):
        return

    configured = os.environ.get("MPLCONFIGDIR")
    if configured:
        # Respect an explicitly managed Matplotlib environment. It may rely on
        # system-font fallback beyond the isolated ASCII fast path below.
        return
    if sys.platform == "darwin":
        cache_dir = Path.home() / "Library" / "Caches" / "spectro" / "matplotlib"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = root / "spectro" / "matplotlib"

    cache_path = cache_dir / f"fontlist-v{version}.json"
    if cache_path.is_file():
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
        return

    has_index = "index: int = 0" in source

    def font_entry(filename, family, style="normal", weight=400):
        entry = {
            "fname": f"fonts/ttf/{filename}",
            "name": family,
            "style": style,
            "variant": "normal",
            "weight": weight,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry",
        }
        if has_index:
            entry["index"] = 0
        return entry

    fonts = [
        font_entry("DejaVuSansMono.ttf", "DejaVu Sans Mono"),
        font_entry("DejaVuSansMono-Bold.ttf", "DejaVu Sans Mono", weight=700),
        font_entry("DejaVuSansMono-Oblique.ttf", "DejaVu Sans Mono", "oblique"),
        font_entry("DejaVuSansMono-BoldOblique.ttf", "DejaVu Sans Mono", "oblique", 700),
        font_entry("DejaVuSans.ttf", "DejaVu Sans"),
        font_entry("DejaVuSans-Bold.ttf", "DejaVu Sans", weight=700),
    ]
    payload = {
        "_version": version,
        "_FontManager__default_weight": "normal",
        "default_size": None,
        "defaultFamily": {"ttf": "DejaVu Sans", "afm": "Helvetica"},
        "afmlist": [],
        "ttflist": fonts,
        "__class__": "FontManager",
    }

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, cache_path)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    except OSError:
        pass


def lazy_pyplot(required_text: str = None):
    # defer matplotlib until something actually plots
    if required_text is not None and required_text.isascii():
        _seed_ascii_font_cache()
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
