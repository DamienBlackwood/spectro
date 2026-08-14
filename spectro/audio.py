from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import numpy as np


_AUDIO_MAGIC = (b'RIFF', b'fLaC', b'OggS', b'ID3', b'FORM', b'\xff\xfb', b'\xff\xf1')
_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.opus', '.mp3', '.m4a', '.aac', '.wma',
               '.aiff', '.aif', '.ape', '.wv', '.mpc', '.dff', '.dsf', '.caf',
               '.alac', '.mp4', '.oga', '.au', '.w64', '.tta'}


def looks_like_audio(path: Path) -> bool:
    """Cheap audio-file sniff by extension or magic bytes."""
    if path.suffix.lower() in _AUDIO_EXTS:
        return True
    try:
        with open(path, 'rb') as f:
            head = f.read(12)
    except OSError:
        return False
    if head[4:8] == b'ftyp':  # m4a/mp4 box, the brand sits after it
        return True
    return any(head.startswith(m) for m in _AUDIO_MAGIC)


def open_file(path: str) -> None:
    """Open file with system default application."""
    try:
        if platform.system() == 'Darwin':
            subprocess.run(['open', path], check=True)
        elif platform.system() == 'Windows':
            os.startfile(path)
        else:
            subprocess.run(['xdg-open', path], check=True)
    except Exception:
        pass


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio with soundfile, fallback to FFmpeg for unsupported formats."""
    import soundfile as sf

    try:
        return sf.read(file_path, dtype='float32')
    except (sf.LibsndfileError, OSError, RuntimeError):
        import shutil, tempfile
        if not shutil.which('ffmpeg'):
            print("Error: FFmpeg not found. Install it to handle this file format.", file=sys.stderr)
            sys.exit(1)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-v', 'error', '-i', file_path,
                 '-acodec', 'pcm_f32le', tmp_path],
                check=True, capture_output=True
            )
            return sf.read(tmp_path, dtype='float32')
        except subprocess.CalledProcessError:
            print(f"Error: Could not decode '{file_path}'. Doesn't look like a valid audio file.", file=sys.stderr)
            sys.exit(1)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
