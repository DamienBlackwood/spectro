import time

import numpy as np
from scipy.signal import stft

from ..dataclasses_ import T
from ..reporting import fmt_time
from ..term import render_term_spectrogram


def cmd_preview(data_display: np.ndarray, sr: int, script_start: float) -> None:
    noverlap = int(T.display_nperseg * T.display_overlap)
    frequencies, times, Zxx = stft(data_display, fs=sr, nperseg=T.display_nperseg,
                                   noverlap=noverlap, window='hann')
    Sxx_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    print()
    render_term_spectrogram(Sxx_db, frequencies, duration=len(data_display) / sr)

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)}")
