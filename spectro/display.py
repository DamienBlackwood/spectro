"""Exact, bounded-memory analysis for visual spectrogram modes.

The original visual paths computed a full overlapping STFT and then retained
only every Nth time frame. This module derives those retained frame positions
first and computes the identical windows directly. Nothing represented in the
old output is skipped; only FFT results that were immediately discarded are
never created.
"""

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .dataclasses_ import DynamicsResult, T
from .dynamics import DynamicsAccumulator


@dataclass
class DisplayFrames:
    windows: np.ndarray
    sample_rate: int
    frame_count: int
    channels: int
    times: np.ndarray
    dynamics: DynamicsResult
    full_stft_frames: int
    time_decimation: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.sample_rate


@dataclass
class DisplaySpectrogram:
    frequencies: np.ndarray
    times: np.ndarray
    Sxx_db: np.ndarray
    frequency_resolution: float


def retained_frame_layout(frame_count: int, nperseg: int,
                          max_time_bins: int):
    """Match scipy.signal.stft(..., boundary='zeros', padded=True)[::N]."""
    noverlap = nperseg // 2
    hop = nperseg - noverlap
    boundary = nperseg // 2
    extended_delta = frame_count + 2 * boundary - nperseg
    full_stft_frames = ((extended_delta + hop - 1) // hop) + 1
    time_decimation = max(1, full_stft_frames // max_time_bins)
    frame_indices = np.arange(0, full_stft_frames, time_decimation,
                              dtype=np.int64)
    starts = frame_indices * hop - boundary
    return starts, frame_indices, full_stft_frames, time_decimation


class _RetainedFrameCollector:
    def __init__(self, frame_count: int, sample_rate: int,
                 nperseg: int, max_time_bins: int):
        starts, frame_indices, full_count, decimation = retained_frame_layout(
            frame_count, nperseg, max_time_bins)
        self.nperseg = nperseg
        self.starts = starts
        self.valid_starts = np.maximum(starts, 0)
        self.valid_ends = np.minimum(starts + nperseg, frame_count)
        self.windows = np.zeros((len(starts), nperseg), dtype=np.float32)
        hop = nperseg - nperseg // 2
        self.times = ((frame_indices * hop + nperseg / 2) / sample_rate -
                      (nperseg / 2) / sample_rate)
        self.full_stft_frames = full_count
        self.time_decimation = decimation
        self.frame_offset = 0
        self.cursor = 0

    def update(self, mono: np.ndarray) -> None:
        block_start = self.frame_offset
        block_end = block_start + len(mono)
        i = self.cursor

        while i < len(self.starts) and self.valid_ends[i] <= block_start:
            i += 1
        j = i
        while j < len(self.starts) and self.valid_starts[j] < block_end:
            source_start = max(int(self.valid_starts[j]), block_start)
            source_end = min(int(self.valid_ends[j]), block_end)
            if source_end > source_start:
                source_offset = source_start - block_start
                target_offset = source_start - int(self.starts[j])
                count = source_end - source_start
                self.windows[j, target_offset:target_offset + count] = (
                    mono[source_offset:source_offset + count])
            j += 1

        while (self.cursor < len(self.starts) and
               self.valid_ends[self.cursor] <= block_end):
            self.cursor += 1
        self.frame_offset = block_end


def _blocksize(accumulator: DynamicsAccumulator) -> int:
    windows_per_block = max(1, 1_048_576 // accumulator.window_size)
    return windows_per_block * accumulator.window_size


def analyze_display_file(file_path: str,
                         max_time_bins: int = 2000) -> DisplayFrames:
    """Scan a seekable file once for exact dynamics and retained STFT frames."""
    with sf.SoundFile(file_path) as audio:
        frame_count = len(audio)
        if frame_count == 0:
            raise ValueError("audio has no samples")

        sample_rate = int(audio.samplerate)
        channels = int(audio.channels)
        nperseg = min(T.display_nperseg, frame_count)
        accumulator = DynamicsAccumulator(sample_rate, channels)
        collector = _RetainedFrameCollector(
            frame_count, sample_rate, nperseg, max_time_bins)

        for block in audio.blocks(blocksize=_blocksize(accumulator),
                                  dtype='float32', always_2d=True):
            mono = accumulator.update(block)
            collector.update(mono)

    return DisplayFrames(
        windows=collector.windows,
        sample_rate=sample_rate,
        frame_count=frame_count,
        channels=channels,
        times=collector.times,
        dynamics=accumulator.finish(),
        full_stft_frames=collector.full_stft_frames,
        time_decimation=collector.time_decimation,
    )


def analyze_display_array(data: np.ndarray, sample_rate: int,
                          max_time_bins: int = 2000) -> DisplayFrames:
    """Exact in-memory fallback for formats decoded through FFmpeg."""
    frame_count = len(data)
    if frame_count == 0:
        raise ValueError("audio has no samples")

    data_2d = data[:, None] if data.ndim == 1 else data
    channels = data_2d.shape[1]
    nperseg = min(T.display_nperseg, frame_count)
    accumulator = DynamicsAccumulator(sample_rate, channels)
    collector = _RetainedFrameCollector(
        frame_count, sample_rate, nperseg, max_time_bins)
    blocksize = _blocksize(accumulator)

    for start in range(0, frame_count, blocksize):
        block = np.asarray(data_2d[start:start + blocksize], dtype=np.float32)
        mono = accumulator.update(block)
        collector.update(mono)

    return DisplayFrames(
        windows=collector.windows,
        sample_rate=sample_rate,
        frame_count=frame_count,
        channels=channels,
        times=collector.times,
        dynamics=accumulator.finish(),
        full_stft_frames=collector.full_stft_frames,
        time_decimation=collector.time_decimation,
    )


def analyze_display(file_path: str,
                    max_time_bins: int = 2000) -> DisplayFrames:
    """Use the streaming path when possible, with an exact FFmpeg fallback."""
    try:
        return analyze_display_file(file_path, max_time_bins)
    except (sf.LibsndfileError, OSError, RuntimeError):
        from .audio import load_audio
        data, sample_rate = load_audio(file_path)
        return analyze_display_array(data, sample_rate, max_time_bins)


def compute_display_spectrogram(frames: DisplayFrames,
                                preview_floor: bool = False
                                ) -> DisplaySpectrogram:
    """Compute the same retained STFT values as scipy.signal.stft."""
    from scipy import fft

    nperseg = frames.windows.shape[1]
    if nperseg == 1:
        hann = np.ones(1, dtype=np.float32)
    else:
        hann = np.hanning(nperseg + 1)[:-1].astype(np.float32)
    # scipy.signal.stft casts the window to float32 before its in-place scale.
    scale = float(np.sum(hann))

    spectrum = fft.rfft(frames.windows * hann, axis=1)
    spectrum *= 1.0 / scale
    magnitude = np.abs(spectrum)
    if preview_floor:
        Sxx_db = 20 * np.log10(magnitude + 1e-10)
    else:
        Sxx_db = 10 * np.log10(magnitude * magnitude + 1e-10)
    frequencies = fft.rfftfreq(nperseg, 1 / frames.sample_rate)

    return DisplaySpectrogram(
        frequencies=frequencies,
        times=frames.times,
        Sxx_db=Sxx_db.T,
        frequency_resolution=frames.sample_rate / nperseg,
    )
