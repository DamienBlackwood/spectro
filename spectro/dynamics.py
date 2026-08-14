import numpy as np

from .dataclasses_ import DR_THRESHOLDS, DynamicsResult, T


def _dr_rating(crest_factor: float) -> str:
    if crest_factor < DR_THRESHOLDS['brickwalled']:
        return "brickwalled"
    if crest_factor < DR_THRESHOLDS['compressed']:
        return "compressed"
    if crest_factor < DR_THRESHOLDS['moderate']:
        return "moderate"
    return "dynamic"


def clip_runs(peaks: np.ndarray, sr: int, limit: int = 8) -> list:
    """Start times of the first few clipped stretches, not every sample."""
    hits = np.flatnonzero(peaks >= T.clip_threshold)
    if len(hits) == 0:
        return []
    # a run is a group of hits with no real gap between them
    gap = max(1, int(0.01 * sr))
    starts = hits[np.r_[True, np.diff(hits) > gap]]
    return (starts[:limit] / sr).tolist()


def analyze_dynamics(data: np.ndarray, sr: int) -> DynamicsResult:
    """Analyze exact dynamics in bounded chunks, even for in-memory audio."""
    data_2d = data[:, None] if data.ndim == 1 else data
    channels = data_2d.shape[1] if data_2d.ndim == 2 else 1
    accumulator = DynamicsAccumulator(sr, channels)
    windows_per_block = max(1, 1_048_576 // accumulator.window_size)
    blocksize = windows_per_block * accumulator.window_size
    for start in range(0, len(data_2d), blocksize):
        accumulator.update(data_2d[start:start + blocksize])
    return accumulator.finish()


class DynamicsAccumulator:
    """Exact whole-file dynamics in one bounded-memory streaming pass."""

    def __init__(self, sr: int, channels: int):
        self.sr = sr
        self.channels = channels
        self.window_size = max(1, int(T.clip_window_sec * sr))
        self.clip_gap = max(1, int(0.01 * sr))

        self.peak = 0.0
        self.sum_squares = 0.0
        self.sample_values = 0
        self.frame_offset = 0
        self.clipped_samples = 0
        self.clip_times = []
        self.last_clip_hit = None
        self.window_rms_db = []
        self.mono_pending = np.empty(0, dtype=np.float32)

    def update(self, block: np.ndarray) -> np.ndarray:
        """Consume an always-2D float32 block and return its mono downmix."""
        if block.ndim != 2 or block.shape[1] != self.channels:
            raise ValueError("dynamics block has the wrong channel shape")
        if len(block) == 0:
            return np.empty(0, dtype=np.float32)

        block_min = float(np.min(block))
        block_max = float(np.max(block))
        block_peak = max(abs(block_min), abs(block_max))
        self.peak = max(self.peak, block_peak)

        flat = block.reshape(-1)
        self.sum_squares += float(np.dot(flat, flat))
        self.sample_values += block.size

        if block_peak >= T.clip_threshold:
            clipped = np.abs(block) >= T.clip_threshold
            self.clipped_samples += int(np.count_nonzero(clipped))
            hits = np.flatnonzero(np.any(clipped, axis=1)) + self.frame_offset
            if len(hits):
                first_is_start = (self.last_clip_hit is None or
                                  hits[0] - self.last_clip_hit > self.clip_gap)
                starts = hits[np.r_[first_is_start, np.diff(hits) > self.clip_gap]]
                room = max(0, 8 - len(self.clip_times))
                if room:
                    self.clip_times.extend((starts[:room] / self.sr).tolist())
                self.last_clip_hit = int(hits[-1])

        if self.channels == 1:
            mono = block[:, 0].copy()
        else:
            mono = np.mean(block, axis=1, dtype=np.float32)

        if len(self.mono_pending):
            mono_for_windows = np.concatenate((self.mono_pending, mono))
        else:
            mono_for_windows = mono
        n_windows = len(mono_for_windows) // self.window_size
        used = n_windows * self.window_size
        if n_windows:
            windowed = mono_for_windows[:used].reshape(n_windows, self.window_size)
            power = np.einsum('ij,ij->i', windowed, windowed, optimize=True)
            power = power / self.window_size
            self.window_rms_db.append(20 * np.log10(np.sqrt(power) + 1e-10))
        self.mono_pending = mono_for_windows[used:].copy()
        self.frame_offset += len(block)
        return mono

    def finish(self) -> DynamicsResult:
        rms = (np.sqrt(self.sum_squares / self.sample_values)
               if self.sample_values else 0.0)
        peak_db = float(20 * np.log10(self.peak + 1e-10))
        rms_db = float(20 * np.log10(rms + 1e-10))
        crest_factor = peak_db - rms_db

        if self.window_rms_db:
            levels = np.concatenate(self.window_rms_db)
            dynamic_range = float(np.percentile(levels, 95) -
                                  np.percentile(levels, 5))
        else:
            dynamic_range = 0.0

        clip_percentage = ((self.clipped_samples / self.sample_values) * 100
                           if self.sample_values else 0.0)
        return DynamicsResult(
            peak_db=peak_db,
            rms_db=rms_db,
            crest_factor=crest_factor,
            dynamic_range=dynamic_range,
            dr_rating=_dr_rating(crest_factor),
            clipped_samples=self.clipped_samples,
            clip_percentage=clip_percentage,
            clip_times=self.clip_times,
        )


def analyze_dynamics_file(file_path: str):
    """Return exact dynamics and stream metadata without loading the whole file."""
    import soundfile as sf

    try:
        with sf.SoundFile(file_path) as audio:
            sr = int(audio.samplerate)
            channels = int(audio.channels)
            frame_count = len(audio)
            accumulator = DynamicsAccumulator(sr, channels)
            windows_per_block = max(1, 1_048_576 // accumulator.window_size)
            blocksize = windows_per_block * accumulator.window_size
            for block in audio.blocks(blocksize=blocksize, dtype='float32',
                                      always_2d=True):
                accumulator.update(block)
        return accumulator.finish(), sr, frame_count, channels
    except (sf.LibsndfileError, OSError, RuntimeError):
        from .audio import load_audio
        data, sr = load_audio(file_path)
        channels = data.shape[1] if data.ndim > 1 else 1
        return analyze_dynamics(data, sr), sr, len(data), channels
