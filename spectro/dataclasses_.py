from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class Thresholds:
    # windows are 50% overlapped, see stft_params
    detect_nperseg: int = 8192
    detect_max_seconds: float = 150.0
    display_nperseg: int = 1024
    display_overlap: float = 0.5

    floor_margin_db: float = 18.0
    frame_mean_margin: float = 12.0
    high_band_margin: float = 6.0
    high_band_low_hz: float = 8000.0
    high_band_high_hz: float = 16000.0

    cutoff_search_start_hz: float = 12000.0
    nyquist_search_end: float = 0.995
    nyquist_high_band_cap: float = 0.90
    nyquist_near_low: float = 0.90
    nyquist_near_high: float = 0.99
    nyquist_persistence_default: float = 0.94
    nyquist_suspicious_cap: float = 0.92
    nyquist_cutoff_cand_cap: float = 0.97

    grad_threshold_offset: float = 1.5
    timewin_local_drop: float = -3.0

    drop_search_local_threshold: float = 10.0

    hard_cutoff_drop_db: float = 10.0
    # a shallow drop still counts as a shelf when the wall next to it is steep, quiet top ends (analog masters) never had much to lose in the first place
    soft_cutoff_drop_db: float = 5.0
    hard_cutoff_slope_db_per_khz: float = 32.0
    candidate_cutoffs: tuple = tuple(range(13000, 21501, 250))
    cutoff_drop_width_hz: float = 1000.0
    cutoff_persistence_pad_hz: float = 500.0
    cutoff_refine_window_hz: float = 750.0

    # shelf steepness, read off the measured slope instead of a gradient walk
    shelf_hard_slope: float = -50.0
    shelf_medium_slope: float = -32.0
    shelf_soft_slope: float = -10.0

    # Shelf depth, how far the spectrum falls across the cutoff
    shelf_depth_floor_db: float = 6.0
    shelf_depth_mid_db: float = 10.0
    shelf_depth_full_db: float = 18.0

    sbr_max_cutoff_hz: float = 16000.0
    sbr_above_energy_margin: float = 15.0
    sbr_band_delta_db: float = 12.0
    sbr_corr_threshold: float = 0.5
    sbr_flatness_ratio: float = 1.3

    ultrasonic_floor_hz: float = 24000.0
    ultrasonic_delta_threshold: float = 20.0
    high_sample_rate_hz: int = 48000

    fail_cutoff_hz: float = 18000.0
    fail_persistence: float = 0.5
    warn_persistence: float = 0.4
    pass_near_nyquist_margin: float = 8.0
    low_high_band_margin: float = 6.0
    low_cutoff_hz: float = 14000.0
    upsample_cutoff_hz: float = 20000.0

    clip_threshold: float = 0.99
    clip_window_sec: float = 0.05

    dpi: int = 150
    detect_dpi: int = 110
    output_fmt: str = "png"

    # Evidence score verdict
    # Codec cutoff anchors, I got these empirically from LAME wiki, https://www.rfc-editor.org/rfc/rfc6716
    # 128/192/256/320 mp3, plus 17500 where aac-128 lands (measured, both the ffmpeg encoder and Apple's)
    codec_cutoffs: tuple = (16000, 17500, 18600, 19700, 20500)
    codec_cutoff_window_hz: float = 2500.0  # ± this from anchor = a lossy match
    codec_ceiling_hz: float = 21000.0       # no codec lowpass lives above this
    # under this the content edge is already below every anchor, nothing to read
    codec_blind_edge_hz: float = 12000.0

    # Slope thresholds, measured on a 60 Hz-smoothed spectrum. genuine masters top out near -30 dB/kHz, codec shelves run -33 to -100.
    slope_smooth_hz: float = 60.0
    slope_natural_threshold: float = 10.0   # gentler than this is just content
    slope_lossy_threshold: float = -25.0
    slope_strong_threshold: float = -45.0
    slope_brickwall_threshold: float = 80.0

    # jitter is read off the top quartile of active frames, the ones that reach the ceiling. quiet frames just tell you how the content moves.
    ceiling_quantile: float = 75.0
    jitter_strong_threshold: float = 100.0
    jitter_lossy_threshold: float = 250.0
    jitter_loose_threshold: float = 500.0
    jitter_none_threshold: float = 800.0

    # High/low band energy ratio. Lossless ~-5 to -15, MP3-128 <= -25
    band_ratio_lossy_dB: float = -20.0
    band_ratio_strong_dB: float = -30.0

    fail_score: float = 70.0
    warn_score: float = 45.0
    quality_required: float = 50.0

    # rolloff-85 variance used to be in here. It measured the same on clean and transcoded files, so its weight went to shelf depth, which actually splits
    w_edge: float = 20.0
    w_slope: float = 25.0
    w_shelf: float = 20.0
    w_jitter: float = 10.0
    w_high_band: float = 10.0
    w_sbr: float = 10.0
    w_persistence: float = 5.0

    # Penalties
    quality_silent_rms_dB: float = -60.0
    quality_narrow_bandwidth_hz: float = 8000.0
    quality_high_noise_floor_dB: float = -65.0


T = Thresholds()


# cutoffs re-measured by encoding known-lossless tracks and reading the drop backout. he_aac is still a published figure, nothing here encodes it.
CODEC_PROFILES = {
    'mp3_128':   {'cutoff': (16000, 16800), 'sbr': False, 'shelf': 'hard'},
    'mp3_192':   {'cutoff': (18300, 19000), 'sbr': False, 'shelf': 'hard'},
    'mp3_256':   {'cutoff': (19300, 20000), 'sbr': False, 'shelf': 'hard'},
    'mp3_320':   {'cutoff': (19800, 20600), 'sbr': False, 'shelf': 'hard'},
    'mp3_v0':    {'cutoff': (20800, 21800), 'sbr': False, 'shelf': 'medium'},
    'aac_128':   {'cutoff': (16800, 17800), 'sbr': False, 'shelf': 'hard'},
    'aac_192':   {'cutoff': (19200, 19900), 'sbr': False, 'shelf': 'hard'},
    'aac_256':   {'cutoff': (20900, 21800), 'sbr': False, 'shelf': 'soft'},
    'he_aac':    {'cutoff': (13000, 15000), 'sbr': True,  'shelf': 'soft'},
    'vorbis_128':{'cutoff': (17000, 18900), 'sbr': False, 'shelf': 'hard'},
    'opus_128':  {'cutoff': (19700, 20300), 'sbr': False, 'shelf': 'hard'},
}

DR_THRESHOLDS = {
    'brickwalled': 6,
    'compressed': 9,
    'moderate': 14,
}


@dataclass
class DynamicsResult:
    peak_db: float
    rms_db: float
    crest_factor: float
    dynamic_range: float
    dr_rating: str
    clipped_samples: int
    clip_percentage: float
    clip_times: List[float]


@dataclass
class ChannelAnalysis:
    cutoff_freq: float
    shelf_type: str
    shelf_depth_db: float
    slope_db_per_khz: float
    sbr_likelihood: str
    frequencies: np.ndarray
    times: np.ndarray
    Sxx_db: np.ndarray
    avg_db: np.ndarray
    Zxx_full: np.ndarray
    times_full: np.ndarray
    hop: int


@dataclass
class EvidenceFlag:
    severity: str 
    name: str
    detail: str


@dataclass
class TranscodeEvidence:
    verdict: str
    explanation: str
    cutoff_freq: float
    cutoff_persistence: float
    hard_cutoff: bool
    high_band_db: float
    near_nyquist_db: float
    noise_floor: float
    suspicious_flags: List[EvidenceFlag]
    edge_p10: float
    edge_p50: float
    edge_p90: float
    best_drop_freq: float
    max_drop_db: float
    sbr_likelihood: str
    active_frames_pct: float
    suspicious_windows: List[str]
    edge_p97: float = 0.0
    max_slope_db_per_khz: float = 0.0
    edge_jitter_hz: float = 0.0
    rolloff_85_var_hz: float = 0.0
    band_ratio_db: float = 0.0
    lossy_score: float = 0.0
    quality_score: float = 100.0
    subscores: Optional[Dict[str, float]] = None


@dataclass
class SpectralAnalysisResult:
    profile: str
    cutoff_freq: float
    shelf_type: str
    sbr_likelihood: str
    transcode_warning: Optional[str]
    ultrasonic_energy: Optional[float]
    ultrasonic_delta: Optional[float]
    noise_floor: float
    scores: Dict[str, int]
    frequencies: np.ndarray
    times: np.ndarray
    Sxx_db: np.ndarray
    avg_spectrum_db: np.ndarray
    nyquist: float
    evidence: TranscodeEvidence
    edge_times: np.ndarray
    edge_values: np.ndarray
