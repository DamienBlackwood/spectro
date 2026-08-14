from ..dataclasses_ import DynamicsResult


def cmd_info(dynamics: DynamicsResult, sr: int, duration: float, channels: int) -> None:
    print("\n--- DYNAMICS ANALYSIS ---")
    print(f"  Format:          {sr} Hz, {channels} ch, {duration:.1f}s")
    print(f"  Peak level:      {dynamics.peak_db:.1f} dB")
    print(f"  RMS level:       {dynamics.rms_db:.1f} dB")
    print(f"  Crest factor:    {dynamics.crest_factor:.1f} dB")
    print(f"  Dynamic range:   {dynamics.dynamic_range:.1f} dB")
    print(f"  Rating:          {dynamics.dr_rating.upper()}")
    print(f"  Clipped samples: {dynamics.clipped_samples:,} ({dynamics.clip_percentage:.4f}%)")
    if dynamics.clip_times:
        times_str = ", ".join(f"{t:.2f}s" for t in dynamics.clip_times)
        print(f"  Clips start at:  {times_str}")
