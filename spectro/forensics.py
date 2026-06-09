from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def effective_bit_depth(file_path: str, max_frames: int = 5_000_000) -> Optional[Tuple[int, int]]:
    try:
        info = sf.info(file_path)
        if info.subtype not in ("PCM_24", "PCM_32"):
            return None
        data, _ = sf.read(file_path, dtype="int32", frames=max_frames)
    except Exception:
        return None

    x = np.asarray(data).ravel()
    x = x[x != 0]
    if len(x) == 0:
        return None

    # x & -x isolates the lowest set bit, a padded 16-bit master never touches the low bits
    lowest = int(np.min(np.bitwise_and(x, -x).view(np.uint32)))
    effective = 32 - int(np.log2(lowest))
    claimed = 24 if info.subtype == "PCM_24" else 32
    return claimed, effective


def stereo_correlation(data: np.ndarray) -> Optional[float]:
    if data.ndim < 2 or data.shape[1] < 2:
        return None
    a, b = data[::16, 0], data[::16, 1]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])
