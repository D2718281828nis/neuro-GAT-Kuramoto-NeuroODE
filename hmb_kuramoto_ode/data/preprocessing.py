"""Leakage-safe windowing and interpretable rhythm features."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from ..contracts import BANDS

DEFAULT_BANDS = {"delta": (1., 4.), "theta": (4., 8.), "alpha": (8., 13.), "beta": (13., 30.), "gamma": (30., 45.)}

@dataclass
class RhythmPreprocessor:
    sfreq: float = 128.0
    window_seconds: float = 4.0
    overlap: float = 0.5
    bands: dict[str, tuple[float, float]] | None = None

    def windows(self, raw: np.ndarray) -> list[tuple[int, np.ndarray]]:
        if raw.ndim != 2: raise ValueError("raw recording must be [channels, samples]")
        size = int(self.sfreq * self.window_seconds); hop = max(1, int(size * (1-self.overlap)))
        return [(s, raw[:, s:s+size]) for s in range(0, raw.shape[1]-size+1, hop)]

    def transform_window(self, x: np.ndarray) -> np.ndarray:
        """Return [channels,5,6]: log/relative power, amplitude, phase sin/cos, entropy."""
        bands = self.bands or DEFAULT_BANDS
        freq, psd = welch(x, fs=self.sfreq, axis=-1, nperseg=min(x.shape[-1], 256))
        total = np.trapezoid(psd[:, (freq >= 1) & (freq <= 45)], freq[(freq >= 1) & (freq <= 45)], axis=-1) + 1e-12
        out = []
        for name in BANDS:
            lo, hi = bands[name]; select = (freq >= lo) & (freq < hi)
            power = np.trapezoid(psd[:, select], freq[select], axis=-1) + 1e-12
            sos = butter(3, (lo, min(hi, self.sfreq/2-.1)), btype="bandpass", fs=self.sfreq, output="sos")
            analytic = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)
            phase = np.angle(np.mean(analytic, axis=-1)); amp = np.mean(np.abs(analytic), axis=-1)
            q = psd[:, select] / (psd[:, select].sum(-1, keepdims=True)+1e-12)
            entropy = -(q*np.log(q+1e-12)).sum(-1) / np.log(max(2, q.shape[-1]))
            out.append(np.stack([np.log(power), power/total, amp, np.sin(phase), np.cos(phase), entropy], -1))
        return np.stack(out, axis=1).astype("float32")

class TrainNormalizer:
    """Normalizer whose fitted subject IDs are auditable."""
    def fit(self, x: np.ndarray, subject_ids: list[str]) -> "TrainNormalizer":
        if not subject_ids: raise ValueError("training subject IDs cannot be empty")
        self.mean = x.mean(axis=tuple(range(x.ndim-1)), keepdims=True)
        self.std = x.std(axis=tuple(range(x.ndim-1)), keepdims=True) + 1e-6
        self.fitted_subjects = frozenset(subject_ids); return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean"): raise RuntimeError("fit on training subjects before transform")
        return (x-self.mean)/self.std

