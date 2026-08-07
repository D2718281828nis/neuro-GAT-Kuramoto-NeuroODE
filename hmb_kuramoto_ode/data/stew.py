"""STEW text-recording discovery. Dataset files are never copied into outputs."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from .preprocessing import RhythmPreprocessor
from ..contracts import DEFAULT_CHANNELS


def condition_from_path(path: Path) -> str | None:
    """Return the explicit STEW condition encoded in a recording filename.

    Metadata tables such as ``ratings.txt`` intentionally return ``None`` and
    are not EEG recordings.  Separators and compact suffixes (``sub01_hi`` and
    ``sub01hi``) are both supported.
    """
    stem = path.stem.lower()
    match = re.search(r"(?:^|[_\-\s])(high|hi|low|lo)(?:$|[_\-\s])", stem)
    marker = match.group(1) if match else next(
        (suffix for suffix in ("high", "low", "hi", "lo") if stem.endswith(suffix)),
        None,
    )
    if marker is None:
        return None
    return "high" if marker in {"high", "hi"} else "low"


def discover_stew(data_root: str | Path) -> list[Path]:
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"STEW data root not found: {root}. Expected <root>/*.txt recordings such as sub01_hi.txt and sub01_lo.txt (14 EEG columns). Set data.data_root or --data-root; synthetic data is not substituted.")
    text_files = sorted(p for p in root.rglob("*.txt") if not p.name.startswith("."))
    files = [path for path in text_files if condition_from_path(path) is not None]
    if not files:
        ignored = ", ".join(path.name for path in text_files[:5]) or "none"
        raise FileNotFoundError(
            f"No STEW high/low EEG recordings below {root}. Expected names such "
            f"as sub01_hi.txt and sub01_lo.txt; ignored metadata/unrecognized "
            f"text files: {ignored}."
        )
    return files

@dataclass
class STEWRecord:
    path: Path; subject_id: str; condition: str; label: int

class STEWDataset:
    def __init__(self, data_root: str | Path, preprocessor: RhythmPreprocessor | None = None):
        self.preprocessor = preprocessor or RhythmPreprocessor(); self.records=[]
        for p in discover_stew(data_root):
            stem=p.stem.lower(); m=re.search(r"(\d+)", stem); subject=m.group(1) if m else p.stem
            condition=condition_from_path(p)
            assert condition is not None  # discover_stew already enforces this contract.
            self.records.append(STEWRecord(p, subject, condition, int(condition == "high")))
    def inspect(self) -> dict:
        return {"recordings":len(self.records), "subjects":len({r.subject_id for r in self.records}), "channels":list(DEFAULT_CHANNELS), "sampling_frequency":self.preprocessor.sfreq}

    def load(self, record: STEWRecord) -> np.ndarray:
        """Load a STEW recording as ``[14, samples]``.

        STEW mirrors in the wild use both whitespace-separated text and CSV
        (sometimes with a header or a leading sample/time column).  Delimiter
        detection is based on the first non-empty line; channel validation
        remains strict so malformed metadata cannot silently become EEG data.
        """
        lines = record.path.read_text(encoding="utf-8-sig").splitlines()
        first = next((line for line in lines if line.strip()), "")
        if not first:
            raise ValueError(f"{record.path}: recording is empty")
        counts = {delimiter: first.count(delimiter) for delimiter in (",", ";", "\t")}
        delimiter = max(counts, key=counts.get) if max(counts.values()) else None
        try:
            values = np.genfromtxt(
                record.path,
                delimiter=delimiter,
                dtype=np.float32,
                encoding="utf-8-sig",
                invalid_raise=True,
            )
        except (OSError, TypeError, ValueError) as error:
            shown = "whitespace" if delimiter is None else repr(delimiter)
            raise ValueError(
                f"{record.path}: could not parse numeric STEW samples using "
                f"detected delimiter {shown}. Expected 14 numeric EEG columns, "
                "optionally preceded by one sample/time column."
            ) from error

        values = np.atleast_2d(values)
        # genfromtxt represents a textual header as one all-NaN row/column.
        values = values[~np.isnan(values).all(axis=1)]
        values = values[:, ~np.isnan(values).all(axis=0)]
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"{record.path}: recording contains missing or non-numeric sample values")

        def index_like(column: np.ndarray) -> bool:
            differences = np.diff(column.astype(np.float64))
            return bool(differences.size and np.all(differences >= 0) and np.any(differences > 0))

        if values.shape[1] == 15 and index_like(values[:, 0]):
            values = values[:, 1:]
        elif values.shape[0] == 15 and index_like(values[0]):
            values = values[1:, :]

        if values.shape[1] == 14:
            values = values.T
        elif values.shape[0] != 14:
            shown = "whitespace" if delimiter is None else repr(delimiter)
            raise ValueError(
                f"{record.path}: expected 14 EEG channels after parsing delimiter "
                f"{shown}, got numeric shape {values.shape}"
            )
        return np.ascontiguousarray(values, dtype=np.float32)
