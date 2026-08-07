"""STEW text-recording discovery. Dataset files are never copied into outputs."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from .preprocessing import RhythmPreprocessor
from ..contracts import DEFAULT_CHANNELS

def discover_stew(data_root: str | Path) -> list[Path]:
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"STEW data root not found: {root}. Expected <root>/*.txt recordings such as sub01_hi.txt and sub01_lo.txt (14 EEG columns). Set data.data_root or --data-root; synthetic data is not substituted.")
    files = sorted(p for p in root.rglob("*.txt") if not p.name.startswith("."))
    if not files: raise FileNotFoundError(f"No STEW .txt recordings below {root}; expected subject/condition text files with 14 EEG columns.")
    return files

@dataclass
class STEWRecord:
    path: Path; subject_id: str; condition: str; label: int

class STEWDataset:
    def __init__(self, data_root: str | Path, preprocessor: RhythmPreprocessor | None = None):
        self.preprocessor = preprocessor or RhythmPreprocessor(); self.records=[]
        for p in discover_stew(data_root):
            stem=p.stem.lower(); m=re.search(r"(\d+)", stem); subject=m.group(1) if m else p.stem
            condition="high" if any(k in stem for k in ("hi", "high")) else "low"
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
