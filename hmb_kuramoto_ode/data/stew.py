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
        x=np.loadtxt(record.path, dtype="float32"); x=x.T if x.shape[1] == 14 else x
        if x.shape[0] != 14: raise ValueError(f"{record.path}: expected 14 channels, got {x.shape}")
        return x

