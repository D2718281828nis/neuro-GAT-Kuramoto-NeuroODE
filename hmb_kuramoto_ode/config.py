from pathlib import Path
import yaml
def load_config(path):
    with Path(path).open(encoding="utf8") as f: return yaml.safe_load(f)
