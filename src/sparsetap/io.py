import json
from pathlib import Path


def _ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_candidates(candidates, path):
    path = _ensure_parent(path)
    path.write_text(json.dumps(candidates, indent=2))
    return path


def load_candidates(path):
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_results(results, path):
    path = _ensure_parent(path)
    path.write_text(json.dumps(results, indent=2))
    return path
