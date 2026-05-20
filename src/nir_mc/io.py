from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "features": root / "features",
        "models": root / "models",
        "results": root / "results",
        "figures": root / "figures",
        "submissions": root / "submissions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
