from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_submission(sample_ids, pred: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"sample number": sample_ids, "pred": pred})
    sub.to_csv(path, index=False, header=False, encoding="utf-8-sig")
