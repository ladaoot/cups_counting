from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HistoryRecord:
    ts_iso: str
    kind: str  # image | video | camera
    input_name: str
    model_name: str
    target_classes: List[str]
    conf: float
    iou: float
    total_count: int
    per_class_count: Dict[str, int]
    inference_ms: float
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    output_artifact: Optional[str] = None  # annotated video path, etc.
    max_per_frame: Optional[int] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_data_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def append_history(record: HistoryRecord, history_path: str | Path) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_history(history_path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(history_path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupted line
                continue
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


def clear_history(history_path: str | Path) -> None:
    path = Path(history_path)
    if path.exists():
        path.unlink()

