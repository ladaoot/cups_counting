# src/history_manager.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.history import HistoryRecord, append_history, read_history, clear_history


class HistoryManager:
    """Менеджер для работы с историей"""

    def __init__(self, history_path: str | Path = "data/history.jsonl"):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def add_image_record(self,
                         input_name: str,
                         model_name: str,
                         total_count: int,
                         per_class_count: Dict[str, int],
                         inference_ms: float,
                         target_classes: Optional[List[str]] = None,
                         conf: float = 0.35,
                         iou: float = 0.45,
                         image_width: Optional[int] = None,
                         image_height: Optional[int] = None) -> None:
        """Добавляет запись об обработке изображения"""
        record = HistoryRecord(
            ts_iso=self._utc_now_iso(),
            kind="image",
            input_name=input_name,
            model_name=model_name,
            target_classes=target_classes or ["cup"],
            conf=conf,
            iou=iou,
            total_count=total_count,
            per_class_count=per_class_count,
            inference_ms=inference_ms,
            image_width=image_width,
            image_height=image_height,
        )
        append_history(record, self.history_path)

    def add_video_record(self,
                         input_name: str,
                         model_name: str,
                         total_count: int,
                         per_class_count: Dict[str, int],
                         inference_ms: float,
                         output_artifact: str,
                         max_per_frame: int,
                         target_classes: Optional[List[str]] = None,
                         conf: float = 0.35,
                         iou: float = 0.45,
                         image_width: Optional[int] = None,
                         image_height: Optional[int] = None) -> None:
        """Добавляет запись об обработке видео"""
        record = HistoryRecord(
            ts_iso=self._utc_now_iso(),
            kind="video",
            input_name=input_name,
            model_name=model_name,
            target_classes=target_classes or ["cup"],
            conf=conf,
            iou=iou,
            total_count=total_count,
            per_class_count=per_class_count,
            inference_ms=inference_ms,
            image_width=image_width,
            image_height=image_height,
            output_artifact=output_artifact,
            max_per_frame=max_per_frame,
        )
        append_history(record, self.history_path)



    def add_camera_record(self,
                          input_name: str,
                          model_name: str,
                          total_count: int,
                          per_class_count: Dict[str, int],
                          inference_ms: float,
                          target_classes: Optional[List[str]] = None,
                          conf: float = 0.35,
                          iou: float = 0.45) -> None:
        """Добавляет запись с камеры"""
        record = HistoryRecord(
            ts_iso=self._utc_now_iso(),
            kind="camera",
            input_name=input_name,
            model_name=model_name,
            target_classes=target_classes or ["cup"],
            conf=conf,
            iou=iou,
            total_count=total_count,
            per_class_count=per_class_count,
            inference_ms=inference_ms,
        )
        append_history(record, self.history_path)

    def get_all_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Получает все записи истории"""
        return read_history(self.history_path, limit)

    def clear_history(self) -> None:
        """Очищает историю"""
        clear_history(self.history_path)

    def _utc_now_iso(self) -> str:
        """Текущее время в ISO формате"""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()