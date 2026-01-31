from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional

from ultralytics import YOLO


@dataclass
class CameraState:
    """Состояние камеры для онлайн-подсчета"""
    last_count: int = 0
    last_ms: float = 0.0
    last_per_class: Dict[str, int] = field(default_factory=dict)
    last_save: float = 0.0
    save_count: int = 0
    lock: Lock = field(default_factory=Lock)
    is_running: bool = False


class CameraProcessor:
    """Процессор для обработки потока с камеры"""

    def __init__(self, model_path: str = "yolov8s.pt",
                 target_classes: Optional[list] = None,
                 conf: float = 0.35,
                 iou: float = 0.45):
        self.model = YOLO(model_path)
        self.target_classes = target_classes or ["cup"]
        self.conf = conf
        self.iou = iou

        # Получаем ID целевых классов
        self.class_ids = self._get_class_ids()
        self.state = CameraState()

    def _get_class_ids(self) -> list:
        """Получаем ID целевых классов"""
        names = getattr(self.model, "names", {}) or {}
        want = {c.strip().lower() for c in self.target_classes if c.strip()}
        out = []
        for cid, cname in names.items():
            if str(cname).lower() in want:
                out.append(int(cid))
        return out

    def process_frame(self, frame_bgr) -> tuple:
        """
        Обрабатывает один кадр и возвращает результат
        Returns: (annotated_frame, count, inference_time, per_class_count)
        """
        import time as timer

        t0 = timer.perf_counter()

        # Детекция объектов
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_ids if self.class_ids else None,
            verbose=False,
            max_det=300,
        )

        r0 = results[0]
        boxes = r0.boxes

        # Подсчет объектов
        names = getattr(self.model, "names", {})
        per_class_count: Dict[str, int] = {}
        cnt = 0

        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            for cid in cls_ids:
                cname = names.get(int(cid), str(int(cid)))
                # Считаем только целевые классы
                if self.class_ids and int(cid) in self.class_ids:
                    per_class_count[cname] = per_class_count.get(cname, 0) + 1
                    cnt += 1

        dt_ms = (timer.perf_counter() - t0) * 1000.0

        # Аннотированный кадр
        annotated_frame = r0.plot()

        # Обновляем состояние
        with self.state.lock:
            self.state.last_count = cnt
            self.state.last_ms = float(dt_ms)
            self.state.last_per_class = per_class_count.copy()

        return annotated_frame, cnt, dt_ms, per_class_count

    def get_current_stats(self) -> tuple:
        """Возвращает текущую статистику"""
        with self.state.lock:
            return (
                self.state.last_count,
                self.state.last_ms,
                self.state.last_per_class.copy(),
                self.state.save_count
            )

    def start(self):
        """Запускает процессор"""
        self.state.is_running = True

    def stop(self):
        """Останавливает процессор"""
        self.state.is_running = False


# Singleton для глобального доступа к процессору камеры
_camera_processor: Optional[CameraProcessor] = None


def get_camera_processor(model_path: str = "yolov8s.pt",
                         target_classes: Optional[list] = None,
                         conf: float = 0.35,
                         iou: float = 0.45) -> CameraProcessor:
    """Получает или создает глобальный процессор камеры"""
    global _camera_processor

    if _camera_processor is None:
        _camera_processor = CameraProcessor(
            model_path=model_path,
            target_classes=target_classes,
            conf=conf,
            iou=iou
        )
    return _camera_processor


def reset_camera_processor():
    """Сбрасывает глобальный процессор камеры"""
    global _camera_processor
    _camera_processor = None