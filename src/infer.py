from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DetectionSummary:
    model_name: str
    selected_class_ids: List[int]
    selected_class_names: List[str]
    total_count: int
    per_class_count: Dict[str, int]
    conf_threshold: float
    iou_threshold: float
    image_width: int
    image_height: int
    inference_ms: float
    # For video we may additionally compute max count per frame.
    max_per_frame: Optional[int] = None


def _lazy_import_ultralytics():
    # Streamlit reloads; keep import local to reduce startup issues.
    from ultralytics import YOLO  # type: ignore

    return YOLO


@lru_cache(maxsize=3)
def load_model(model_path: str = "yolov8n.pt"):
    YOLO = _lazy_import_ultralytics()
    return YOLO(model_path)


def _ensure_bgr_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """
    Streamlit/PIL often provides RGB uint8.
    Ultralytics accepts numpy arrays; we keep BGR for drawing with OpenCV later.
    """
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB image")
    # RGB -> BGR
    return image_rgb[:, :, ::-1].copy()


def _get_selected_classes(names: Dict[int, str], want: Iterable[str]) -> Tuple[List[int], List[str]]:
    want_lower = {w.strip().lower() for w in want if w.strip()}
    selected_ids: List[int] = []
    selected_names: List[str] = []
    for cls_id, cls_name in names.items():
        if cls_name.lower() in want_lower:
            selected_ids.append(int(cls_id))
            selected_names.append(cls_name)
    return selected_ids, selected_names


def detect_and_count_on_image(
    image_rgb: np.ndarray,
    *,
    model_path: str = "yolov8n.pt",
    target_classes: Optional[List[str]] = None,
    conf: float = 0.35,
    iou: float = 0.45,
    max_det: int = 300,
) -> Tuple[np.ndarray, DetectionSummary]:
    """
    Returns: (annotated_image_rgb, summary)
    """
    import time

    if target_classes is None:
        # For coffee shop: "cup" is the main COCO class.
        target_classes = ["cup"]

    model = load_model(model_path)

    bgr = _ensure_bgr_uint8(image_rgb)
    h, w = bgr.shape[:2]

    names: Dict[int, str] = getattr(model, "names", {})
    selected_ids, selected_names = _get_selected_classes(names, target_classes)

    t0 = time.perf_counter()
    results = model.predict(
        source=bgr,
        conf=conf,
        iou=iou,
        classes=selected_ids if selected_ids else None,
        verbose=False,
        max_det=max_det,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    r0 = results[0]
    # r0.boxes: xyxy, cls, conf
    boxes = r0.boxes

    per_class: Dict[str, int] = {}
    total = 0

    if boxes is not None and len(boxes) > 0:
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        for cid in cls_ids:
            cname = names.get(int(cid), str(int(cid)))
            per_class[cname] = per_class.get(cname, 0) + 1
            total += 1

    # Use built-in plotting (returns BGR)
    plotted_bgr = r0.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1].copy()

    summary = DetectionSummary(
        model_name=model_path,
        selected_class_ids=selected_ids,
        selected_class_names=selected_names if selected_names else target_classes,
        total_count=total,
        per_class_count=per_class,
        conf_threshold=float(conf),
        iou_threshold=float(iou),
        image_width=int(w),
        image_height=int(h),
        inference_ms=float(dt_ms),
    )
    return plotted_rgb, summary


def detect_and_count_on_video(
    video_path: str,
    *,
    model_path: str = "yolov8n.pt",
    target_classes: Optional[List[str]] = None,
    conf: float = 0.35,
    iou: float = 0.45,
    max_det: int = 300,
    sample_every_n_frames: int = 1,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[str, DetectionSummary, str, List[int]]:  # Добавили возврат списка кадров
    """
    Обрабатывает видео файл
    Returns: (output_video_path, summary, frames_data_path, frame_counts)
    """
    import time
    from pathlib import Path

    import cv2  # type: ignore

    if target_classes is None:
        target_classes = ["cup"]

    model = load_model(model_path)
    names: Dict[int, str] = getattr(model, "names", {})
    selected_ids, selected_names = _get_selected_classes(names, target_classes)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    in_path = Path(video_path)
    out_path = str(in_path.with_name(in_path.stem + "_annotated.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise ValueError("Cannot create video writer (mp4v). Try installing proper codecs.")

    total_frames = 0
    processed_frames = 0
    total_count_sum = 0
    max_count = 0
    per_class_sum: Dict[str, int] = {}
    frame_counts: List[int] = []  # Сохраняем количество объектов на каждом кадре

    t0 = time.perf_counter()
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        total_frames += 1
        frame_idx += 1

        # Обновляем прогресс
        if progress_callback and total_frames_estimate > 0:
            progress = min(frame_idx / total_frames_estimate, 1.0)
            progress_callback(progress)

        if sample_every_n_frames > 1 and (frame_idx % sample_every_n_frames != 0):
            writer.write(frame_bgr)
            continue

        results = model.predict(
            source=frame_bgr,
            conf=conf,
            iou=iou,
            classes=selected_ids if selected_ids else None,
            verbose=False,
            max_det=max_det,
        )
        r0 = results[0]
        boxes = r0.boxes

        cnt = 0
        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            for cid in cls_ids:
                cname = names.get(int(cid), str(int(cid)))
                per_class_sum[cname] = per_class_sum.get(cname, 0) + 1
                cnt += 1

        total_count_sum += cnt
        max_count = max(max_count, cnt)
        processed_frames += 1
        frame_counts.append(cnt)  # Сохраняем количество на этом кадре

        plotted = r0.plot()
        writer.write(plotted)
    
    if progress_callback:
        progress_callback(1.0)

    dt_ms = (time.perf_counter() - t0) * 1000.0

    cap.release()
    writer.release()

    avg_per_frame = int(round(total_count_sum / max(processed_frames, 1)))
    per_class_avg: Dict[str, int] = {
        k: int(round(v / max(processed_frames, 1))) for k, v in per_class_sum.items()
    }

    # Сохраняем данные по кадрам в JSON файл
    frames_data_path = str(in_path.with_name(in_path.stem + "_frames_data.json"))
    try:
        import json
        frames_data = {
            "video_path": video_path,
            "total_frames": processed_frames,
            "fps": float(fps),
            "frame_counts": frame_counts,
            "sample_every_n_frames": sample_every_n_frames
        }
        with open(frames_data_path, "w", encoding="utf-8") as f:
            json.dump(frames_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Не удалось сохранить данные по кадрам: {e}")

    summary = DetectionSummary(
        model_name=model_path,
        selected_class_ids=selected_ids,
        selected_class_names=selected_names if selected_names else target_classes,
        total_count=avg_per_frame,
        per_class_count=per_class_avg,
        conf_threshold=float(conf),
        iou_threshold=float(iou),
        image_width=int(w),
        image_height=int(h),
        inference_ms=float(dt_ms),
        max_per_frame=int(max_count),
    )

    # Возвращаем данные
    return out_path, summary, frames_data_path, frame_counts

