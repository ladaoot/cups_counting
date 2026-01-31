from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd



class AnalyticsEngine:
    """Движок для аналитики и визуализации"""

    def __init__(self, history_manager):
        self.history_manager = history_manager

    def get_basic_stats(self) -> Dict:
        """Возвращает базовую статистику"""
        rows = self.history_manager.get_all_records()
        if not rows:
            return {
                "total_requests": 0,
                "total_objects": 0,
                "avg_inference_time": 0.0,
                "active_days": 0
            }

        df = pd.DataFrame(rows)

        # Основные метрики
        total_objects = df["total_count"].fillna(0).sum()
        avg_time = df["inference_ms"].fillna(0).mean()

        # Уникальные дни
        if "ts_iso" in df.columns:
            df["ts_iso"] = pd.to_datetime(df["ts_iso"], errors="coerce")
            unique_dates = df["ts_iso"].dt.date.nunique()
        else:
            unique_dates = 0

        return {
            "total_requests": len(df),
            "total_objects": int(total_objects),
            "avg_inference_time": float(avg_time),
            "active_days": unique_dates
        }

    def get_kind_distribution(self) -> pd.DataFrame:
        """Распределение по типам запросов"""
        rows = self.history_manager.get_all_records()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "kind" not in df.columns:
            return pd.DataFrame()

        return df["kind"].value_counts().reset_index()

    def get_hourly_stats(self) -> pd.DataFrame:
        """Статистика по часам"""
        rows = self.history_manager.get_all_records()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "ts_iso" not in df.columns:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["ts_iso"], errors="coerce")
        df["hour"] = df["datetime"].dt.hour

        hourly_stats = df.groupby("hour").agg({
            "total_count": "sum",
            "ts_iso": "count"
        }).rename(columns={"ts_iso": "requests"})

        return hourly_stats.reset_index()

    def get_video_analytics(self) -> Dict:
        """Аналитика по видео"""
        rows = self.history_manager.get_all_records()
        if not rows:
            return {"video_count": 0, "data": pd.DataFrame()}

        df = pd.DataFrame(rows)
        video_df = df[df["kind"] == "video"]

        return {
            "video_count": len(video_df),
            "data": video_df
        }

    def get_frame_analysis(self, video_name: str, frames_data_path: Path) -> Optional[Dict]:
        """Анализ данных по кадрам видео"""
        try:
            if not frames_data_path.exists():
                return None

            with frames_data_path.open("r", encoding="utf-8") as f:
                frames_data = json.load(f)

            frame_counts = frames_data.get("frame_counts", [])
            if not frame_counts:
                return None

            # Статистика по кадрам
            series = pd.Series(frame_counts)

            return {
                "video_name": video_name,
                "total_frames": len(frame_counts),
                "fps": frames_data.get("fps", 0),
                "frame_stats": {
                    "mean": float(series.mean()),
                    "max": int(series.max()),
                    "min": int(series.min()),
                    "std": float(series.std())
                },
                "frame_counts": frame_counts
            }
        except Exception:
            return None