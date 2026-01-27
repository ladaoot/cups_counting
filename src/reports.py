from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def history_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "ts_iso",
                "kind",
                "input_name",
                "model_name",
                "target_classes",
                "conf",
                "iou",
                "total_count",
                "per_class_count",
                "max_per_frame",
                "inference_ms",
                "image_width",
                "image_height",
                "output_artifact",
            ]
        )
    df = pd.DataFrame(rows)
    # Keep stable column order
    ordered_cols = [
        "ts_iso",
        "kind",
        "input_name",
        "model_name",
        "target_classes",
        "conf",
        "iou",
        "total_count",
        "per_class_count",
        "max_per_frame",
        "inference_ms",
        "image_width",
        "image_height",
        "output_artifact",
    ]
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = None
    return df[ordered_cols]


def make_excel_bytes(rows: List[Dict[str, Any]]) -> bytes:
    df = history_to_dataframe(rows)
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="history")
        # basic autosize
        ws = writer.sheets["history"]
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                if cell.value is None:
                    continue
                max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max(10, max_len + 2), 60)
    return bio.getvalue()


def _register_cyrillic_font_if_possible() -> str:
    """
    ReportLab built-ins don't fully support Cyrillic. We try to register a common Windows font.
    If not found, fall back to Helvetica (Cyrillic may render as squares).
    """
    import os
    from pathlib import Path

    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "times.ttf",
    ]
    for p in candidates:
        if p.exists():
            font_name = "CyrillicFont"
            pdfmetrics.registerFont(TTFont(font_name, str(p)))
            return font_name
    return "Helvetica"


def make_pdf_bytes(rows: List[Dict[str, Any]]) -> bytes:
    df = history_to_dataframe(rows)
    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)

    font_name = _register_cyrillic_font_if_possible()
    c.setFont(font_name, 12)

    width, height = A4
    x0 = 15 * mm
    y = height - 20 * mm

    c.drawString(x0, y, "Отчёт по истории запросов (детекция стаканов/кружек)")
    y -= 10 * mm
    c.setFont(font_name, 10)

    if df.empty:
        c.drawString(x0, y, "История пуста.")
        c.showPage()
        c.save()
        return bio.getvalue()

    total_requests = len(df)
    avg_time = float(df["inference_ms"].fillna(0).mean()) if "inference_ms" in df.columns else 0.0
    avg_count = float(df["total_count"].fillna(0).mean()) if "total_count" in df.columns else 0.0
    c.drawString(x0, y, f"Всего запросов: {total_requests}")
    y -= 6 * mm
    c.drawString(x0, y, f"Среднее время инференса: {avg_time:.1f} ms")
    y -= 6 * mm
    c.drawString(x0, y, f"Средний подсчёт (на изображение / среднее на кадр): {avg_count:.2f}")
    y -= 10 * mm

    # Table header (simplified to fit page)
    headers = ["ts_iso", "kind", "input_name", "total_count", "inference_ms"]
    col_widths = [45 * mm, 18 * mm, 60 * mm, 22 * mm, 25 * mm]
    c.setFont(font_name, 9)
    c.drawString(x0, y, "Последние записи:")
    y -= 6 * mm

    def draw_row(vals: List[str], y_pos: float) -> None:
        x = x0
        for v, w in zip(vals, col_widths):
            c.drawString(x, y_pos, v[: max(1, int(w / 2.5))])
            x += w

    draw_row(headers, y)
    y -= 5 * mm
    c.line(x0, y, x0 + sum(col_widths), y)
    y -= 5 * mm

    tail = df.tail(20)
    for _, row in tail.iterrows():
        if y < 20 * mm:
            c.showPage()
            c.setFont(font_name, 9)
            y = height - 20 * mm
        vals = [
            str(row.get("ts_iso", "")),
            str(row.get("kind", "")),
            str(row.get("input_name", "")),
            str(row.get("total_count", "")),
            str(row.get("inference_ms", "")),
        ]
        draw_row(vals, y)
        y -= 5 * mm

    c.showPage()
    c.save()
    return bio.getvalue()

