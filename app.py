from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.history import HistoryRecord, append_history, clear_history, read_history, utc_now_iso
from src.infer import detect_and_count_on_image, detect_and_count_on_video, load_model
from src.reports import make_excel_bytes, make_pdf_bytes


APP_TITLE = "Практика CV: подсчёт стаканов/кружек (YOLOv8)"
DATA_DIR = Path("data")
HISTORY_PATH = DATA_DIR / "history.jsonl"


def _parse_target_classes(text: str) -> List[str]:
    # Accept comma/space separated
    raw = [t.strip() for t in text.replace(";", ",").split(",")]
    return [t for t in raw if t]


def _save_uploaded_to_temp(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getbuffer())
        return f.name


def _get_class_ids(model, class_names: List[str]) -> List[int]:
    names = getattr(model, "names", {}) or {}
    want = {c.strip().lower() for c in class_names if c.strip()}
    out: List[int] = []
    for cid, cname in names.items():
        if str(cname).lower() in want:
            out.append(int(cid))
    return out


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Настройки модели")
    model_path = st.text_input("Модель (веса Ultralytics)", value="yolov8n.pt", help="Например: yolov8n.pt / yolov8s.pt")
    st.info("🔍 Поиск: стаканы и кружки (cup)")
    conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("IoU", 0.05, 0.95, 0.45, 0.05)

    st.subheader("Видео")
    sample_every = st.number_input("Обрабатывать каждый N‑й кадр", min_value=1, max_value=30, value=1, step=1)
    
    st.subheader("Камера")
    auto_save_enabled = st.checkbox("Автосохранение", value=True)
    auto_save_interval = st.number_input("Интервал автосохранения (сек)", min_value=5, max_value=300, value=15, step=5, disabled=not auto_save_enabled)

tabs = st.tabs(["Изображение", "Видео", "Камера", "История и отчёты"])


with tabs[0]:
    st.write("Загрузите одно или несколько изображений и нажмите **Запустить обработку**.")
    up_list = st.file_uploader(
        "Изображения (jpg/png/webp)", 
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    run = st.button("Запустить обработку изображений", type="primary", disabled=not up_list)

    if up_list:
        st.write(f"Загружено изображений: **{len(up_list)}**")
        cols = st.columns(min(3, len(up_list)))
        for idx, up in enumerate(up_list):
            with cols[idx % len(cols)]:
                img = Image.open(up).convert("RGB")
                st.image(img, caption=up.name, use_container_width=True)

    if run and up_list:
        classes = ["cup"]  # Всегда ищем стаканы/кружки
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, up in enumerate(up_list):
            status_text.text(f"Обработка {idx + 1}/{len(up_list)}: {up.name}")
            progress_bar.progress((idx + 1) / len(up_list))
            
            image_rgb = np.array(Image.open(up).convert("RGB"))
            annotated_rgb, summary = detect_and_count_on_image(
                image_rgb,
                model_path=model_path,
                target_classes=classes,
                conf=float(conf),
                iou=float(iou),
            )

            st.subheader(f"Результат: {up.name}")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(annotated_rgb, caption="Аннотированное изображение", use_container_width=True)
            with col2:
                st.metric("Всего объектов", summary.total_count)
                st.metric("Время инференса (ms)", f"{summary.inference_ms:.1f}")
                st.metric("Классы", ", ".join(summary.selected_class_names))
                st.json(summary.per_class_count)

            append_history(
                HistoryRecord(
                    ts_iso=utc_now_iso(),
                    kind="image",
                    input_name=up.name,
                    model_name=summary.model_name,
                    target_classes=classes,
                    conf=summary.conf_threshold,
                    iou=summary.iou_threshold,
                    total_count=summary.total_count,
                    per_class_count=summary.per_class_count,
                    inference_ms=summary.inference_ms,
                    image_width=summary.image_width,
                    image_height=summary.image_height,
                ),
                HISTORY_PATH,
            )
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Обработано {len(up_list)} изображений. Записи добавлены в историю.")


with tabs[1]:
    st.write("Загрузите одно или несколько видео и нажмите **Запустить обработку** (создастся аннотированный mp4).")
    upv_list = st.file_uploader(
        "Видео (mp4/avi/mov)", 
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True
    )
    runv = st.button("Запустить обработку видео", type="primary", disabled=not upv_list)

    if upv_list:
        st.write(f"Загружено видео: **{len(upv_list)}**")
        for upv in upv_list:
            st.text(f"📹 {upv.name} ({upv.size / 1024 / 1024:.2f} MB)")

    if runv and upv_list:
        classes = ["cup"]  # Всегда ищем стаканы/кружки
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, upv in enumerate(upv_list):
            status_text.text(f"Обработка видео {idx + 1}/{len(upv_list)}: {upv.name}... Это может занять некоторое время.")
            progress_bar.progress((idx + 1) / len(upv_list))
            
            tmp_in = _save_uploaded_to_temp(upv)
            
            # Создаём прогресс-бар для этого видео
            video_progress = st.progress(0)
            video_status = st.empty()
            
            def update_video_progress(p: float):
                video_progress.progress(p)
                video_status.text(f"Обработка кадров: {p*100:.1f}%")
            
            with st.spinner(f"Обработка {upv.name}..."):
                result = detect_and_count_on_video(
                    tmp_in,
                    model_path=model_path,
                    target_classes=classes,
                    conf=float(conf),
                    iou=float(iou),
                    sample_every_n_frames=int(sample_every),
                    progress_callback=update_video_progress,
                )
                # Функция всегда возвращает 3 значения: out_path, summary, frames_data_path
                out_path, summary, frames_data_path = result
            
            video_progress.empty()
            video_status.empty()

            st.subheader(f"Результат: {upv.name}")
            st.write(
                f"Подсчёт: **{summary.total_count}** (среднее на обработанный кадр), "
                f"максимум на кадр: **{summary.max_per_frame}**"
            )
            st.write(f"Время обработки (ms): **{summary.inference_ms:.1f}**")

            # Исправляем отображение видео - используем правильный путь
            video_file = Path(out_path)
            if video_file.exists():
                with open(video_file, "rb") as video_file_handler:
                    video_bytes = video_file_handler.read()
                    st.video(video_bytes)
                    st.download_button(
                        f"Скачать аннотированное видео: {video_file.name}", 
                        data=video_bytes, 
                        file_name=video_file.name,
                        key=f"download_{idx}"
                    )
            else:
                st.error(f"Файл не найден: {out_path}")

            # Сохраняем путь к данным по кадрам в истории
            video_record = HistoryRecord(
                ts_iso=utc_now_iso(),
                kind="video",
                input_name=upv.name,
                model_name=summary.model_name,
                target_classes=classes,
                conf=summary.conf_threshold,
                iou=summary.iou_threshold,
                total_count=summary.total_count,
                per_class_count=summary.per_class_count,
                inference_ms=summary.inference_ms,
                image_width=summary.image_width,
                image_height=summary.image_height,
                output_artifact=str(out_path),
                max_per_frame=summary.max_per_frame,
            )
            append_history(video_record, HISTORY_PATH)
            
            # Сохраняем путь к данным по кадрам в отдельном файле для быстрого доступа
            if frames_data_path:
                try:
                    frames_index_path = DATA_DIR / "video_frames_index.jsonl"
                    frames_index_path.parent.mkdir(parents=True, exist_ok=True)
                    with frames_index_path.open("a", encoding="utf-8") as f:
                        import json
                        index_entry = {
                            "video_name": upv.name,
                            "frames_data_path": frames_data_path,
                            "ts_iso": video_record.ts_iso
                        }
                        f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    st.warning(f"Не удалось сохранить индекс данных по кадрам: {e}")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Обработано {len(upv_list)} видео. Записи добавлены в историю.")


with tabs[2]:
    st.write("Нажмите **Start**, чтобы открыть поток. Обработка идёт в реальном времени (может быть медленно на CPU).")

    from threading import Lock
    import time
    from datetime import datetime, timezone

    import av  # type: ignore
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer  # type: ignore

    classes = ["cup"]  # Всегда ищем стаканы/кружки
    model = load_model(model_path)
    class_ids = _get_class_ids(model, classes)
    stats_lock = Lock()
    
    # Инициализация состояния камеры
    if "camera_last" not in st.session_state:
        st.session_state.camera_last = {"count": 0, "ms": 0.0, "per_class": {}}
    if "camera_last_save" not in st.session_state:
        st.session_state.camera_last_save = 0.0
    if "camera_save_count" not in st.session_state:
        st.session_state.camera_save_count = 0

    # Используем общий словарь для синхронизации между потоками
    camera_shared_state = {"last_save": 0.0, "save_count": 0, "last_count": 0, "last_ms": 0.0, "last_per_class": {}}
    
    class YoloVideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_save_time = 0.0
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            t0 = time.perf_counter()
            results = model.predict(
                source=img,
                conf=float(conf),
                iou=float(iou),
                classes=class_ids if class_ids else None,
                verbose=False,
                max_det=300,
            )
            r0 = results[0]
            boxes = r0.boxes
            
            # Правильный подсчёт: считаем только выбранные классы (cup)
            names = getattr(model, "names", {})
            per_class_count: Dict[str, int] = {}
            cnt = 0
            
            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
                for cid in cls_ids:
                    cname = names.get(int(cid), str(int(cid)))
                    # Считаем только если это выбранный класс
                    if class_ids and int(cid) in class_ids:
                        per_class_count[cname] = per_class_count.get(cname, 0) + 1
                        cnt += 1
            
            dt_ms = (time.perf_counter() - t0) * 1000.0

            plotted = r0.plot()
            current_time = time.time()
            
            with stats_lock:
                # Обновляем общее состояние для отображения (работает в отдельном потоке)
                camera_shared_state["last_count"] = cnt
                camera_shared_state["last_ms"] = float(dt_ms)
                camera_shared_state["last_per_class"] = per_class_count.copy()
                
                # Обновляем session_state если доступен
                try:
                    st.session_state.camera_last = {
                        "count": cnt, 
                        "ms": float(dt_ms),
                        "per_class": per_class_count.copy()
                    }
                except:
                    pass  # Если session_state недоступен в потоке, пропускаем
                
                # Автосохранение через общий словарь
                if auto_save_enabled:
                    time_since_last_save = current_time - camera_shared_state["last_save"]
                    if time_since_last_save >= auto_save_interval:
                        per_class_snapshot = per_class_count.copy()
                        if not per_class_snapshot and cnt > 0:
                            per_class_snapshot = {"cup": cnt}
                        
                        try:
                            append_history(
                                HistoryRecord(
                                    ts_iso=utc_now_iso(),
                                    kind="camera",
                                    input_name=f"webcam_auto_{camera_shared_state['save_count']}",
                                    model_name=model_path,
                                    target_classes=classes,
                                    conf=float(conf),
                                    iou=float(iou),
                                    total_count=cnt,
                                    per_class_count=per_class_snapshot,
                                    inference_ms=float(dt_ms),
                                ),
                                HISTORY_PATH,
                            )
                            camera_shared_state["last_save"] = current_time
                            camera_shared_state["save_count"] += 1
                            
                            # Обновляем session_state если доступен
                            try:
                                st.session_state.camera_last_save = current_time
                                st.session_state.camera_save_count = camera_shared_state["save_count"]
                            except:
                                pass
                        except Exception as e:
                            # Логируем ошибку, но не прерываем обработку
                            print(f"Ошибка автосохранения: {e}")
            
            return av.VideoFrame.from_ndarray(plotted, format="bgr24")

    ctx = webrtc_streamer(
        key="camera",
        video_transformer_factory=YoloVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Используем общее состояние для отображения (работает в основном потоке)
    with stats_lock:
        current_count = camera_shared_state["last_count"]
        current_ms = camera_shared_state["last_ms"]
        current_per_class = camera_shared_state["last_per_class"].copy()
        # Синхронизируем счетчик из общего состояния
        if "camera_save_count" in st.session_state:
            st.session_state.camera_save_count = camera_shared_state["save_count"]
        else:
            st.session_state.camera_save_count = camera_shared_state["save_count"]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Текущий подсчёт", current_count)
    c2.metric("Инференс на кадр (ms)", f"{current_ms:.1f}")
    c3.metric("Автосохранений", camera_shared_state["save_count"])
    
    # Показываем детальную статистику по классам
    if current_per_class:
        st.write("**Подсчёт по классам:**")
        st.json(current_per_class)
    
    # Показываем статус автосохранения
    if auto_save_enabled and ctx.state.playing:
        time_since_last = time.time() - camera_shared_state["last_save"]
        remaining = max(0, auto_save_interval - time_since_last)
        st.info(f"⏱️ Следующее автосохранение через {remaining:.1f} сек (интервал: {auto_save_interval} сек)")

    if ctx.state.playing and st.button("Сохранить снимок в историю вручную"):
        with stats_lock:
            manual_count = camera_shared_state["last_count"]
            manual_ms = camera_shared_state["last_ms"]
            manual_per_class = camera_shared_state["last_per_class"].copy()
        
        per_class_snapshot = manual_per_class.copy()
        if not per_class_snapshot and manual_count > 0:
            per_class_snapshot = {"cup": manual_count}
        
        append_history(
            HistoryRecord(
                ts_iso=utc_now_iso(),
                kind="camera",
                input_name="webcam_manual",
                model_name=model_path,
                target_classes=classes,
                conf=float(conf),
                iou=float(iou),
                total_count=manual_count,
                per_class_count=per_class_snapshot,
                inference_ms=manual_ms,
            ),
            HISTORY_PATH,
        )
        camera_shared_state["last_save"] = time.time()
        st.success("Снимок добавлен в историю вручную.")


with tabs[3]:
    rows = read_history(HISTORY_PATH)
    
    # Дашборд статистики
    st.subheader("📊 Дашборд статистики")
    
    if rows:
        df = pd.DataFrame(rows)
        
        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего запросов", len(df))
        with col2:
            total_objects = df["total_count"].fillna(0).sum()
            st.metric("Всего обнаружено объектов", int(total_objects))
        with col3:
            avg_time = df["inference_ms"].fillna(0).mean()
            st.metric("Среднее время (ms)", f"{avg_time:.1f}")
        with col4:
            if "ts_iso" in df.columns:
                df["ts_iso"] = pd.to_datetime(df["ts_iso"], errors="coerce")
                unique_dates = df["ts_iso"].dt.date.nunique()
                st.metric("Дней активности", unique_dates)
        
        # Статистика по типам
        st.subheader("📈 Статистика по типам запросов")
        if "kind" in df.columns:
            kind_counts = df["kind"].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(kind_counts)
            with col2:
                st.write("**Распределение:**")
                for kind, count in kind_counts.items():
                    st.write(f"- {kind}: {count}")
        
        # Временная динамика по часам
        st.subheader("⏰ Временная динамика по часам")
        if "ts_iso" in df.columns and "total_count" in df.columns:
            df["datetime"] = pd.to_datetime(df["ts_iso"], errors="coerce")
            df["hour"] = df["datetime"].dt.hour
            hourly_stats = df.groupby("hour").agg({
                "total_count": "sum",
                "ts_iso": "count"
            }).rename(columns={"ts_iso": "requests"})
            st.line_chart(hourly_stats)
            
            # Круговая диаграмма распределения по часам
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Распределение запросов по часам")
                hour_counts = df["hour"].value_counts().sort_index()
                st.bar_chart(hour_counts)
            with col2:
                st.subheader("🥧 Распределение объектов по часам")
                hour_objects = df.groupby("hour")["total_count"].sum()
                st.area_chart(hour_objects)
        
        # Статистика по видео (динамика по кадрам)
        st.subheader("🎬 Статистика по видео")
        video_rows = [r for r in rows if r.get("kind") == "video"]
        if video_rows:
            st.write(f"Найдено видео записей: **{len(video_rows)}**")
            video_df = pd.DataFrame(video_rows)
            if "total_count" in video_df.columns and "max_per_frame" in video_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Среднее количество на кадр:**")
                    st.bar_chart(video_df.set_index("input_name")["total_count"])
                with col2:
                    st.write("**Максимум на кадр:**")
                    st.bar_chart(video_df.set_index("input_name")["max_per_frame"])
                
                # Точечная диаграмма: среднее vs максимум
                st.write("**Сравнение среднего и максимума:**")
                scatter_data = pd.DataFrame({
                    "Среднее": video_df["total_count"],
                    "Максимум": video_df["max_per_frame"]
                })
                st.scatter_chart(scatter_data)
            
            # Динамика по кадрам для выбранного видео
            st.subheader("📹 Динамика количества стаканов по кадрам")
            frames_index_path = DATA_DIR / "video_frames_index.jsonl"
            if frames_index_path.exists():
                try:
                    import json
                    frames_index = []
                    with frames_index_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                frames_index.append(json.loads(line))
                    
                    if frames_index:
                        video_names = [entry["video_name"] for entry in frames_index]
                        selected_video = st.selectbox("Выберите видео для анализа кадров:", video_names)
                        
                        if selected_video:
                            selected_entry = next((e for e in frames_index if e["video_name"] == selected_video), None)
                            if selected_entry:
                                frames_data_path = Path(selected_entry["frames_data_path"])
                                if frames_data_path.exists():
                                    with frames_data_path.open("r", encoding="utf-8") as f:
                                        frames_data = json.load(f)
                                    
                                    frame_counts = frames_data.get("frame_counts", [])
                                    if frame_counts:
                                        # Создаем DataFrame для визуализации
                                        frames_df = pd.DataFrame({
                                            "Кадр": range(len(frame_counts)),
                                            "Количество стаканов": frame_counts
                                        })
                                        
                                        st.write(f"**Видео: {selected_video}**")
                                        st.write(f"Всего обработанных кадров: {len(frame_counts)}")
                                        st.write(f"FPS: {frames_data.get('fps', 'N/A')}")
                                        
                                        # Линейный график динамики
                                        st.line_chart(frames_df.set_index("Кадр")["Количество стаканов"])
                                        
                                        # Статистика
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Среднее на кадр", f"{pd.Series(frame_counts).mean():.2f}")
                                        with col2:
                                            st.metric("Максимум на кадр", max(frame_counts))
                                        with col3:
                                            st.metric("Минимум на кадр", min(frame_counts))
                                else:
                                    st.warning(f"Файл данных по кадрам не найден: {frames_data_path}")
                            else:
                                st.warning("Данные по выбранному видео не найдены")
                    else:
                        st.info("Нет сохраненных данных по кадрам видео")
                except Exception as e:
                    st.error(f"Ошибка при загрузке данных по кадрам: {e}")
            else:
                st.info("Данные по кадрам видео будут доступны после обработки видео")
        else:
            st.info("Нет данных по видео для анализа")
        
        # Топ по количеству объектов
        st.subheader("🏆 Топ запросов по количеству объектов")
        if "total_count" in df.columns and "input_name" in df.columns:
            top_requests = df.nlargest(10, "total_count")[["input_name", "total_count", "ts_iso", "kind"]]
            st.dataframe(top_requests, use_container_width=True, hide_index=True)
            
            # Гистограмма распределения количества объектов
            st.subheader("📊 Распределение количества объектов")
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                counts = df["total_count"].dropna()
                ax.hist(counts, bins=min(20, len(counts.unique())), edgecolor='black')
                ax.set_xlabel("Количество объектов")
                ax.set_ylabel("Частота")
                ax.set_title("Гистограмма распределения количества объектов")
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                # Fallback: используем bar_chart с группировкой
                counts = df["total_count"].dropna()
                if len(counts) > 0:
                    bins = pd.cut(counts, bins=min(10, len(counts.unique())), precision=0)
                    hist_data = bins.value_counts().sort_index()
                    hist_df = pd.DataFrame({
                        "Интервал": [str(x) for x in hist_data.index],
                        "Частота": hist_data.values
                    })
                    st.bar_chart(hist_df.set_index("Интервал"))
    
    st.divider()
    
    # История запросов
    st.subheader("📋 История запросов")
    st.write(f"Файл: `{HISTORY_PATH.as_posix()}`. Записей: **{len(rows)}**")
    st.dataframe(rows, use_container_width=True, height=320)

    c1, c2, c3 = st.columns(3)
    with c1:
        excel = make_excel_bytes(rows)
        st.download_button("Скачать Excel", data=excel, file_name="history.xlsx")
    with c2:
        pdf = make_pdf_bytes(rows)
        st.download_button("Скачать PDF", data=pdf, file_name="report.pdf")
    with c3:
        if st.button("Очистить историю", type="secondary", disabled=len(rows) == 0):
            clear_history(HISTORY_PATH)
            st.session_state.camera_save_count = 0
            st.rerun()

