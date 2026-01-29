# app.py (обновленная версия)
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.analytics import AnalyticsEngine
from src.camera import CameraProcessor, get_camera_processor, reset_camera_processor
from src.history_manager import HistoryManager
from src.infer import detect_and_count_on_image, detect_and_count_on_video
from src.reports import make_excel_bytes, make_pdf_bytes

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timezone

# Конфигурация
APP_TITLE = "Практика CV: подсчёт стаканов/кружек (YOLOv8)"
DATA_DIR = Path("data")
HISTORY_PATH = DATA_DIR / "history.jsonl"

# Инициализация менеджеров
history_manager = HistoryManager(HISTORY_PATH)
analytics_engine = AnalyticsEngine(history_manager)


def _parse_target_classes(text: str) -> List[str]:
    """Парсит целевые классы из текста"""
    raw = [t.strip() for t in text.replace(";", ",").split(",")]
    return [t for t in raw if t]


def _save_uploaded_to_temp(uploaded) -> str:
    """Сохраняет загруженный файл во временный файл"""
    suffix = Path(uploaded.name).suffix or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getbuffer())
        return f.name


# Настройка страницы
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Боковая панель
with st.sidebar:
    st.subheader("Настройки модели")
    model_path = st.text_input("Модель (веса Ultralytics)", value="yolov8n.pt")
    conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("IoU", 0.05, 0.95, 0.45, 0.05)

    st.subheader("Видео")
    sample_every = st.number_input("Обрабатывать каждый N‑й кадр", min_value=1, max_value=30, value=1, step=1)

    st.subheader("Камера")
    auto_save_enabled = st.checkbox("Автосохранение", value=True)
    auto_save_interval = st.number_input("Интервал автосохранения (сек)",
                                         min_value=5, max_value=300, value=15, step=5,
                                         disabled=not auto_save_enabled)

# Вкладки
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
        classes = ["cup"]
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, up in enumerate(up_list):
            status_text.text(f"Обработка {idx + 1}/{len(up_list)}: {up.name}")
            progress_bar.progress((idx + 1) / len(up_list))

            # Обработка изображения
            image_rgb = np.array(Image.open(up).convert("RGB"))
            annotated_rgb, summary = detect_and_count_on_image(
                image_rgb,
                model_path=model_path,
                target_classes=classes,
                conf=float(conf),
                iou=float(iou),
            )

            # Отображение результата
            st.subheader(f"Результат: {up.name}")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(annotated_rgb, caption="Аннотированное изображение", use_container_width=True)
            with col2:
                st.metric("Всего объектов", summary.total_count)
                st.metric("Время инференса (ms)", f"{summary.inference_ms:.1f}")
                st.metric("Классы", ", ".join(summary.selected_class_names))
                st.json(summary.per_class_count)

            # Сохранение в историю
            history_manager.add_image_record(
                input_name=up.name,
                model_name=summary.model_name,
                total_count=summary.total_count,
                per_class_count=summary.per_class_count,
                inference_ms=summary.inference_ms,
                target_classes=classes,
                conf=summary.conf_threshold,
                iou=summary.iou_threshold,
                image_width=summary.image_width,
                image_height=summary.image_height,
            )

        progress_bar.empty()
        status_text.empty()
        st.success(f"Обработано {len(up_list)} изображений. Записи добавлены в историю.")

with tabs[1]:
    st.write("Загрузите одно или несколько видео и нажмите **Запустить обработку**.")

    if "video_results" not in st.session_state:
        st.session_state.video_results = []

    upv_list = st.file_uploader(
        "Видео (mp4/avi/mov)",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True,
        key="video_uploader"
    )

    if st.session_state.video_results:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Очистить результаты", type="secondary"):
                st.session_state.video_results = []
                st.rerun()

    runv = st.button("Запустить обработку видео", type="primary", disabled=not upv_list)

    if upv_list:
        st.write(f"Загружено видео: **{len(upv_list)}**")
        for upv in upv_list:
            st.text(f"📹 {upv.name} ({upv.size / 1024 / 1024:.2f} MB)")

    # Показ сохраненных результатов
    if st.session_state.video_results:
        st.divider()
        st.subheader("📹 Результаты обработки видео")

        for result_idx, result in enumerate(st.session_state.video_results):
            st.subheader(f"Результат: {result['input_name']}")
            st.write(
                f"Подсчёт: **{result['total_count']}** (среднее на обработанный кадр), "
                f"максимум на кадр: **{result['max_per_frame']}**"
            )
            st.write(f"Время обработки (ms): **{result['inference_ms']:.1f}**")

            # Отображение видео
            video_file = Path(result['out_path'])
            if video_file.exists() and video_file.stat().st_size > 0:
                try:
                    # Проверяем размер файла
                    file_size_mb = video_file.stat().st_size / (1024 * 1024)

                    if file_size_mb > 50:  # Если файл больше 50 MB
                        st.warning(f"Видео файл очень большой ({file_size_mb:.1f} MB). Для просмотра скачайте его.")

                        # Показываем первый кадр как превью
                        try:
                            import cv2

                            cap = cv2.VideoCapture(str(video_file))
                            ret, frame = cap.read()
                            if ret:
                                # Конвертируем BGR в RGB
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.image(frame_rgb, caption="Первый кадр видео", use_container_width=True)
                            cap.release()
                        except:
                            pass
                    else:
                        # Отображаем видео напрямую
                        with open(video_file, "rb") as f:
                            video_bytes = f.read()

                        st.video(video_bytes)

                    # Кнопка скачивания
                    st.download_button(
                        label=f"💾 Скачать аннотированное видео ({file_size_mb:.1f} MB)",
                        data=video_bytes if file_size_mb <= 50 else open(video_file, "rb").read(),
                        file_name=video_file.name,
                        mime="video/mp4",
                        key=f"download_video_{result_idx}"
                    )

                except Exception as e:
                    st.error(f"Ошибка при отображении видео: {str(e)}")
                    st.code(
                        f"Путь: {video_file}\nРазмер: {video_file.stat().st_size if video_file.exists() else 'файл не найден'} байт")

                    # Пробуем альтернативный метод отображения
                    try:
                        # Пробуем использовать st.video с путем
                        st.video(str(video_file))
                    except Exception as e2:
                        st.error(f"Альтернативный метод также не сработал: {str(e2)}")
            else:
                st.error(f"Видео файл не найден или пуст: {video_file}")
                if video_file.exists():
                    st.info(f"Размер файла: {video_file.stat().st_size} байт")
            st.divider()

    if runv and upv_list:
        classes = ["cup"]
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, upv in enumerate(upv_list):
            status_text.text(f"Обработка видео {idx + 1}/{len(upv_list)}: {upv.name}...")
            progress_bar.progress((idx + 1) / len(upv_list))

            tmp_in = _save_uploaded_to_temp(upv)

            # Прогресс-бар для видео
            video_progress = st.progress(0)
            video_status = st.empty()


            def update_video_progress(p: float):
                video_progress.progress(p)
                video_status.text(f"Обработка кадров: {p * 100:.1f}%")


            with st.spinner(f"Обработка {upv.name}..."):
                out_path, summary, frames_data_path, frame_counts = detect_and_count_on_video(
                    tmp_in,
                    model_path=model_path,
                    target_classes=classes,
                    conf=float(conf),
                    iou=float(iou),
                    sample_every_n_frames=int(sample_every),
                    progress_callback=update_video_progress,
                )

            video_progress.empty()
            video_status.empty()

            # Сохранение результата
            result_data = {
                "input_name": upv.name,
                "out_path": out_path,
                "total_count": summary.total_count,
                "max_per_frame": summary.max_per_frame,
                "inference_ms": summary.inference_ms,
                "summary": summary,
                "frames_data_path": frames_data_path,
                "frame_counts": frame_counts
            }
            st.session_state.video_results.append(result_data)

            st.success(f"✅ Видео {upv.name} обработано!")

            # Сохранение в историю
            history_manager.add_video_record(
                input_name=upv.name,
                model_name=summary.model_name,
                total_count=summary.total_count,
                per_class_count=summary.per_class_count,
                inference_ms=summary.inference_ms,
                output_artifact=str(out_path),
                max_per_frame=summary.max_per_frame,
                target_classes=classes,
                conf=summary.conf_threshold,
                iou=summary.iou_threshold,
                image_width=summary.image_width,
                image_height=summary.image_height,
            )

        if frames_data_path:
            try:
                frames_index_path = DATA_DIR / "video_frames_index.jsonl"
                frames_index_path.parent.mkdir(parents=True, exist_ok=True)
                with frames_index_path.open("a", encoding="utf-8") as f:
                    index_entry = {
                        "video_name": upv.name,
                        "frames_data_path": frames_data_path,
                        "ts_iso": datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                    }
                    f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                st.warning(f"Не удалось сохранить индекс данных по кадрам: {e}")

        progress_bar.empty()
        status_text.empty()
        st.rerun()

with tabs[2]:
    st.write("Нажмите **Start**, чтобы открыть поток с камеры. Онлайн-подсчет в реальном времени.")

    import time
    import av
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

    # Создаем или получаем процессор камеры
    camera_processor = get_camera_processor(
        model_path=model_path,
        target_classes=["cup"],
        conf=float(conf),
        iou=float(iou)
    )


    class OnlineCameraTransformer(VideoTransformerBase):
        """Трансформер для онлайн-обработки камеры"""

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Обработка кадра
            annotated_frame, count, inference_ms, per_class_count = camera_processor.process_frame(img)

            # Автосохранение
            if auto_save_enabled:
                current_time = time.time()
                time_since_last = current_time - camera_processor.state.last_save

                if time_since_last >= auto_save_interval:
                    # Сохраняем в историю
                    history_manager.add_camera_record(
                        input_name=f"webcam_auto_{camera_processor.state.save_count}",
                        model_name=model_path,
                        total_count=count,
                        per_class_count=per_class_count,
                        inference_ms=inference_ms,
                        target_classes=["cup"],
                        conf=float(conf),
                        iou=float(iou),
                    )

                    # Обновляем состояние
                    with camera_processor.state.lock:
                        camera_processor.state.last_save = current_time
                        camera_processor.state.save_count += 1

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


    # Запуск потока камеры
    ctx = webrtc_streamer(
        key="online_camera",
        video_transformer_factory=OnlineCameraTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Контейнеры для онлайн-обновления
    metrics_container = st.container()
    stats_container = st.container()


    # Функция для обновления интерфейса
    def update_camera_ui():
        with metrics_container:
            # Получаем текущую статистику
            count, inference_ms, per_class_count, save_count = camera_processor.get_current_stats()

            # Отображаем метрики
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("📊 Текущий подсчет", count, delta=None)
            with c2:
                st.metric("⚡ Инференс", f"{inference_ms:.1f} ms")
            with c3:
                st.metric("💾 Автосохранений", save_count)
            with c4:
                if auto_save_enabled:
                    time_since_last = time.time() - camera_processor.state.last_save
                    remaining = max(0, auto_save_interval - time_since_last)
                    st.metric("⏱️ Следующее", f"{remaining:.0f} сек")

        with stats_container:
            # Детальная статистика
            if per_class_count:
                st.write("**Детальный подсчет по классам:**")

                # Создаем красивую таблицу
                class_data = []
                for class_name, class_count in per_class_count.items():
                    class_data.append({
                        "Класс": class_name,
                        "Количество": class_count,
                        "Процент": f"{(class_count / max(1, count) * 100):.1f}%"
                    })

                if class_data:
                    stats_df = pd.DataFrame(class_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
            elif count > 0:
                st.info(f"🔍 Обнаружено стаканов/кружек: **{count}**")
            else:
                st.info("🔍 Ожидание обнаружения объектов...")


    # Обновляем интерфейс
    if ctx.state.playing:
        camera_processor.start()
        update_camera_ui()

        # Кнопка ручного сохранения
        if st.button("💾 Сохранить текущий снимок вручную", type="secondary"):
            count, inference_ms, per_class_count, _ = camera_processor.get_current_stats()

            history_manager.add_camera_record(
                input_name="webcam_manual",
                model_name=model_path,
                total_count=count,
                per_class_count=per_class_count,
                inference_ms=inference_ms,
                target_classes=["cup"],
                conf=float(conf),
                iou=float(iou),
            )

            st.success("Снимок сохранен в историю!")
            st.rerun()
    else:
        camera_processor.stop()
        st.info("👆 Нажмите 'Start' для запуска камеры")

with tabs[3]:
    rows = history_manager.get_all_records()

    # Дашборд статистики
    st.subheader("📊 Дашборд статистики")

    if rows:
        # Основные метрики
        basic_stats = analytics_engine.get_basic_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего запросов", basic_stats["total_requests"])
        with col2:
            st.metric("Всего объектов", basic_stats["total_objects"])
        with col3:
            st.metric("Среднее время", f"{basic_stats['avg_inference_time']:.1f} ms")
        with col4:
            st.metric("Дней активности", basic_stats["active_days"])

        # Визуализации
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Распределение по типам")
            kind_dist = analytics_engine.get_kind_distribution()
            if not kind_dist.empty:
                fig = px.pie(kind_dist, values='count', names='kind',
                             title="Типы запросов")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("⏰ Активность по часам")
            hourly_stats = analytics_engine.get_hourly_stats()
            if not hourly_stats.empty:
                fig = px.bar(hourly_stats, x='hour', y='requests',
                             title="Запросы по часам")
                st.plotly_chart(fig, use_container_width=True)

        # Аналитика видео
        st.subheader("🎬 Статистика по видео")
        video_analytics = analytics_engine.get_video_analytics()

        if video_analytics["video_count"] > 0:
            video_df = video_analytics["data"]

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Среднее количество на кадр:**")
                fig = px.bar(video_df, x='input_name', y='total_count',
                             title="Среднее на кадр")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Максимум на кадр:**")
                fig = px.bar(video_df, x='input_name', y='max_per_frame',
                             title="Максимум на кадр")
                st.plotly_chart(fig, use_container_width=True)

            # Анализ кадров для выбранного видео
            st.subheader("📹 Анализ динамики по кадрам")

            # Пытаемся найти данные по кадрам
            frames_index_path = DATA_DIR / "video_frames_index.jsonl"
            if frames_index_path.exists():
                try:
                    frames_index = []
                    with frames_index_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                frames_index.append(json.loads(line))

                    if frames_index:
                        video_names = [entry["video_name"] for entry in frames_index]
                        selected_video = st.selectbox("Выберите видео для анализа:", video_names)

                        if selected_video:
                            selected_entry = next((e for e in frames_index if e["video_name"] == selected_video), None)
                            if selected_entry:
                                frames_data_path = Path(selected_entry["frames_data_path"])
                                frame_analysis = analytics_engine.get_frame_analysis(
                                    selected_video, frames_data_path
                                )

                                if frame_analysis:
                                    st.write(f"**Видео: {selected_video}**")

                                    # Статистика
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Всего кадров", frame_analysis["total_frames"])
                                    with col2:
                                        st.metric("Среднее", f"{frame_analysis['frame_stats']['mean']:.2f}")
                                    with col3:
                                        st.metric("Максимум", frame_analysis['frame_stats']['max'])
                                    with col4:
                                        st.metric("FPS", frame_analysis['fps'])

                                    # График динамики
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        y=frame_analysis["frame_counts"],
                                        mode='lines',
                                        name='Количество стаканов',
                                        line=dict(color='blue', width=2)
                                    ))
                                    fig.update_layout(
                                        title=f"Динамика количества стаканов: {selected_video}",
                                        xaxis_title="Номер кадра",
                                        yaxis_title="Количество стаканов",
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка при анализе кадров: {e}")

    st.divider()

    # История запросов
    st.subheader("📋 История запросов")
    st.write(f"Записей в истории: **{len(rows)}**")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("История пуста")

    # Экспорт
    st.subheader("📤 Экспорт данных")
    col1, col2, col3 = st.columns(3)

    with col1:
        if rows:
            excel_data = make_excel_bytes(rows)
            st.download_button(
                "📊 Скачать Excel",
                data=excel_data,
                file_name="history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.button("📊 Скачать Excel", disabled=True)

    with col2:
        if rows:
            pdf_data = make_pdf_bytes(rows)
            st.download_button(
                "📄 Скачать PDF",
                data=pdf_data,
                file_name="report.pdf",
                mime="application/pdf"
            )
        else:
            st.button("📄 Скачать PDF", disabled=True)

    with col3:
        if rows and st.button("🗑️ Очистить историю", type="secondary"):
            history_manager.clear_history()
            reset_camera_processor()
            st.success("История очищена!")
            st.rerun()