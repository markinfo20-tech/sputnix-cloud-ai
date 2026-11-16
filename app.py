# -*- coding: utf-8 -*-
"""SPUTNIX CLOUD AI — deployment-oriented Streamlit application."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from PIL import Image, ImageOps

from cloud_core import cloud_percentage, expand_uploads, build_results_zip

MODEL_ID = "uga-gaim/CLIPSeg-CloudSEN12Plus-FFT"
LOGO_URL = "https://s3.twcstorage.ru/35afe091-cipr/award-project-logos/Sputnix_new_logo_2025-2.png"
MODEL_WEIGHTS_MB = 603
RECOMMENDED_RAM_MB = 1536  # estimated practical minimum, not a vendor guarantee

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

st.set_page_config(
    page_title="SPUTNIX CLOUD AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --red:#d71920; --black:#151515; --ink:#242424; --muted:#707070; --line:#e8e8e8; }
html, body, [class*="css"] { font-family:Manrope,Arial,sans-serif; }
.stApp { background:#f6f6f6; color:var(--ink); }
.block-container { max-width:1360px; padding-top:1.15rem; padding-bottom:2.5rem; }
[data-testid="stSidebar"] { background:#171717; border-right:3px solid var(--red); }
[data-testid="stSidebar"] * { color:#f7f7f7 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color:#f7f7f7 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#c9c9c9 !important; }
.brand { background:#fff; border:1px solid var(--line); border-radius:18px; padding:20px 26px;
  display:flex; align-items:center; gap:22px; box-shadow:0 7px 24px rgba(0,0,0,.035); margin-bottom:22px; }
.brand-logo { flex:0 0 auto; width:200px; max-width:28%; display:flex; align-items:center; }
.brand-logo img { max-width:100%; max-height:68px; object-fit:contain; }
.brand-divider { height:62px; width:1px; background:#e5e5e5; }
.brand-kicker { color:var(--red); font-size:10px; font-weight:800; letter-spacing:.2em; text-transform:uppercase; }
.brand h1 { margin:2px 0 3px; color:var(--black); font-size:clamp(25px,3vw,37px); line-height:1.15; font-weight:800; letter-spacing:-.04em; }
.brand p { margin:0; color:#666; font-size:13px; }
.section-head { display:flex; align-items:center; gap:10px; margin:23px 0 12px; color:var(--black); font-size:20px; font-weight:800; }
.section-head:before { content:''; width:5px; height:23px; background:var(--red); border-radius:4px; }
.info-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:15px 17px; min-height:94px; }
.info-num { color:var(--red); font-size:10px; letter-spacing:.14em; font-weight:800; }
.info-title { color:#161616; font-weight:800; margin:5px 0 2px; }
.info-text { color:#747474; font-size:12px; line-height:1.45; }
.metric { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px 18px; min-height:112px; }
.metric-label { color:#777; font-size:10px; letter-spacing:.13em; font-weight:800; text-transform:uppercase; }
.metric-value { color:#111; font-size:29px; font-weight:800; letter-spacing:-.04em; padding-top:5px; }
.metric-line { height:4px; width:30px; background:var(--red); border-radius:4px; margin-bottom:12px; }
.stButton>button, .stDownloadButton>button { min-height:44px; border-radius:9px; font-weight:800; }
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] { background:var(--red); border-color:var(--red); color:#fff; }
.stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover { background:#b9151c; border-color:#b9151c; }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #c8c8c8; border-radius:13px; padding:8px; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.small-note { color:#777; font-size:11px; line-height:1.5; }
.result-good { color:#087443; font-weight:800; }
.result-bad { color:var(--red); font-weight:800; }
hr { border-color:#e6e6e6; }
@media(max-width:700px) { .brand { padding:16px; gap:13px; } .brand-logo { width:100px; } .brand-divider { height:48px; } .brand p { font-size:11px; } }
</style>
""", unsafe_allow_html=True)


def get_memory_limit_mb() -> int | None:
    """Read Linux cgroup memory limit, where available."""
    candidates = [Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")]
    for path in candidates:
        try:
            value = path.read_text().strip()
            if value == "max":
                continue
            limit = int(value)
            # cgroup v1 sometimes reports a huge sentinel for no limit.
            if limit > 1_000_000_000_000:
                continue
            return int(limit / (1024 * 1024))
        except (OSError, ValueError):
            continue
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    # Heavy ML imports are deferred until the user actually starts inference.
    import torch
    import torch.nn.functional as F
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID, use_safetensors=True)
    model.eval()
    return processor, model, torch, F


def read_image(name: str, raw: bytes) -> Image.Image:
    suffix = Path(name).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        array = np.asarray(tifffile.imread(io.BytesIO(raw)))
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.ndim == 3:
            if array.shape[-1] in (3, 4):
                array = array[..., :3]
            elif array.shape[0] in (3, 4):
                array = np.moveaxis(array[:3], 0, -1)
            else:
                raise ValueError(f"Не удалось определить RGB-каналы TIFF: {array.shape}. Ожидаются 3 или 4 канала.")
        else:
            raise ValueError(f"Неподдерживаемая размерность TIFF: {array.shape}")
        array = array.astype(np.float32, copy=False)
        rgb = np.zeros(array.shape, dtype=np.uint8)
        for band_index in range(3):
            band = array[..., band_index]
            valid = np.isfinite(band)
            if not valid.any():
                continue
            low, high = np.percentile(band[valid], [2, 98])
            if high <= low:
                high = low + 1.0
            scaled = np.clip((band - low) / (high - low), 0, 1)
            rgb[..., band_index] = np.nan_to_num(scaled * 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")
    with Image.open(io.BytesIO(raw)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def predict(image: Image.Image, processor, model, torch, functional):
    labels_prompt = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
    inputs = processor(text=labels_prompt, images=[image] * 4, padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = functional.interpolate(
            logits.unsqueeze(1), size=(image.height, image.width),
            mode="bilinear", align_corners=False
        ).squeeze(1)
        labels = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    return labels, cloud_percentage(labels)


PALETTE = np.array([
    [20, 20, 20],       # clear
    [215, 25, 32],      # thick cloud
    [245, 154, 45],     # thin cloud
    [130, 137, 148],    # shadow
], dtype=np.uint8)


def make_mask(labels: np.ndarray) -> Image.Image:
    return Image.fromarray(PALETTE[labels])


def make_overlay(image: Image.Image, labels: np.ndarray) -> Image.Image:
    source = np.asarray(image).astype(np.float32)
    mask = np.asarray(make_mask(labels)).astype(np.float32)
    alpha = np.where(labels == 0, 0.0, 0.45)[..., None]
    return Image.fromarray(np.clip(source * (1 - alpha) + mask * alpha, 0, 255).astype(np.uint8))


# Header: a logo image with a clean text fallback if the remote image is unavailable.
st.markdown(f"""
<div class="brand">
  <div class="brand-logo">
    <img src="{LOGO_URL}" alt="SPUTNIX" onerror="this.style.display='none';this.nextElementSibling.style.display='inline'">
    <span style="display:none;font-size:25px;font-weight:900;letter-spacing:-1px;color:#111">SPUTNIX</span>
  </div>
  <div class="brand-divider"></div>
  <div>
    <div class="brand-kicker">EARTH OBSERVATION · AI ANALYTICS</div>
    <h1>CLOUD AI</h1>
    <p>Интеллектуальная оценка облачности спутниковых изображений</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# SPUTNIX CLOUD AI")
    st.caption("Параметры обработки")
    threshold = st.slider("Допустимая облачность", 0, 100, 75, 1, format="%d%%")
    st.markdown("---")
    st.markdown("**Правило отбора**")
    st.markdown(f"🟢 Принять: **≤ {threshold}%**")
    st.markdown(f"🔴 Отклонить: **> {threshold}%**")
    st.markdown("---")
    st.markdown("**Модель**")
    st.caption("CLIPSeg · CloudSEN12+ · 4 класса")
    st.caption("Облаками считаются плотные и тонкие облака. Тени показываются отдельно и в процент облачности не входят.")
    st.markdown("---")
    st.caption("Модель скачивается при первом анализе. Рекомендуемая память хоста — примерно от 1,5 ГБ.")
    st.caption("Многоканальный TIFF: первые три канала интерпретируются как RGB. Проверьте их порядок для данных Sputnix.")

st.markdown('<div class="section-head">Рабочий процесс</div>', unsafe_allow_html=True)
cols = st.columns(3)
for col, number, title, description in zip(
    cols,
    ["01", "02", "03"],
    ["Загрузите", "Анализируйте", "Получите результат"],
    ["Отдельные изображения или ZIP-архив с изображениями.", "Модель строит маску облаков и рассчитывает долю пикселей.", "Просмотрите карту и скачайте полный ZIP-отчёт."],
):
    col.markdown(
        f'<div class="info-card"><div class="info-num">STEP {number}</div>'
        f'<div class="info-title">{title}</div><div class="info-text">{description}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-head">1. Добавить снимки</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Выберите изображения или ZIP-архив",
    type=["png", "jpg", "jpeg", "tif", "tiff", "webp", "zip"],
    accept_multiple_files=True,
    help="ZIP распаковывается в памяти без сохранения файлов на диск. Внутри поддерживаются PNG, JPG, TIFF и WEBP.",
    key="satellite_uploads",
)

records = []
zip_warnings = []
if uploaded:
    source_payloads = [(file.name, file.getvalue()) for file in uploaded]
    records, zip_warnings = expand_uploads(source_payloads)
    st.success(f"Найдено изображений для анализа: {len(records)}")
    if zip_warnings:
        with st.expander(f"Предупреждения при чтении ({len(zip_warnings)})"):
            for warning in zip_warnings[:20]:
                st.write(f"• {warning}")
    if records:
        st.caption("Первые найденные файлы: " + ", ".join(record["name"] for record in records[:5]))
else:
    st.caption("Можно загрузить сразу несколько файлов или один ZIP с набором снимков.")

host_ram = get_memory_limit_mb()
if host_ram is not None:
    if host_ram < RECOMMENDED_RAM_MB:
        st.warning(
            f"Ограничение памяти сервера: около {host_ram} МБ. Файл весов модели занимает примерно "
            f"{MODEL_WEIGHTS_MB} МБ, а для работы нужны дополнительные ресурсы Python и PyTorch. "
            "На таком тарифе анализ этой моделью не запускается, чтобы страница не зависала. "
            "Нужен сервер с большим объёмом RAM."
        )

button_col, hint_col = st.columns([1, 2])
with button_col:
    start = st.button("НАЧАТЬ АНАЛИЗ", type="primary", use_container_width=True, disabled=not bool(records))
with hint_col:
    st.markdown('<div class="small-note" style="padding-top:12px">Загрузка файла сама по себе не запускает модель. Анализ начинается только после нажатия кнопки.</div>', unsafe_allow_html=True)

if start and records:
    if host_ram is not None and host_ram < RECOMMENDED_RAM_MB:
        st.error("Недостаточно памяти на сервере для выбранной ИИ-модели. Перезапустите приложение на хосте с большей RAM.")
    else:
        try:
            with st.status("Подготовка модели ИИ…", expanded=True) as status:
                st.write("Проверяем и загружаем веса CLIPSeg. При первом запуске скачивается около 603 МБ.")
                processor, model, torch, functional = load_model()
                status.update(label="Модель готова", state="complete", expanded=False)

            outputs = []
            progress = st.progress(0, text="Начинаем обработку…")
            started_at = time.perf_counter()
            for index, record in enumerate(records, 1):
                file_started = time.perf_counter()
                try:
                    original = read_image(record["name"], record["raw"])
                    work = original.copy()
                    work.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    labels, cloud_pct = predict(work, processor, model, torch, functional)
                    decision = "ОТКЛОНЁН" if cloud_pct > threshold else "ПРИНЯТ"
                    outputs.append({
                        "name": record["name"], "raw": record["raw"], "image": work,
                        "cloud": cloud_pct, "decision": decision, "threshold": threshold,
                        "mask": make_mask(labels), "overlay": make_overlay(work, labels),
                        "seconds": time.perf_counter() - file_started,
                    })
                except Exception as exc:
                    st.error(f"{record['name']}: {exc}")
                progress.progress(index / len(records), text=f"Обработано {index} из {len(records)}")
            if outputs:
                st.session_state["sputnix_results"] = outputs
                st.session_state["sputnix_elapsed"] = time.perf_counter() - started_at
                st.session_state["sputnix_threshold"] = threshold
            else:
                st.session_state.pop("sputnix_results", None)
                st.error("Не удалось обработать ни одного файла. Проверьте формат и каналы TIFF.")
        except Exception as exc:
            st.error(f"Не удалось запустить модель: {type(exc).__name__}: {exc}")
            st.info("Если сбой связан с памятью или тайм-аутом, тариф сервера не подходит для этой модели. Полный лог ошибки доступен в панели хостинга.")

results = st.session_state.get("sputnix_results", [])
if results:
    st.markdown('<div class="section-head">2. Результаты анализа</div>', unsafe_allow_html=True)
    accepted = sum(row["decision"] == "ПРИНЯТ" for row in results)
    rejected = len(results) - accepted
    avg_cloud = float(np.mean([row["cloud"] for row in results]))
    metrics = [
        ("Обработано", str(len(results))),
        ("Принято", str(accepted)),
        ("Отклонено", str(rejected)),
        ("Средняя облачность", f"{avg_cloud:.1f}%"),
    ]
    metric_cols = st.columns(4)
    for col, (label, value) in zip(metric_cols, metrics):
        col.markdown(
            f'<div class="metric"><div class="metric-line"></div><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )
    if "sputnix_elapsed" in st.session_state:
        st.caption(f"Общее время обработки: {st.session_state['sputnix_elapsed']:.1f} сек.")

    st.markdown('<div class="section-head">Сводная таблица</div>', unsafe_allow_html=True)
    table = pd.DataFrame([{
        "Файл": row["name"],
        "Облачность": f"{row['cloud']:.2f}%",
        "Порог": f"{row['threshold']}%",
        "Решение": row["decision"],
        "Время": f"{row['seconds']:.1f} с",
    } for row in results])
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-head">Детальный просмотр</div>', unsafe_allow_html=True)
    selected = st.selectbox("Выберите снимок", range(len(results)), format_func=lambda idx: results[idx]["name"])
    current = results[selected]
    image_cols = st.columns(3)
    image_cols[0].image(current["image"], caption="Исходное изображение", use_container_width=True)
    image_cols[1].image(current["mask"], caption="Маска классов", use_container_width=True)
    image_cols[2].image(current["overlay"], caption="Наложение маски", use_container_width=True)
    st.caption("Красный — плотные облака · Оранжевый — тонкие облака · Серый — тени · Чёрный — чистые пиксели.")
    if current["decision"] == "ПРИНЯТ":
        st.success(f"ПРИНЯТ — облачность {current['cloud']:.2f}% не превышает порог {current['threshold']}%.")
    else:
        st.error(f"ОТКЛОНЁН — облачность {current['cloud']:.2f}% выше порога {current['threshold']}%.")

    st.markdown('<div class="section-head">3. Скачать результаты</div>', unsafe_allow_html=True)
    st.download_button(
        "СКАЧАТЬ ПОЛНЫЙ ZIP-ОТЧЁТ",
        data=build_results_zip(results),
        file_name="SPUTNIX_CLOUD_AI_results.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
    st.caption("Внутри ZIP: исходные снимки в папках accepted/rejected, маски, наложения, report.csv и пояснение.")

st.markdown("---")
st.markdown(
    '<div class="small-note">SPUTNIX CLOUD AI · Демонстрационный прототип. Модель обучена на Sentinel-2 / CloudSEN12+. '
    'Показатель облачности — доля пикселей, классифицированных как thick cloud или thin cloud. '
    'Точность на данных Sputnix нельзя считать подтверждённой без размеченной контрольной выборки. '
    'Для многоспектрального TIFF требуется подтвердить порядок каналов.</div>',
    unsafe_allow_html=True,
)
