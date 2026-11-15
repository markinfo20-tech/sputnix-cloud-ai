# -*- coding: utf-8 -*-
"""SPUTNIX CLOUD AI — Streamlit Cloud edition."""
import os
from pathlib import Path

# Project configuration
LOGO_URL = "https://s3.twcstorage.ru/35afe091-cipr/award-project-logos/Sputnix_new_logo_2025-2.png"
MODEL_ID = "uga-gaim/CLIPSeg-CloudSEN12Plus-FFT"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import io
import csv
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import tifffile
from PIL import Image, ImageOps
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

st.set_page_config(
    page_title="SPUTNIX CLOUD AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Manrope, Arial, sans-serif; }
.stApp { background:#f3f4f6; color:#171717; }
.block-container { max-width:1480px; padding-top:1.3rem; padding-bottom:2rem; }
[data-testid="stSidebar"] { background:#171717; border-right:3px solid #d71920; }
[data-testid="stSidebar"] * { color:#f7f7f7 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color:#f7f7f7; }
.brand-head { background:#fff; border:1px solid #e3e3e3; border-radius:18px;
   padding:18px 25px; display:flex; align-items:center; gap:25px;
   box-shadow:0 5px 24px rgba(0,0,0,.045); margin-bottom:20px; }
.brand-head img { width:245px; max-width:35%; height:auto; object-fit:contain; }
.brand-copy { border-left:1px solid #dedede; padding-left:24px; }
.eyebrow { color:#d71920; font-size:11px; font-weight:800; letter-spacing:.17em; }
.brand-copy h1 { margin:4px 0 3px; color:#111; font-size:clamp(24px,3vw,36px); font-weight:800; }
.brand-copy p { margin:0; color:#666; font-size:14px; }
.section-title { color:#151515; font-size:21px; font-weight:800; margin:12px 0 12px; }
.card { background:#fff; border:1px solid #e5e5e5; border-radius:15px;
   padding:18px 20px; box-shadow:0 4px 16px rgba(0,0,0,.035); height:100%; }
.card-label { font-size:11px; color:#777; text-transform:uppercase; letter-spacing:.11em; font-weight:700; }
.card-value { font-size:31px; font-weight:800; color:#141414; margin-top:4px; }
.card-accent { width:33px; height:4px; background:#d71920; border-radius:9px; margin-bottom:13px; }
.step { background:#fff; border:1px solid #e4e4e4; border-radius:13px; padding:16px;
   color:#333; min-height:110px; }
.step-num { color:#d71920; font-weight:800; font-size:12px; letter-spacing:.12em; }
.step-title { font-weight:800; color:#171717; margin:5px 0 4px; }
.muted { color:#707070; font-size:12px; }
.decision-good { color:#16834a; font-weight:800; }
.decision-bad { color:#d71920; font-weight:800; }
.stButton>button, .stDownloadButton>button { border-radius:9px; font-weight:800; min-height:44px; }
.stButton>button[kind="primary"] { background:#d71920; border:1px solid #d71920; color:white; }
.stButton>button[kind="primary"]:hover { background:#b9141a; border-color:#b9141a; }
div[data-testid="stDataFrame"] { background:#fff; border-radius:12px; }
hr { border-color:#dedede; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model, device

def read_uploaded_image(upload):
    raw = upload.getvalue()
    suffix = Path(upload.name).suffix.lower()
    if suffix in (".tif", ".tiff"):
        arr = tifffile.imread(io.BytesIO(raw))
        arr = np.asarray(arr)
        if arr.ndim == 2:
            raise ValueError("В TIFF найден один канал. Для RGB-модели нужны минимум три канала.")
        if arr.ndim == 3:
            # Accept HWC or CHW; prefer channel-last when final dimension is 3/4.
            if arr.shape[-1] in (3, 4):
                rgb = arr[..., :3]
            elif arr.shape[0] in (3, 4):
                rgb = np.moveaxis(arr[:3], 0, -1)
            else:
                raise ValueError(f"Не удалось определить RGB-каналы TIFF: размер {arr.shape}")
        else:
            raise ValueError(f"Неподдерживаемая размерность TIFF: {arr.shape}")
        rgb = rgb.astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for ch in range(3):
            band = rgb[..., ch]
            good = np.isfinite(band)
            if not good.any():
                continue
            lo, hi = np.percentile(band[good], [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((band - lo) / (hi - lo), 0, 1)
            out[..., ch] = np.nan_to_num(scaled * 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")

def run_segmentation(image, processor, model, device):
    prompts = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
    inputs = processor(
        text=prompts, images=[image] * len(prompts),
        padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = F.interpolate(
            logits.unsqueeze(1), size=(image.height, image.width),
            mode="bilinear", align_corners=False
        ).squeeze(1)
        labels = logits.argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
    cloud_percent = float(np.isin(labels, [1, 2]).mean() * 100.0)
    return labels, cloud_percent

palette = np.array([
    [25, 25, 25],       # clear / black
    [215, 25, 32],      # thick cloud / red
    [245, 155, 45],     # thin cloud / amber
    [135, 135, 145],    # cloud shadow / grey
], dtype=np.uint8)

def make_mask(labels):
    return Image.fromarray(palette[labels])

def make_overlay(image, labels):
    src = np.asarray(image).astype(np.float32)
    clr = np.asarray(make_mask(labels)).astype(np.float32)
    alpha = np.where(labels == 0, 0.0, 0.43)[..., None]
    return Image.fromarray(np.clip(src * (1-alpha) + clr * alpha, 0, 255).astype(np.uint8))

def to_png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()

def make_result_zip(results):
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w", zipfile.ZIP_DEFLATED) as zf:
        report = []
        for i, item in enumerate(results, start=1):
            original_name = Path(item["name"]).name
            safe_name = f"{i:04d}_{Path(original_name).stem}"
            original_ext = Path(original_name).suffix.lower() or ".jpg"
            zf.writestr(f"{item['folder']}/{safe_name}{original_ext}", item["raw"])
            zf.writestr(f"masks/{safe_name}_mask.png", to_png(item["mask"]))
            zf.writestr(f"overlays/{safe_name}_overlay.png", to_png(item["overlay"]))
            report.append({
                "file": original_name,
                "cloud_percent": round(item["cloud"], 2),
                "threshold_percent": item["threshold"],
                "decision": item["decision"],
            })
        txt = io.StringIO()
        writer = csv.DictWriter(txt, fieldnames=[
            "file", "cloud_percent", "threshold_percent", "decision"
        ])
        writer.writeheader()
        writer.writerows(report)
        zf.writestr("report.csv", txt.getvalue())
    buff.seek(0)
    return buff.getvalue()

# Header with the actual logo image (PNG asset served from the award site).
st.markdown(f"""
<div class="brand-head">
  <img src="{LOGO_URL}" alt="Логотип СПУТНИКС">
  <div class="brand-copy">
    <div class="eyebrow">SPUTNIX · EARTH OBSERVATION</div>
    <h1>CLOUD AI</h1>
    <p>Интеллектуальный контроль облачности спутниковых снимков</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ПАНЕЛЬ ОТБОРА")
    st.caption("Настройте порог и добавьте изображения.")
    threshold = st.slider("Допустимая облачность", 0, 100, 75, 1, format="%d%%")
    st.markdown("---")
    st.markdown("**ПРАВИЛО ОТБОРА**")
    st.markdown(f"✅ Принимать: **≤ {threshold}%**")
    st.markdown(f"⛔ Отклонять: **> {threshold}%**")
    st.markdown("---")
    st.markdown("**МОДЕЛЬ ИИ**")
    st.caption("CLIPSeg · CloudSEN12+")
    st.caption("Классы: чисто, плотные облака, тонкие облака, тени.")
    st.markdown("---")
    st.caption("Первый запуск скачивает модель автоматически. Нужен интернет.")

st.markdown('<div class="section-title">Рабочее пространство</div>', unsafe_allow_html=True)
step_cols = st.columns(4)
steps = [
    ("01", "Загрузите", "Выберите снимки с компьютера."),
    ("02", "Анализируйте", "ИИ найдёт облака и тени."),
    ("03", "Проверьте", "Посмотрите процент и маску."),
    ("04", "Скачайте", "Получите отбор и отчёт CSV."),
]
for col, (num, title, desc) in zip(step_cols, steps):
    col.markdown(
        f'<div class="step"><div class="step-num">{num} / SPUTNIX</div>'
        f'<div class="step-title">{title}</div><div class="muted">{desc}</div></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="section-title">1. Добавить спутниковые снимки</div>', unsafe_allow_html=True)
uploads = st.file_uploader(
    "Перетащите файлы сюда или нажмите «Browse files»",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
    help="Можно выбрать несколько PNG, JPG или TIFF. Очень большие наборы загружайте партиями."
)
if uploads:
    st.success(f"Добавлено файлов: {len(uploads)}")
    st.caption("Поддерживаются обычные RGB-изображения и TIFF с минимум тремя каналами.")
run_col, note_col = st.columns([1, 2])
with run_col:
    launch = st.button(
        "▶  НАЧАТЬ АНАЛИЗ",
        type="primary",
        disabled=not bool(uploads),
        use_container_width=True
    )
with note_col:
    st.markdown(
        '<div class="muted" style="padding:12px 4px;">'
        'Исходники остаются нетронутыми. Анализ выполняется на сервере SPUTNIX CLOUD AI.</div>',
        unsafe_allow_html=True
    )

if launch and uploads:
    try:
        with st.spinner("Загрузка ИИ-модели. При первом запуске скачивается около 603 МБ весов…"):
            processor, model, device = load_model()
        results = []
        progress = st.progress(0, text="Подготовка файлов…")
        for idx, upload in enumerate(uploads):
            try:
                raw = upload.getvalue()
                image = read_uploaded_image(upload)
                # Limit working resolution on very large images to avoid exhausting RAM.
                # The original file is preserved in the downloadable package.
                work = image.copy()
                work.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                labels_small, cloud_pct = run_segmentation(work, processor, model, device)
                if work.size != image.size:
                    labels = np.asarray(
                        Image.fromarray(labels_small).resize(image.size, Image.Resampling.NEAREST)
                    )
                else:
                    labels = labels_small
                decision = "ОТКЛОНЁН" if cloud_pct > threshold else "ПРИНЯТ"
                results.append({
                    "name": upload.name, "raw": raw, "image": image,
                    "cloud": cloud_pct, "decision": decision, "threshold": threshold,
                    "folder": "rejected" if decision == "ОТКЛОНЁН" else "accepted",
                    "mask": make_mask(labels), "overlay": make_overlay(image, labels),
                })
            except Exception as exc:
                st.error(f"Не удалось обработать {upload.name}: {exc}")
            progress.progress(
                (idx + 1) / len(uploads),
                text=f"Анализ: {idx + 1} из {len(uploads)}"
            )
        st.session_state["sputnix_results"] = results
        st.session_state["sputnix_threshold"] = threshold
        st.session_state["sputnix_errors"] = len(uploads) - len(results)
    except Exception as exc:
        st.error(f"Не удалось запустить модель: {exc}")
        st.info("Проверьте интернет-соединение и свободное место. Веса модели загружаются с Hugging Face.")

results = st.session_state.get("sputnix_results", [])
if results:
    st.markdown("---")
    st.markdown('<div class="section-title">2. Результаты обработки</div>', unsafe_allow_html=True)
    ok = sum(x["decision"] == "ПРИНЯТ" for x in results)
    bad = len(results) - ok
    average = float(np.mean([x["cloud"] for x in results]))
    kpis = [
        ("Обработано", str(len(results))),
        ("Принято", str(ok)),
        ("Отклонено", str(bad)),
        ("Средняя облачность", f"{average:.1f}%"),
    ]
    metric_cols = st.columns(4)
    for col, (label, value) in zip(metric_cols, kpis):
        col.markdown(
            f'<div class="card"><div class="card-accent"></div>'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value">{value}</div></div>',
            unsafe_allow_html=True
        )
    if st.session_state.get("sputnix_errors", 0):
        st.warning(f"Не обработано файлов: {st.session_state['sputnix_errors']}.")

    st.markdown('<div class="section-title">Список снимков</div>', unsafe_allow_html=True)
    table = pd.DataFrame([
        {
            "Файл": x["name"],
            "Облачность": f'{x["cloud"]:.2f}%',
            "Порог": f'{x["threshold"]}%',
            "Решение": x["decision"],
        } for x in results
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Откройте снимок для подробного просмотра",
        options=list(range(len(results))),
        format_func=lambda i: results[i]["name"]
    )
    item = results[selected]
    image_cols = st.columns(3)
    image_cols[0].image(item["image"], caption="Исходный снимок", use_container_width=True)
    image_cols[1].image(item["mask"], caption="Карта сегментации", use_container_width=True)
    image_cols[2].image(item["overlay"], caption="Облака на снимке", use_container_width=True)
    st.caption("Красный — плотные облака · Оранжевый — тонкие облака · Серый — тени · Чёрный — чистые пиксели")
    verdict_class = "decision-bad" if item["decision"] == "ОТКЛОНЁН" else "decision-good"
    st.markdown(
        f'<div class="card" style="height:auto;margin-top:8px;">'
        f'<div class="card-label">Решение системы</div>'
        f'<div class="card-value" style="font-size:24px;">'
        f'<span class="{verdict_class}">{item["decision"]}</span></div>'
        f'<div class="muted">Облачность: {item["cloud"]:.2f}% · Допустимый порог: {item["threshold"]}%</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("")
    st.download_button(
        "⬇  СКАЧАТЬ ОТЧЁТ И РЕЗУЛЬТАТЫ (ZIP)",
        data=make_result_zip(results),
        file_name="SPUTNIX_CLOUD_AI_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption("В ZIP: принятые и отклонённые оригиналы, маски, изображения с наложением и report.csv.")

st.markdown("---")
st.markdown(
    '<div class="muted">SPUTNIX CLOUD AI · Демонстрационный прототип. '
    'Модель обучена на Sentinel-2; её результаты на данных Sputnix нужно проверить отдельно. '
    'Для многоспектральных TIFF требуется подтвердить порядок каналов.</div>',
    unsafe_allow_html=True
)
