"""Small, testable utilities for SPUTNIX CLOUD AI."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import PurePosixPath, Path
from typing import Iterable

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MAX_ARCHIVE_FILES = 200
MAX_ARCHIVE_TOTAL_UNCOMPRESSED = 250 * 1024 * 1024
MAX_SINGLE_IMAGE_SIZE = 60 * 1024 * 1024


def cloud_percentage(labels: np.ndarray, include_shadow: bool = False) -> float:
    """Cloud percentage from a 4-class mask: 0 clear, 1 thick, 2 thin, 3 shadow."""
    arr = np.asarray(labels)
    if arr.size == 0:
        raise ValueError("Пустая карта классов — невозможно рассчитать облачность.")
    cloud = (arr == 1) | (arr == 2)
    if include_shadow:
        cloud |= arr == 3
    return float(np.count_nonzero(cloud) * 100.0 / arr.size)


def expand_uploads(uploaded: Iterable[tuple[str, bytes]]) -> tuple[list[dict], list[str]]:
    """Return image records from direct images and ZIP files without extracting paths."""
    records: list[dict] = []
    warnings: list[str] = []
    total_uncompressed = 0

    def add_image(name: str, raw: bytes) -> None:
        nonlocal records
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            return
        if len(raw) > MAX_SINGLE_IMAGE_SIZE:
            warnings.append(f"{name}: файл больше 60 МБ и пропущен.")
            return
        safe_name = Path(name.replace("\\", "/")).name or "satellite_image"
        records.append({"name": safe_name, "raw": raw})

    for name, raw in uploaded:
        ext = Path(name).suffix.lower()
        if ext != ".zip":
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                add_image(name, raw)
            else:
                warnings.append(f"{name}: неподдерживаемый формат, пропущен.")
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                valid_infos = []
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    member_ext = PurePosixPath(info.filename).suffix.lower()
                    if member_ext not in SUPPORTED_IMAGE_EXTENSIONS:
                        continue
                    # Never extract to disk. Reject suspicious or oversized archive members.
                    if info.file_size > MAX_SINGLE_IMAGE_SIZE:
                        warnings.append(f"{info.filename}: файл внутри ZIP больше 60 МБ, пропущен.")
                        continue
                    total_uncompressed += info.file_size
                    if total_uncompressed > MAX_ARCHIVE_TOTAL_UNCOMPRESSED:
                        raise ValueError("Суммарный распакованный объём ZIP превышает 250 МБ.")
                    valid_infos.append(info)

                if len(records) + len(valid_infos) > MAX_ARCHIVE_FILES:
                    raise ValueError("В наборе больше 200 изображений. Разбейте его на несколько загрузок.")

                for info in valid_infos:
                    try:
                        add_image(info.filename, archive.read(info))
                    except (RuntimeError, zipfile.BadZipFile) as exc:
                        warnings.append(f"{info.filename}: не удалось прочитать ({exc}).")
        except zipfile.BadZipFile:
            warnings.append(f"{name}: ZIP-архив повреждён или имеет неверный формат.")
        except Exception as exc:
            warnings.append(f"{name}: {exc}")

    if len(records) > MAX_ARCHIVE_FILES:
        warnings.append("В наборе больше 200 изображений; оставлены первые 200.")
        records = records[:MAX_ARCHIVE_FILES]
    return records, warnings


def encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_results_zip(results: list[dict]) -> bytes:
    """Build a ZIP containing originals, masks, overlays and a CSV summary."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        rows = []
        for index, item in enumerate(results, 1):
            stem = Path(item["name"]).stem[:80] or f"image_{index}"
            base = f"{index:04d}_{stem}"
            original_ext = Path(item["name"]).suffix.lower() or ".bin"
            folder = "accepted" if item["decision"] == "ПРИНЯТ" else "rejected"
            zf.writestr(f"{folder}/{base}{original_ext}", item["raw"])
            zf.writestr(f"masks/{base}_mask.png", encode_png(item["mask"]))
            zf.writestr(f"overlays/{base}_overlay.png", encode_png(item["overlay"]))
            rows.append({
                "file": item["name"],
                "cloud_percent": f"{item['cloud']:.2f}",
                "threshold_percent": item["threshold"],
                "decision": item["decision"],
            })
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=["file", "cloud_percent", "threshold_percent", "decision"])
        writer.writeheader()
        writer.writerows(rows)
        zf.writestr("report.csv", text.getvalue().encode("utf-8-sig"))
        zf.writestr(
            "README.txt",
            "SPUTNIX CLOUD AI results\n"
            "accepted/ — images within the selected cloud threshold\n"
            "rejected/ — images above the selected cloud threshold\n"
            "masks/ — predicted class masks (black clear, red thick cloud, orange thin cloud, grey shadow)\n"
            "overlays/ — masks overlaid on the analyzed image\n"
            "report.csv — cloud percentage and decision per image\n",
        )
    return out.getvalue()
