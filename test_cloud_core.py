import io
import zipfile

import numpy as np
from PIL import Image

from cloud_core import cloud_percentage, expand_uploads, build_results_zip


def test_cloud_percentage_exact_known_values():
    labels = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 3],
        [0, 0, 0, 3],
        [0, 1, 0, 0],
    ], dtype=np.uint8)
    # 3 thick + 2 thin = 5 cloudy pixels out of 16 = 31.25%.
    assert cloud_percentage(labels) == 31.25
    # Adding 2 shadow pixels gives 7/16 = 43.75%.
    assert cloud_percentage(labels, include_shadow=True) == 43.75


def test_cloud_percentage_boundaries():
    assert cloud_percentage(np.zeros((10, 10), dtype=np.uint8)) == 0.0
    assert cloud_percentage(np.ones((10, 10), dtype=np.uint8)) == 100.0
    assert cloud_percentage(np.full((2, 2), 2, dtype=np.uint8)) == 100.0


def test_zip_upload_expands_images_only_without_extracting_paths():
    buffer = io.BytesIO()
    image_buffer = io.BytesIO()
    Image.new("RGB", (3, 3), "white").save(image_buffer, format="PNG")
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("photos/scene_a.png", image_buffer.getvalue())
        archive.writestr("notes.txt", "not an image")
        archive.writestr("../../outside.jpg", image_buffer.getvalue())
    records, warnings = expand_uploads([("batch.zip", buffer.getvalue())])
    assert len(records) == 2
    assert {record["name"] for record in records} == {"scene_a.png", "outside.jpg"}
    assert all(record["raw"].startswith(b"\x89PNG") for record in records)


def test_result_zip_contains_report_mask_and_original():
    original = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(original, format="PNG")
    mask = Image.new("RGB", (2, 2), "red")
    result = {
        "name": "scene.png", "raw": original.getvalue(), "mask": mask,
        "overlay": mask, "cloud": 25.0, "threshold": 75, "decision": "ПРИНЯТ"
    }
    output = build_results_zip([result])
    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        names = set(archive.namelist())
        assert "accepted/0001_scene.png" in names
        assert "masks/0001_scene_mask.png" in names
        assert "overlays/0001_scene_overlay.png" in names
        assert "report.csv" in names
