"""
Unit and API integration tests for remove (Manga Character Separation & Animation Pipeline).
"""
import pytest
import os
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from src.core.segmenter import SAMSegmenter, create_transparent_png
from src.core.inpaint import generate_clean_plate
from src.core.pipeline import MangaSeparationPipeline, CharacterTarget
from apps.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_manga_panel(tmp_path):
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Character 1 (Gojo)
    draw.rectangle([50, 80, 220, 350], fill=(50, 50, 50))
    # Character 2 (Hanami)
    draw.rectangle([350, 100, 520, 360], fill=(80, 80, 80))

    panel_path = os.path.join(tmp_path, "sample_panel.png")
    img.save(panel_path)
    return panel_path


def test_segmenter_and_inpaint(sample_manga_panel):
    img = Image.open(sample_manga_panel)
    segmenter = SAMSegmenter()
    
    # Test bounding box segmentation
    mask = segmenter.segment(image=img, box=(50, 80, 220, 350))
    assert mask.shape == (400, 600)
    assert np.any(mask > 0)

    # Test transparent RGBA output
    rgba = create_transparent_png(img, mask)
    assert rgba.mode == "RGBA"

    # Test clean plate inpainting
    clean_bg = generate_clean_plate(img, mask)
    assert clean_bg.size == (600, 400)
    assert clean_bg.mode == "RGB"


def test_full_pipeline_run(sample_manga_panel, tmp_path):
    out_dir = os.path.join(tmp_path, "out")
    pipeline = MangaSeparationPipeline()
    targets = [
        CharacterTarget(name="gojo", box=(50, 80, 220, 350)),
        CharacterTarget(name="hanami", box=(350, 100, 520, 360))
    ]
    res = pipeline.process(sample_manga_panel, targets, out_dir, panel_id="c01")
    assert "gojo" in res.character_paths
    assert "hanami" in res.character_paths
    assert os.path.exists(res.character_paths["gojo"])
    assert os.path.exists(res.clean_plate_path)
    assert os.path.exists(res.manifest_path)


def test_api_health_and_demo(client):
    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["service"] == "remove-manga-separation-api"

    # Demo execution
    demo_res = client.post("/api/demo")
    assert demo_res.status_code == 200
    demo_data = demo_res.json()
    assert "gojo" in demo_data["characters"]
    assert "hanami" in demo_data["characters"]
    assert "clean_background" in demo_data

    # Web UI served
    ui_res = client.get("/")
    assert ui_res.status_code == 200
    assert "Remove" in ui_res.text
