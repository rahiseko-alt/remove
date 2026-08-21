"""
Manga Character Separation & Transparent Export API (remove)
"""
import os
import uuid
import shutil
import json
from typing import List, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from src.core.pipeline import MangaSeparationPipeline, CharacterTarget

app = FastAPI(
    title="Remove - Manga Character Separation & Animation Pipeline",
    version="1.0.0",
    description="Extract characters from manga panels and export transparent layers for Cartoon Animator / AE / Spine."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = os.path.abspath("storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
OUTPUT_DIR = os.path.join(STORAGE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = MangaSeparationPipeline()


class CharacterSpec(BaseModel):
    name: str
    box: Optional[Tuple[int, int, int, int]] = None
    points: Optional[List[Tuple[int, int]]] = None
    prompt: Optional[str] = None


class ExtractionRequest(BaseModel):
    image_id: str
    characters: List[CharacterSpec]
    generate_bg: bool = True
    crop_characters: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "service": "remove-manga-separation-api"}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".png"
    image_id = str(uuid.uuid4())
    filename = f"{image_id}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with Image.open(dest_path) as img:
        width, height = img.size

    return {
        "image_id": image_id,
        "filename": filename,
        "width": width,
        "height": height,
        "url": f"/storage/uploads/{filename}"
    }


@app.post("/api/extract")
async def extract_characters(req: ExtractionRequest):
    matched = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(req.image_id)]
    if not matched:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = os.path.join(UPLOAD_DIR, matched[0])
    job_out_dir = os.path.join(OUTPUT_DIR, req.image_id)

    targets = [
        CharacterTarget(
            name=c.name,
            box=c.box,
            points=c.points,
            prompt=c.prompt,
        )
        for c in req.characters
    ]

    result = pipeline.process(
        image_path=image_path,
        targets=targets,
        output_dir=job_out_dir,
        panel_id="panel_01",
        generate_bg=req.generate_bg,
        crop_characters=req.crop_characters,
    )

    char_urls = {
        name: f"/storage/outputs/{req.image_id}/{os.path.basename(path)}"
        for name, path in result.character_paths.items()
    }
    bg_url = f"/storage/outputs/{req.image_id}/{os.path.basename(result.clean_plate_path)}" if result.clean_plate_path else None
    manifest_url = f"/storage/outputs/{req.image_id}/{os.path.basename(result.manifest_path)}"

    return {
        "image_id": req.image_id,
        "characters": char_urls,
        "clean_background": bg_url,
        "manifest": manifest_url,
        "tool_compatibility": ["Cartoon Animator 5", "Adobe After Effects", "Spine 2D"]
    }


@app.post("/api/demo")
async def run_demo():
    """Instant demo with synthetic manga panel."""
    demo_id = str(uuid.uuid4())
    job_out_dir = os.path.join(OUTPUT_DIR, demo_id)
    os.makedirs(job_out_dir, exist_ok=True)

    demo_img_path = os.path.join(job_out_dir, "sample_panel.jpg")
    from PIL import ImageDraw
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(0, 800, 40):
        draw.line([(x, 0), (400, 300)], fill=(220, 220, 220), width=1)
    draw.rectangle([60, 80, 320, 540], fill=(40, 40, 50), outline=(0, 0, 0), width=3)
    draw.ellipse([110, 40, 270, 150], fill=(240, 240, 240), outline=(0, 0, 0), width=2)
    draw.rectangle([460, 100, 740, 550], fill=(90, 90, 90), outline=(0, 0, 0), width=3)
    draw.ellipse([510, 50, 690, 170], fill=(130, 130, 130), outline=(0, 0, 0), width=2)
    img.save(demo_img_path)

    targets = [
        CharacterTarget(name="gojo", box=(60, 40, 320, 540)),
        CharacterTarget(name="hanami", box=(460, 50, 740, 550)),
    ]

    result = pipeline.process(
        image_path=demo_img_path,
        targets=targets,
        output_dir=job_out_dir,
        panel_id="sample",
        generate_bg=True,
    )

    return {
        "demo_id": demo_id,
        "sample_image": f"/storage/outputs/{demo_id}/sample_panel.jpg",
        "characters": {
            name: f"/storage/outputs/{demo_id}/{os.path.basename(p)}"
            for name, p in result.character_paths.items()
        },
        "clean_background": f"/storage/outputs/{demo_id}/{os.path.basename(result.clean_plate_path)}",
        "manifest": f"/storage/outputs/{demo_id}/{os.path.basename(result.manifest_path)}",
    }


app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

WEB_DIR = os.path.abspath("apps/web")
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
