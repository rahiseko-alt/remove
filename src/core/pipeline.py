"""
End-to-end Manga Character Separation & Export Pipeline
"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

from src.core.segmenter import SAMSegmenter, create_transparent_png
from src.core.inpaint import generate_clean_plate


@dataclass
class CharacterTarget:
    name: str
    points: Optional[List[Tuple[int, int]]] = None
    point_labels: Optional[List[int]] = None
    box: Optional[Tuple[int, int, int, int]] = None
    prompt: Optional[str] = None


@dataclass
class ExtractionResult:
    character_paths: Dict[str, str] = field(default_factory=dict)
    clean_plate_path: Optional[str] = None
    manifest_path: Optional[str] = None


class MangaSeparationPipeline:
    def __init__(self, segmenter: Optional[SAMSegmenter] = None):
        self.segmenter = segmenter or SAMSegmenter()

    def process(
        self,
        image_path: str,
        targets: List[CharacterTarget],
        output_dir: str,
        panel_id: str = "panel_01",
        generate_bg: bool = True,
        feather_radius: int = 1,
        crop_characters: bool = False,
    ) -> ExtractionResult:
        os.makedirs(output_dir, exist_ok=True)
        img = Image.open(image_path)
        w, h = img.size

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        char_paths = {}

        # 1. Extract each character
        for target in targets:
            mask = self.segmenter.segment(
                image=img,
                points=target.points,
                point_labels=target.point_labels,
                box=target.box,
                prompt=target.prompt,
            )
            combined_mask = np.bitwise_or(combined_mask, mask)

            char_rgba = create_transparent_png(
                image=img,
                mask=mask,
                feather_radius=feather_radius,
                crop_to_content=crop_characters,
            )
            out_filename = f"{panel_id}_char_{target.name}.png"
            out_path = os.path.join(output_dir, out_filename)
            char_rgba.save(out_path, format="PNG")
            char_paths[target.name] = out_path

        # 2. Inpaint background (Clean Plate)
        clean_bg_path = None
        if generate_bg and len(targets) > 0:
            clean_bg = generate_clean_plate(img, combined_mask)
            bg_filename = f"{panel_id}_bg_clean.png"
            clean_bg_path = os.path.join(output_dir, bg_filename)
            clean_bg.save(clean_bg_path, format="PNG")

        # 3. Output Manifest for Animation Tools (CTA5 / AE / Spine)
        manifest_data = {
            "panel_id": panel_id,
            "original_image": image_path,
            "resolution": {"width": w, "height": h},
            "background": clean_bg_path,
            "characters": [
                {
                    "name": target.name,
                    "file": char_paths.get(target.name),
                    "box": target.box,
                }
                for target in targets
            ],
            "tool_compatibility": [
                "Cartoon Animator 5 (G3 Free Bone)",
                "Adobe After Effects (Advanced Puppet Tool)",
                "Spine 2D (Mesh Skinning)"
            ]
        }
        manifest_path = os.path.join(output_dir, f"{panel_id}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        return ExtractionResult(
            character_paths=char_paths,
            clean_plate_path=clean_bg_path,
            manifest_path=manifest_path,
        )
