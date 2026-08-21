"""
High-Precision Manga Character Segmentation Engine.
Integrates SAM, Rembg (U2-Net / BiRefNet), and Smart Manga Edge Analysis.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        points: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Generate binary mask (2D numpy uint8 array, 0 or 255)."""
        pass


class SAMSegmenter(BaseSegmenter):
    """
    High-precision character segmenter combining Bounding Box Cropping + Rembg AI / GrabCut.
    """
    def __init__(self, model_type: str = "sam_vit_h", checkpoint_path: Optional[str] = None):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self._init_sam()

    def _init_sam(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor
            if self.checkpoint_path:
                sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                self.predictor = SamPredictor(sam)
        except Exception:
            self.predictor = None

    def segment(
        self,
        image: Image.Image,
        points: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. If SAM predictor is available, use it directly
        if self.predictor is not None:
            img_np = np.array(img_rgb)
            self.predictor.set_image(img_np)
            input_points = np.array(points) if points else None
            input_labels = np.array(point_labels) if point_labels else None
            input_box = np.array(box) if box else None

            masks, _, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=False,
            )
            return (masks[0] * 255).astype(np.uint8)

        # 2. Scoped AI Background Removal via Rembg / Manga Alpha
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop_img = img_rgb.crop((x1, y1, x2, y2))
                crop_mask = self._extract_character_alpha(crop_img)
                full_mask[y1:y2, x1:x2] = crop_mask
                return full_mask

        # 3. Fallback for whole image
        return self._extract_character_alpha(img_rgb)

    def _extract_character_alpha(self, cropped_pil: Image.Image) -> np.ndarray:
        """Extract high-precision alpha mask from character region."""
        # Try Rembg AI model first
        try:
            import rembg
            nobg_pil = rembg.remove(cropped_pil)
            alpha = np.array(nobg_pil.split()[-1])
            # Threshold alpha to solid 0/255 mask
            mask = np.where(alpha > 15, 255, 0).astype(np.uint8)
            return mask
        except Exception:
            pass

        # Smart Line-art & GrabCut Manga Fallback
        img_np = np.array(cropped_pil)
        h, w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Auto threshold for manga line art & tones (remove pure white page background)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Morphological closing to solidate character body
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find largest contours (main character body)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, -1)
            # Combine with line details
            mask = np.bitwise_and(mask, closed)
            mask = np.where(closed > 0, 255, 0).astype(np.uint8)
        else:
            mask = closed

        return mask


def create_transparent_png(
    image: Image.Image,
    mask: np.ndarray,
    feather_radius: int = 1,
    padding: int = 20,
    crop_to_content: bool = False,
) -> Image.Image:
    """Apply mask as alpha channel to image and return RGBA Image."""
    img_rgb = image.convert("RGB")
    mask_pil = Image.fromarray(mask, mode="L")

    if feather_radius > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    rgba = img_rgb.copy()
    rgba.putalpha(mask_pil)

    if crop_to_content:
        bbox = mask_pil.getbbox()
        if bbox:
            x1, y1, x2, y2 = bbox
            w, h = image.size
            nx1, ny1 = max(0, x1 - padding), max(0, y1 - padding)
            nx2, ny2 = min(w, x2 + padding), min(h, y2 + padding)
            rgba = rgba.crop((nx1, ny1, nx2, ny2))

    return rgba
