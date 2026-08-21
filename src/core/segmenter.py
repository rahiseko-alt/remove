"""
Segmentation engine supporting SAM (Segment Anything Model) and smart fallbacks.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter
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
    SAM (Segment Anything Model) wrapper with automatic smart GrabCut fallback.
    """
    def __init__(self, model_type: str = "sam_vit_h", checkpoint_path: Optional[str] = None):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self._init_model()

    def _init_model(self):
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
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape

        if self.predictor is not None:
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

        # Smart GrabCut / Bounding Box fallback
        return self._smart_fallback_segment(img_np, points, point_labels, box)

    def _smart_fallback_segment(
        self,
        img_np: np.ndarray,
        points: Optional[List[Tuple[int, int]]],
        point_labels: Optional[List[int]],
        box: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        if box is not None:
            x1, y1, x2, y2 = box
            rect = (max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1))
            try:
                cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            except Exception:
                mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
        elif points:
            mask.fill(cv2.GC_BGD)
            for (px, py), label in zip(points, point_labels or [1]*len(points)):
                cv2.circle(mask, (px, py), 15, cv2.GC_FGD if label == 1 else cv2.GC_BGD, -1)
            all_x = [p[0] for p in points]
            all_y = [p[1] for p in points]
            min_x, max_x = max(0, min(all_x) - 50), min(w, max(all_x) + 50)
            min_y, max_y = max(0, min(all_y) - 50), min(h, max(all_y) + 50)
            sub_mask = mask[min_y:max_y, min_x:max_x]
            sub_mask[sub_mask == cv2.GC_BGD] = cv2.GC_PR_FGD
            try:
                cv2.grabCut(img_np, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            except Exception:
                mask[min_y:max_y, min_x:max_x] = cv2.GC_PR_FGD
        else:
            rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
            cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return final_mask


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
