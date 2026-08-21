"""
Inpainting engine for generating Clean Plate backgrounds after character extraction.
"""
import numpy as np
from PIL import Image
import cv2


def generate_clean_plate(
    image: Image.Image,
    character_mask: np.ndarray,
    dilate_pixels: int = 7,
    inpaint_radius: int = 5,
    method: str = "telea",
) -> Image.Image:
    """Remove extracted characters from background by inpainting the masked region."""
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    if dilate_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels, dilate_pixels))
        inpaint_mask = cv2.dilate(character_mask, kernel, iterations=1)
    else:
        inpaint_mask = character_mask.copy()

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    inpainted_bgr = cv2.inpaint(img_bgr, inpaint_mask, inpaint_radius, flag)

    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)
