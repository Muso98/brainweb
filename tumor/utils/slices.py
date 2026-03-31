# tumor/utils/slices.py
import os
from typing import List, Optional
import uuid

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from django.conf import settings


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _rescale_to_uint8(slice_img: np.ndarray, window: Optional[tuple] = None) -> np.ndarray:
    if window:
        mn, mx = window
    else:
        mn, mx = np.percentile(slice_img, [1, 99])
    clipped = np.clip(slice_img, mn, mx)
    norm = (clipped - mn) / (mx - mn + 1e-8)
    uint8 = (norm * 255.0).astype(np.uint8)
    return uint8


def _draw_contours_on_rgb(rgb: np.ndarray, mask_slice: np.ndarray, color=(255, 0, 0), width=2):
    """
    Draw contour lines from binary mask on top of rgb image using PIL.
    rgb: (H,W,3) uint8
    mask_slice: (H,W) bool or 0/1
    """
    H, W = mask_slice.shape
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    # find contours with a simple marching squares using numpy diff (lightweight)
    # fallback: draw mask boundary pixels
    ys, xs = np.where(mask_slice)
    if len(xs) == 0:
        return pil
    # bounding box of mask, then draw polygon from mask edge via marching along outer perimeter
    # simple perimeter extraction:
    from skimage import measure
    contours = measure.find_contours(mask_slice.astype(np.uint8), 0.5)
    for c in contours:
        # c is array of (row, col) floats -> convert to list of tuples (x,y)
        pts = [(float(p[1]), float(p[0])) for p in c]
        if len(pts) > 2:
            draw.line(pts, fill=color, width=width)
    return pil


def generate_slice_gallery(
    nifti_path: str,
    mask_path: str,
    out_dir: Optional[str] = None,
    n_slices: int = 8,
    window: Optional[tuple] = None,
) -> List[str]:
    """
    Generate gallery of slice PNGs (orig + overlay) and return list of relative paths (MEDIA_ROOT relative).

    - nifti_path, mask_path: absolute paths to files
    - out_dir: optional absolute directory to save (if None -> MEDIA_ROOT/results/stacks/<uuid>/)
    - returns: list of relative paths (to settings.MEDIA_ROOT)
    """
    # Prepare out dir
    if out_dir is None:
        base = os.path.join(settings.MEDIA_ROOT, "results", "stacks", uuid.uuid4().hex)
    else:
        base = out_dir
    _ensure_dir(base)
    orig_dir = os.path.join(base, "orig")
    overlay_dir = os.path.join(base, "overlay")
    _ensure_dir(orig_dir)
    _ensure_dir(overlay_dir)

    # Load volumes
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    mask = nib.load(mask_path).get_fdata() > 0.5

    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D NIfTI, got shape {data.shape}")

    X, Y, Z = data.shape
    # choose slice indices centered around tumor region if mask exists
    mask_sums = mask.sum(axis=(0, 1))
    mask_idx = np.where(mask_sums > 0)[0]
    if mask_idx.size > 0:
        start, end = int(mask_idx.min()), int(mask_idx.max())
        if end - start + 1 <= n_slices:
            indices = list(range(start, end + 1))
        else:
            indices = np.linspace(start, end, n_slices, dtype=int).tolist()
    else:
        # fallback: evenly spaced through central volume
        indices = np.linspace(Z // 4, 3 * Z // 4, n_slices, dtype=int).tolist()

    rel_paths: List[str] = []
    for i, z in enumerate(indices):
        slice_img = data[:, :, z]
        mask_slice = mask[:, :, z].astype(bool)

        uint8 = _rescale_to_uint8(slice_img, window=window)
        rgb = np.stack([uint8] * 3, axis=-1)

        # save orig
        orig_fname = f"slice_orig_{i}.png"
        orig_full = os.path.join(orig_dir, orig_fname)
        Image.fromarray(rgb).save(orig_full)

        # overlay
        try:
            pil_overlay = _draw_contours_on_rgb(rgb, mask_slice, color=(255, 0, 0), width=2)
        except Exception:
            # fallback: just save rgb
            pil_overlay = Image.fromarray(rgb)

        overlay_fname = f"slice_overlay_{i}.png"
        overlay_full = os.path.join(overlay_dir, overlay_fname)
        pil_overlay.save(overlay_full)

        # return overlay paths (preferred for PDF)
        rel = os.path.relpath(overlay_full, settings.MEDIA_ROOT)
        rel_paths.append(rel)

    return rel_paths
