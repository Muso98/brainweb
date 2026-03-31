# tumor/nnunet_utils.py  (revamped)
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


# Config (muqim muhit parametrlarini os.environ orqali o'zgartiring)
NNUNET_RAW = os.environ.get("nnUNet_raw", "/home/muso/nnUNet_raw")
NNUNET_RESULTS = os.environ.get("nnUNet_results", "/home/muso/nnUNet_results")

DATASET_ID = int(os.environ.get("NNUNET_DATASET_ID", 2))
TRAINER = os.environ.get("NNUNET_TRAINER", "nnUNetTrainer")
PLANS = os.environ.get("NNUNET_PLANS", "nnUNetPlans")
CONFIG = os.environ.get("NNUNET_CONFIG", "3d_fullres")


# -------------------------
# Helpers
# -------------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _check_executable(name: str) -> bool:
    """System PATH-da executable bormi."""
    from shutil import which
    return which(name) is not None


# -------------------------
# nnU-Net wrapper
# -------------------------
def run_nnunet_on_nifti(input_nifti_path: str,
                       device: str = "cuda",
                       folds: Optional[List[int]] = None,
                       verbose: bool = False) -> str:
    """
    input_nifti_path -> nnUNet inference -> returns absolute path to predicted mask .nii.gz
    - device: 'cuda' yoki 'cpu' (agar GPU yo'q bo'lsa 'cpu' ga tushirish foydali)
    - folds: ro'yxat (masalan [0,1,2,3,4]) yoki None (default behaviour)
    """
    if not os.path.exists(input_nifti_path):
        raise FileNotFoundError(f"Input NIfTI not found: {input_nifti_path}")

    if not _check_executable("nnUNetv2_predict"):
        raise RuntimeError("nnUNetv2_predict not found in PATH. Iltimos nnU-Net environment ni yoqing.")

    case_id = uuid.uuid4().hex[:8]
    input_dir = Path(NNUNET_RESULTS) / "Dataset" / f"tmp_inputs_{case_id}"
    output_dir = Path(NNUNET_RESULTS) / "Dataset" / "inference" / case_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # nnU-Net input formati: bir nechta kanal fayllari. Agar sizda bitta NIfTI bo'lsa, uni
    # nnU-Net kutgan nomlash formatiga nusxa qilamiz (caseid_0000.nii.gz ...).
    for m in range(4):
        dst = input_dir / f"{case_id}_{m:04d}.nii.gz"
        shutil.copy(input_nifti_path, dst)

    cmd = [
        "nnUNetv2_predict",
        "-d", str(DATASET_ID),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-tr", TRAINER,
        "-c", CONFIG,
        "-p", PLANS,
    ]

    if folds:
        # folds may be like [0,1,2,3,4] -> pass as strings
        cmd += ["-f"] + [str(int(f)) for f in folds]

    # device flag - nnUNetv2 may accept CUDA_VISIBLE_DEVICES env instead; attempt with -device
    if device.startswith("cuda"):
        cmd += ["-device", "cuda"]
    else:
        cmd += ["-device", "cpu"]

    if verbose:
        print("Running nnUNetv2_predict:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # include stderr for debugging
        stderr = e.stderr.decode(errors="ignore") if hasattr(e, "stderr") and e.stderr else str(e)
        raise RuntimeError(f"nnUNetv2_predict failed: {stderr}")

    result_mask = output_dir / f"{case_id}.nii.gz"
    if not result_mask.exists():
        # maybe output uses different naming -> search
        candidates = list(output_dir.glob("*.nii*"))
        if not candidates:
            raise RuntimeError(f"nnU-Net produced no output in {output_dir}")
        # choose first candidate
        result_mask = candidates[0]

    return str(result_mask.resolve())


# -------------------------
# NIfTI -> PNG helpers
# -------------------------
def get_voxel_zooms(nifti_path: str) -> Tuple[float, float, float]:
    import nibabel as nib
    img = nib.load(nifti_path)
    return tuple(map(float, img.header.get_zooms()[:3]))


def nifti_to_png(nifti_path: str, png_path: str, z_index: Optional[int] = None,
                 window: Optional[Tuple[float, float]] = None) -> str:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    img = nib.load(nifti_path)
    data = img.get_fdata()
    if data.ndim < 3:
        raise RuntimeError("Expected 3D volume for nifti_to_png")

    Z = data.shape[2]
    if z_index is None:
        z = Z // 2
    else:
        z = int(max(0, min(Z - 1, z_index)))

    slice2d = data[:, :, z].astype(np.float32)

    # Choose window
    if window is None:
        p1, p99 = np.percentile(slice2d, [1, 99])
    else:
        p1, p99 = window

    clipped = np.clip(slice2d, p1, p99)
    norm = (clipped - p1) / (p99 - p1 + 1e-8)
    arr = (norm * 255).astype(np.uint8)

    png_dir = Path(png_path).parent
    png_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(png_path)
    return str(Path(png_path).resolve())


# -------------------------
# Overlay / bbox / gallery
# -------------------------
def make_bbox_overlay(mri_nifti_path: str, mask_nifti_path: str,
                      png_path: str, bbox_color: Tuple[int, int, int] = (255, 0, 0),
                      bbox_width: int = 3, expand_px: int = 3) -> str:
    import nibabel as nib
    import numpy as np
    from PIL import Image, ImageDraw
    mri_img = nib.load(mri_nifti_path).get_fdata()
    mask_img = nib.load(mask_nifti_path).get_fdata()

    tumor = mask_img > 0

    # ensure 3D
    if mri_img.ndim < 3 or tumor.ndim < 3:
        raise RuntimeError("Expected 3D NIfTI volumes")

    if not tumor.any():
        # save center slice grayscale
        png_dir = Path(png_path).parent
        png_dir.mkdir(parents=True, exist_ok=True)
        z = mri_img.shape[2] // 2
        slice2d = mri_img[:, :, z].astype(np.float32)
        p1, p99 = np.percentile(slice2d, [1, 99])
        arr = np.clip(slice2d, p1, p99)
        arr = ((arr - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(arr).convert("RGB").save(png_path)
        return str(Path(png_path).resolve())

    # find slice with largest tumor area
    slice_sums = tumor.sum(axis=(0, 1))
    z = int(slice_sums.argmax())

    slice2d = mri_img[:, :, z].astype(np.float32)
    mask2d = tumor[:, :, z]

    # normalize slice
    p1, p99 = np.percentile(slice2d, [1, 99])
    arr = np.clip(slice2d, p1, p99)
    arr = ((arr - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")

    ys, xs = np.where(mask2d)
    y_min, y_max = int(max(0, ys.min() - expand_px)), int(min(mask2d.shape[0] - 1, ys.max() + expand_px))
    x_min, x_max = int(max(0, xs.min() - expand_px)), int(min(mask2d.shape[1] - 1, xs.max() + expand_px))

    draw = ImageDraw.Draw(img)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=bbox_color, width=bbox_width)

    png_dir = Path(png_path).parent
    png_dir.mkdir(parents=True, exist_ok=True)
    img.save(png_path)
    return str(Path(png_path).resolve())


def generate_slice_gallery(nifti_path: str, mask_nifti_path: str, out_dir: str,
                           n_slices: int = 8, window: Optional[Tuple[float, float]] = None) -> List[str]:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    nii = nib.load(nifti_path)
    data = nii.get_fdata().astype(np.float32)
    mask = nib.load(mask_nifti_path).get_fdata() > 0.5

    if data.ndim != 3:
        return []

    X, Y, Z = data.shape
    # choose Z indices around tumor center (max mask sum)
    sums = mask.sum(axis=(0, 1))
    if sums.sum() == 0:
        indices = list(range(0, Z, max(1, Z // n_slices)))[:n_slices]
    else:
        center = int(sums.argmax())
        half = max(1, (n_slices // 2))
        start = max(0, center - half)
        end = min(Z, start + n_slices)
        indices = list(range(start, end))

    out_paths: List[str] = []
    for i, z in enumerate(indices):
        slice_img = data[:, :, z]
        mask2d = mask[:, :, z]

        # windowing
        if window is None:
            p1, p99 = np.percentile(slice_img, [1, 99])
        else:
            p1, p99 = window

        sl = np.clip(slice_img, p1, p99)
        sl = ((sl - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)
        rgb = np.stack([sl] * 3, axis=-1)

        # overlay red
        red = np.zeros_like(rgb); red[..., 0] = 255
        alpha = 0.5
        overlay = rgb.copy()
        overlay[mask2d] = (alpha * red[mask2d] + (1 - alpha) * overlay[mask2d]).astype(np.uint8)

        fname = f"study_gallery_{uuid.uuid4().hex[:8]}_{i}.png"
        full = out_base / fname
        Image.fromarray(overlay).save(str(full))
        out_paths.append(str(full.resolve()))

    return out_paths


# -------------------------
# Volume computation
# -------------------------
def compute_mask_volume_mm3(mask_nifti_path: str, largest_component_only: bool = True) -> Dict[str, Any]:
    import nibabel as nib
    import numpy as np
    nii = nib.load(mask_nifti_path)
    data = nii.get_fdata()
    voxel_zooms = tuple(map(float, nii.header.get_zooms()[:3]))
    voxel_vol = voxel_zooms[0] * voxel_zooms[1] * voxel_zooms[2]

    mask = (data > 0.5).astype(np.uint8)
    if mask.sum() == 0:
        return {
            "voxel_count": 0,
            "voxel_volume_mm3": voxel_vol,
            "tumor_volume_mm3": 0.0,
            "tumor_volume_cm3": 0.0,
            "bbox": None,
        }

    if largest_component_only:
        from scipy.ndimage import label
        labeled, n = label(mask)
        if n > 1:
            sizes = np.bincount(labeled.flatten())
            sizes[0] = 0
            largest_label = int(np.argmax(sizes))
            mask = (labeled == largest_label).astype(np.uint8)

    voxel_count = int(mask.sum())
    vol_mm3 = voxel_count * voxel_vol
    vol_cm3 = vol_mm3 / 1000.0

    coords = np.array(np.where(mask))
    if coords.size == 0:
        bbox = None
    else:
        mins = coords.min(axis=1)
        maxs = coords.max(axis=1)
        bbox = {
            "x_min": int(mins[0]), "y_min": int(mins[1]), "z_min": int(mins[2]),
            "x_max": int(maxs[0]), "y_max": int(maxs[1]), "z_max": int(maxs[2]),
        }

    return {
        "voxel_count": voxel_count,
        "voxel_volume_mm3": voxel_vol,
        "tumor_volume_mm3": float(vol_mm3),
        "tumor_volume_cm3": float(vol_cm3),
        "bbox": bbox,
        "voxel_zooms": voxel_zooms,
    }
