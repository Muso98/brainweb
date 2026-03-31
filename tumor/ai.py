# tumor/ai.py
# Groq pipeline uchun faqat preprocess_to_nifti kerak.
# nnUNet, classifier, segmentation funksiyalari olib tashlandi.

import os
import uuid
import zipfile
import logging
import subprocess
from django.conf import settings

logger = logging.getLogger(__name__)


def _ensure_media_subdir(subdir: str) -> str:
    base = getattr(settings, "MEDIA_ROOT", "/tmp/media")
    full = os.path.join(base, subdir)
    os.makedirs(full, exist_ok=True)
    return full


def _normalize_and_save_nifti(nifti_input_path: str) -> str:
    """NIfTI faylni normalizatsiya qilib yangi fayl qaytaradi."""
    import nibabel as nib
    import numpy as np

    img = nib.load(nifti_input_path)
    data = img.get_fdata().astype(np.float32)

    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        data = (data - mean) / (std + 1e-8)
    else:
        data = data - mean

    new_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)

    out_dir = _ensure_media_subdir("nifti_preprocessed")
    out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.nii.gz")
    nib.save(new_img, out_path)
    return out_path


def _dicom_zip_to_nifti(zip_path: str) -> str:
    """DICOM zip -> NIfTI (dcm2niix kerak)."""
    tmp_root = _ensure_media_subdir("dicom_tmp")
    tmp_dir = os.path.join(tmp_root, uuid.uuid4().hex)
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    nifti_raw_dir = _ensure_media_subdir("nifti_raw")

    try:
        subprocess.run(
            ["dcm2niix", "-z", "y", "-o", nifti_raw_dir, "-f", "%p_%s", tmp_dir],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("dcm2niix topilmadi. DICOM zip fayllar qo'llab-quvvatlanmaydi.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"dcm2niix xatosi: {e.stderr.decode(errors='ignore')}")

    candidates = [
        f for f in os.listdir(nifti_raw_dir)
        if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")
    ]
    if not candidates:
        raise RuntimeError("dcm2niix NIfTI fayl yaratmadi.")
    return os.path.join(nifti_raw_dir, candidates[0])


def preprocess_to_nifti(uploaded_path: str) -> str:
    """
    Yuklangan faylni NIfTI ga o'tkazadi va normalizatsiya qiladi.
    .nii, .nii.gz -> normalizatsiya
    .zip (DICOM) -> dcm2niix -> normalizatsiya
    """
    lower = uploaded_path.lower()

    if lower.endswith(".zip"):
        nifti_raw_path = _dicom_zip_to_nifti(uploaded_path)
    elif lower.endswith(".nii") or lower.endswith(".nii.gz"):
        nifti_raw_path = uploaded_path
    else:
        raise ValueError(
            f"Qo'llab-quvvatlanmaydigan fayl turi: {os.path.basename(uploaded_path)}. "
            ".nii, .nii.gz yoki DICOM .zip yuklang."
        )

    preprocessed_path = _normalize_and_save_nifti(nifti_raw_path)
    return preprocessed_path
