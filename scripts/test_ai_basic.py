#!/usr/bin/env python3
# scripts/test_ai_basic.py
import os
import sys

# --- Ensure project root is on sys.path so "import brainweb" works ---
# Compute project root as one level up from this scripts/ dir:
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # /home/muso/brainweb
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now set Django settings and initialize
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brainweb.settings")

import django
django.setup()

from tumor import ai

# --- UPDATE THESE PATHS to real local test files (or use dummy generator below) ---
SAMPLE_NIFTI = "/home/muso/test_data/sample_image.nii.gz"
SAMPLE_MASK  = "/home/muso/test_data/sample_mask.nii.gz"
STUDY_ID     = 9999

def safe_run():
    print("\n--- Testing generate_overlay_png ---")
    try:
        o = ai.generate_overlay_png(SAMPLE_NIFTI, SAMPLE_MASK)
        print("overlay rel:", o)
    except Exception as e:
        print("overlay error:", e)

    print("\n--- Testing generate_tumor_mesh_json ---")
    try:
        m = ai.generate_tumor_mesh_json(SAMPLE_MASK, STUDY_ID)
        print("mesh json:", m)
    except Exception as e:
        print("mesh json error:", e)

    print("\n--- Testing generate_overlay_stack ---")
    try:
        orig, overlay = ai.generate_overlay_stack(SAMPLE_NIFTI, SAMPLE_MASK, STUDY_ID, num_slices=6)
        print("orig count:", len(orig), "overlay count:", len(overlay))
    except Exception as e:
        print("overlay stack error:", e)

    print("\n--- Testing predict_growth_and_simulate ---")
    try:
        res = ai.predict_growth_and_simulate(patient_id=1, current_volume_mm3=100000.0, history_volumes=[80000.0, 90000.0])
        print("predict:", res)
    except Exception as e:
        print("predict error:", e)

    print("\n--- Testing generate_gradcam_for_slices (may require model file) ---")
    try:
        gc = ai.generate_gradcam_for_slices(SAMPLE_NIFTI)
        print("gradcam rel:", gc)
    except Exception as e:
        print("gradcam error:", e)

    print("\n--- Testing generate_uncertainty_map ---")
    try:
        unc = ai.generate_uncertainty_map(SAMPLE_MASK, SAMPLE_NIFTI)
        print("uncertainty rel:", unc)
    except Exception as e:
        print("uncertainty error:", e)

# Optional: create small dummy NIfTI + mask if files missing
def make_dummy_nifti(nifti_path: str, mask_path: str):
    try:
        import nibabel as nib
        import numpy as np
        from nibabel import Nifti1Image
    except Exception as e:
        print("Need nibabel and numpy to create dummy NIfTI:", e)
        return False

    if os.path.exists(nifti_path) and os.path.exists(mask_path):
        print("Dummy not created — files already exist.")
        return True

    os.makedirs(os.path.dirname(nifti_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    # small volume (64x64x32)
    vol = np.zeros((64, 64, 32), dtype=np.float32)
    # add gaussian blob as synthetic tumor
    cx, cy, cz = 32, 32, 16
    for x in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            for z in range(vol.shape[2]):
                d2 = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                vol[x,y,z] = 1000.0 * np.exp(-d2/(2*8.0**2))
    affine = np.eye(4)
    img = Nifti1Image(vol, affine)
    nib.save(img, nifti_path)
    # create mask by threshold
    mask = (vol > 30).astype(np.uint8)
    mask_img = Nifti1Image(mask, affine)
    nib.save(mask_img, mask_path)
    print("Dummy NIfTI and mask created:", nifti_path, mask_path)
    return True

if __name__ == "__main__":
    # create dummy data if needed
    if not (os.path.exists(SAMPLE_NIFTI) and os.path.exists(SAMPLE_MASK)):
        print("Sample files not found — creating dummy NIfTI and mask under /home/muso/test_data/")
        ok = make_dummy_nifti(SAMPLE_NIFTI, SAMPLE_MASK)
        if not ok:
            print("Cannot create dummy files; please provide real NIfTI and mask files and rerun.")
            sys.exit(1)

    safe_run()
