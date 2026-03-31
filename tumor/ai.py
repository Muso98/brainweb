# tumor/ai.py
import os
import json
import uuid
import shutil
import subprocess
import zipfile
import logging
from typing import Optional, Dict, Any, List, Tuple

from django.conf import settings
from .nnunet_utils import run_nnunet_on_nifti

logger = logging.getLogger(__name__)

# -----------------------
# Helpers
# -----------------------
def _ensure_media_subdir(subdir: str) -> str:
    base = getattr(settings, "MEDIA_ROOT", "/tmp")
    full = os.path.join(base, subdir)
    os.makedirs(full, exist_ok=True)
    return full

def _safe_relpath(fullpath: str) -> str:
    try:
        return os.path.relpath(fullpath, settings.MEDIA_ROOT)
    except Exception:
        return fullpath

def _save_nifti_from_data(data, ref_nii, out_path: str) -> str:
    import nibabel as nib
    import numpy as np
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    dtype = ref_nii.get_data_dtype()
    out_img = nib.Nifti1Image(data.astype(dtype), affine=ref_nii.affine, header=ref_nii.header)
    nib.save(out_img, out_path)
    return out_path

# -----------------------
# Preprocess
# -----------------------
def _normalize_and_save_nifti(nifti_input_path: str) -> str:
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
    out_name = f"{uuid.uuid4().hex}.nii.gz"
    out_path = os.path.join(out_dir, out_name)

    nib.save(new_img, out_path)
    return out_path


def _dicom_zip_to_nifti(zip_path: str) -> str:
    tmp_root = _ensure_media_subdir("dicom_tmp")
    tmp_dir = os.path.join(tmp_root, uuid.uuid4().hex)
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    nifti_raw_dir = _ensure_media_subdir("nifti_raw")

    cmd = [
        "dcm2niix",
        "-z", "y",
        "-o", nifti_raw_dir,
        "-f", "%p_%s",
        tmp_dir,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("dcm2niix not found. Please install dcm2niix and add it to PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"dcm2niix error: {e.stderr.decode(errors='ignore')}")

    candidates = [
        f
        for f in os.listdir(nifti_raw_dir)
        if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")
    ]
    if not candidates:
        raise RuntimeError("dcm2niix produced no NIfTI files.")

    nifti_filename = candidates[0]
    nifti_path = os.path.join(nifti_raw_dir, nifti_filename)
    return nifti_path


def preprocess_to_nifti(uploaded_path: str) -> str:
    lower = uploaded_path.lower()

    if lower.endswith(".zip"):
        nifti_raw_path = _dicom_zip_to_nifti(uploaded_path)
    elif lower.endswith(".nii") or lower.endswith(".nii.gz"):
        nifti_raw_path = uploaded_path
    else:
        raise ValueError(f"Unsupported file type for preprocess_to_nifti: {uploaded_path}")

    preprocessed_path = _normalize_and_save_nifti(nifti_raw_path)
    return preprocessed_path

# -----------------------
# Segmentation (nnU-Net wrapper)
# -----------------------
def run_segmentation(nifti_path: str) -> str:
    # nnUNetv2_predict mavjudligini AVVAL tekshiramiz (torch yuklamasdan oldin)
    from shutil import which
    if not which("nnUNetv2_predict"):
        raise RuntimeError(
            "AI segmentation modeli bu serverda mavjud emas. "
            "nnUNetv2_predict dasturi topilmadi. "
            "Model fayllarini serverga yuklab, nnU-Net ni sozlash kerak."
        )
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    raw_mask_path = run_nnunet_on_nifti(nifti_path, device=device)

    masks_dir = _ensure_media_subdir("results/masks")
    filename = os.path.basename(raw_mask_path)
    final_mask_path = os.path.join(masks_dir, filename)

    shutil.copy(raw_mask_path, final_mask_path)
    return final_mask_path

# -----------------------
# Basic metrics
# -----------------------
# -----------------------
# Basic metrics (better diameter)
# -----------------------
def compute_bbox_and_metrics(mask_nifti_path: str, use_largest_cc: bool = True, sample_for_diameter: int = 2000) -> Dict[str, Any]:
    import nibabel as nib
    import numpy as np
    import scipy.ndimage as ndi
    try:
        nii = nib.load(mask_nifti_path)
        data = nii.get_fdata()
        mask = (data > 0.5).astype(np.uint8)

        if mask.sum() == 0:
            return {
                "tumor_volume_mm3": None,
                "tumor_max_diameter_mm": None,
                "bbox_diag_mm": None,
                "bbox": {"x_min": None, "y_min": None, "z_min": None, "x_max": None, "y_max": None, "z_max": None},
            }

        # keep largest connected component if requested
        if use_largest_cc:
            labeled, n = ndi.label(mask)
            if n > 0:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                largest = int(np.argmax(sizes))
                mask = (labeled == largest).astype(np.uint8)

        # bbox in voxel indices
        x_idx, y_idx, z_idx = np.where(mask)
        x_min, x_max = int(x_idx.min()), int(x_idx.max())
        y_min, y_max = int(y_idx.min()), int(y_idx.max())
        z_min, z_max = int(z_idx.min()), int(z_idx.max())

        sx, sy, sz = map(float, nii.header.get_zooms()[:3])
        voxel_vol = sx * sy * sz
        voxels = int(mask.sum())
        tumor_volume_mm3 = float(voxels * voxel_vol)

        dx = (x_max - x_min + 1) * sx
        dy = (y_max - y_min + 1) * sy
        dz = (z_max - z_min + 1) * sz
        bbox_diag_mm = float(np.sqrt(dx**2 + dy**2 + dz**2))

        # pairwise (or sampled) diameter estimate
        approx_max_diam_mm = None
        coords = np.array(np.where(mask)).T  # (N,3)
        if coords.shape[0] > 0:
            coords_mm = coords.astype(float)
            coords_mm[:,0] *= sx; coords_mm[:,1] *= sy; coords_mm[:,2] *= sz
            N = coords_mm.shape[0]
            try:
                if N <= 3000:
                    from scipy.spatial.distance import cdist
                    d = cdist(coords_mm, coords_mm)
                    approx_max_diam_mm = float(d.max())
                else:
                    sample_n = min(sample_for_diameter, max(1000, int(N*0.1)))
                    idx = np.random.choice(N, size=sample_n, replace=False)
                    sample = coords_mm[idx]
                    from scipy.spatial.distance import cdist
                    d = cdist(sample, sample)
                    approx_max_diam_mm = float(d.max())
            except Exception:
                approx_max_diam_mm = None

        tumor_max_diameter_mm = approx_max_diam_mm if approx_max_diam_mm is not None else bbox_diag_mm

        return {
            "tumor_volume_mm3": tumor_volume_mm3,
            "tumor_max_diameter_mm": float(tumor_max_diameter_mm) if tumor_max_diameter_mm is not None else None,
            "bbox_diag_mm": bbox_diag_mm,
            "bbox": {"x_min": x_min, "y_min": y_min, "z_min": z_min, "x_max": x_max, "y_max": y_max, "z_max": z_max},
        }
    except Exception as e:
        logger.exception("compute_bbox_and_metrics failed: %s", e)
        return {
            "tumor_volume_mm3": None,
            "tumor_max_diameter_mm": None,
            "bbox_diag_mm": None,
            "bbox": {"x_min": None, "y_min": None, "z_min": None, "x_max": None, "y_max": None, "z_max": None},
        }

# -----------------------
# Overlay (single slice)
# -----------------------
def generate_overlay_png(nifti_path: str, mask_nifti_path: str) -> Optional[str]:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    try:
        img_nii = nib.load(nifti_path)
        mask_nii = nib.load(mask_nifti_path)

        data = img_nii.get_fdata()
        mask = mask_nii.get_fdata() > 0.5

        if data.ndim != 3:
            return None

        _, _, z_max = data.shape
        z_mid = z_max // 2

        slice_img = data[:, :, z_mid]
        slice_mask = mask[:, :, z_mid]

        p1, p99 = np.percentile(slice_img, [1, 99])
        slice_clipped = np.clip(slice_img, p1, p99)
        slice_norm = (slice_clipped - p1) / (p99 - p1 + 1e-8)
        slice_uint8 = (slice_norm * 255).astype(np.uint8)

        rgb = np.stack([slice_uint8] * 3, axis=-1)

        red = np.zeros_like(rgb)
        red[..., 0] = 255

        alpha = 0.5
        overlay = rgb.copy()
        overlay[slice_mask] = (
            alpha * red[slice_mask] + (1 - alpha) * overlay[slice_mask]
        ).astype(np.uint8)

        out_dir = _ensure_media_subdir("results/overlays")
        filename = f"{uuid.uuid4().hex}.png"
        full_path = os.path.join(out_dir, filename)

        im = Image.fromarray(overlay)
        im.save(full_path)

        rel_path = _safe_relpath(full_path)
        return rel_path
    except Exception as e:
        logger.exception("generate_overlay_png failed: %s", e)
        return None

# -----------------------
# Classifier (slice-based ResNet)
# -----------------------
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

_tumor_clf_model = None
_tumor_clf_classes: Optional[List[str]] = None
def _load_tumor_clf_model():
    global _tumor_clf_model, _tumor_clf_classes
    import torch
    import torch.nn as nn
    from torchvision import models

    if _tumor_clf_model is not None:
        return _tumor_clf_model, _tumor_clf_classes

    model_path = os.path.join(getattr(settings, "BASE_DIR", "."), "models", "tumor_clf.pt")
    if not os.path.exists(model_path):
        logger.info("Classifier model not found at %s", model_path)
        return None, None

    try:
        checkpoint = torch.load(model_path, map_location=_tumor_clf_device)
    except Exception as e:
        logger.exception("Failed to load classifier: %s", e)
        return None, None

    class_names = checkpoint.get("class_names") or checkpoint.get("classes") or []
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint

    try:
        base_model = models.resnet50(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = torch.nn.Linear(in_features, len(class_names) or 2)
        base_model.load_state_dict(state_dict)
        base_model.eval()
        base_model = base_model.to(_tumor_clf_device)

        _tumor_clf_model = base_model
        _tumor_clf_classes = class_names
        logger.info("Tumor classifier loaded. Classes: %s", class_names)
        return _tumor_clf_model, _tumor_clf_classes
    except Exception as e:
        logger.exception("Preparing classifier failed: %s", e)
        return None, None

def _get_resnet_eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(RESNET_MEAN, RESNET_STD),
    ])

def _nifti_to_resnet_slices(nifti_path: str, num_slices: int = 8):
    import nibabel as nib
    import numpy as np
    import torch
    from PIL import Image
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)

    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D volume for classification, got shape {data.shape}")

    _, _, Z = data.shape
    if Z < num_slices:
        step = 1
        num_slices = Z
    else:
        step = max(1, Z // num_slices)

    center = Z // 2
    half = (num_slices // 2) * step
    start = max(0, center - half)
    zs = list(range(start, min(Z, start + num_slices * step), step))

    transform = _get_resnet_eval_transform()
    tensors = []
    for z in zs:
        slice_img = data[:, :, z]
        p1, p99 = np.percentile(slice_img, [1, 99])
        slice_clipped = np.clip(slice_img, p1, p99)
        slice_norm = (slice_clipped - p1) / (p99 - p1 + 1e-8)
        slice_uint8 = (slice_norm * 255).astype(np.uint8)
        rgb = np.stack([slice_uint8] * 3, axis=-1)
        pil_img = Image.fromarray(rgb)
        tensors.append(transform(pil_img))

    if len(tensors) == 0:
        raise RuntimeError("No slices extracted from NIfTI for classification")

    batch = torch.stack(tensors, dim=0)
    return batch

def classify_tumor(nifti_path: str) -> Tuple[str, Optional[float]]:
    import torch
    import torch.nn.functional as F
    model, class_names = _load_tumor_clf_model()
    if model is None or class_names is None:
        return "", None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        slices_batch = _nifti_to_resnet_slices(nifti_path, num_slices=8)
        slices_batch = slices_batch.to(device)

        with torch.no_grad():
            logits = model(slices_batch)
            probs = F.softmax(logits, dim=1)
            mean_probs = probs.mean(dim=0)

        conf, idx = torch.max(mean_probs, dim=0)
        idx = int(idx.item())
        conf = float(conf.item())

        if 0 <= idx < len(class_names):
            pred_class = class_names[idx]
        else:
            pred_class = "unknown"

        return pred_class, conf
    except Exception as e:
        logger.exception("classify_tumor failed: %s", e)
        return "", None

# -----------------------
# Overlay stack (multiple slices)
# -----------------------
def generate_overlay_stack(nifti_path: str, mask_nifti_path: str, study_id: int, num_slices: int = 12) -> Tuple[List[str], List[str]]:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    img = nib.load(nifti_path)
    mask_img = nib.load(mask_nifti_path)

    data = img.get_fdata()
    mask = mask_img.get_fdata() > 0.5

    if data.ndim != 3:
        return [], []

    X, Y, Z = data.shape
    if Z < num_slices:
        step = 1
        num_slices = Z
    else:
        step = max(1, Z // num_slices)

    center = Z // 2
    half = (num_slices // 2) * step
    start = max(0, center - half)
    zs = list(range(start, min(Z, start + num_slices * step), step))

    stacks_base = _ensure_media_subdir("results/stacks")
    orig_dir = os.path.join(stacks_base, "orig")
    overlay_dir = os.path.join(stacks_base, "overlay")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    orig_rel_paths: List[str] = []
    overlay_rel_paths: List[str] = []

    for idx, z in enumerate(zs):
        slice_img = data[:, :, z]
        slice_mask = mask[:, :, z]

        p1, p99 = np.percentile(slice_img, [1, 99])
        slice_clipped = np.clip(slice_img, p1, p99)
        slice_norm = (slice_clipped - p1) / (p99 - p1 + 1e-8)
        slice_uint8 = (slice_norm * 255).astype(np.uint8)
        rgb = np.stack([slice_uint8] * 3, axis=-1)

        orig_filename = f"study{study_id}_orig_{idx}.png"
        orig_full = os.path.join(orig_dir, orig_filename)
        Image.fromarray(rgb).save(orig_full)
        orig_rel = _safe_relpath(orig_full)
        orig_rel_paths.append(orig_rel)

        red = np.zeros_like(rgb)
        red[..., 0] = 255
        alpha = 0.5
        overlay = rgb.copy()
        overlay[slice_mask] = (
            alpha * red[slice_mask] + (1 - alpha) * overlay[slice_mask]
        ).astype(np.uint8)

        overlay_filename = f"study{study_id}_overlay_{idx}.png"
        overlay_full = os.path.join(overlay_dir, overlay_filename)
        Image.fromarray(overlay).save(overlay_full)
        overlay_rel = _safe_relpath(overlay_full)
        overlay_rel_paths.append(overlay_rel)

    return orig_rel_paths, overlay_rel_paths

# -----------------------
# Tumor mesh (marching cubes -> JSON)
# -----------------------
def generate_tumor_mesh_json(mask_nifti_path: str, study_id: int) -> str:
    import nibabel as nib
    import numpy as np
    from skimage import measure
    nii = nib.load(mask_nifti_path)
    data = nii.get_fdata()

    verts, faces, normals, values = measure.marching_cubes(
        volume=data,
        level=0.5,
        spacing=nii.header.get_zooms()[:3],
    )

    affine = nii.affine
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_world = (affine @ verts_h.T).T[:, :3]

    mesh = {
        "vertices": verts_world.tolist(),
        "faces": faces.tolist(),
    }

    meshes_dir = _ensure_media_subdir("results/meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    out_path = os.path.join(meshes_dir, f"study{study_id}_tumor_mesh.json")
    with open(out_path, "w") as f:
        json.dump(mesh, f)

    return out_path

# -----------------------
# Growth prediction (simple baseline)
# -----------------------
def predict_growth_and_simulate(patient_id: int,
                                current_volume_mm3: float,
                                study_id: Optional[int] = None,
                                history_volumes: Optional[List[float]] = None,
                                days_for_pred: Optional[List[int]] = None,
                                treatment_effectiveness: float = 0.5) -> Dict[str, Any]:
    if days_for_pred is None:
        days_for_pred = [30, 90]

    try:
        vols: List[float] = []
        if history_volumes:
            vols = [v for v in history_volumes if v is not None and v > 0]
        vols.append(current_volume_mm3)

        if len(vols) >= 2:
            v_prev, v_curr = vols[-2], vols[-1]
            if v_prev > 0:
                dt = 30.0
                try:
                    r = (v_curr / v_prev) ** (1.0 / dt) - 1.0
                except Exception:
                    r = 0.0
            else:
                r = 0.0
        else:
            r = 0.002

        r = float(max(min(r, 0.2), -0.5))

        preds = {}
        for d in days_for_pred:
            pred = current_volume_mm3 * ((1.0 + r) ** d)
            preds[f"pred_vol_{d}_days"] = float(pred)

        pred30 = preds.get("pred_vol_30_days", current_volume_mm3)
        growth_rate_30_days = (pred30 - current_volume_mm3) / 30.0
        sim_effect = treatment_effectiveness
        sim_vol_post_treatment = float(current_volume_mm3 * (1.0 - sim_effect))

        return {
            "pred_vol_30_days": preds.get("pred_vol_30_days"),
            "pred_vol_90_days": preds.get("pred_vol_90_days"),
            "growth_rate_30_days": float(growth_rate_30_days),
            "sim_vol_post_treatment": float(sim_vol_post_treatment),
            "daily_growth_rate": float(r),
        }
    except Exception as e:
        logger.exception("predict_growth_and_simulate failed: %s", e)
        return {
            "pred_vol_30_days": None,
            "pred_vol_90_days": None,
            "growth_rate_30_days": None,
            "sim_vol_post_treatment": None,
            "daily_growth_rate": None,
        }

# -----------------------
# Grad-CAM helpers + generator (with robust resize)
# -----------------------
def _make_gradcam(model, target_layer):
    import torch
    activations = {}
    gradients = {}

    def save_activation(module, inp, out):
        activations['value'] = out.detach()

    def save_gradient(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    try:
        target_layer.register_full_backward_hook(save_gradient)
    except Exception:
        try:
            target_layer.register_backward_hook(save_gradient)
        except Exception:
            pass
    try:
        target_layer.register_forward_hook(save_activation)
    except Exception:
        pass

    def compute(input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        model.zero_grad()
        model.eval()
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
        input_tensor = input_tensor.to(device)
        out = model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1)[0].item()
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        act = activations.get('value')
        grad = gradients.get('value')
        if act is None or grad is None:
            raise RuntimeError("Grad-CAM hooks didn't capture activations/gradients")

        act = act[0]
        grad = grad[0]
        weights = torch.mean(grad.view(grad.size(0), -1), dim=1)

        cam = torch.zeros(act.shape[1:], dtype=torch.float32, device=act.device)
        for i, w in enumerate(weights):
            cam += w * act[i]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
        return cam_np

    return compute

def _resize_cam_to_shape(cam: np.ndarray, target_shape: Tuple[int,int]) -> np.ndarray:
    import numpy as np
    from PIL import Image
    Ht, Wt = target_shape
    try:
        from skimage.transform import resize as sk_resize
        resized = sk_resize(cam, (Ht, Wt), order=1, preserve_range=True, anti_aliasing=True)
        resized = np.clip(resized, 0.0, 1.0)
        return resized
    except Exception:
        try:
            cam_img = Image.fromarray((np.clip(cam,0,1)*255).astype(np.uint8))
            cam_img = cam_img.resize((Wt, Ht), resample=Image.BILINEAR)
            arr = np.asarray(cam_img).astype(np.float32) / 255.0
            return arr
        except Exception:
            # fallback nearest
            arr = np.zeros((Ht, Wt), dtype=float)
            sh0, sh1 = cam.shape
            for i in range(Ht):
                for j in range(Wt):
                    src_i = int(i * sh0 / Ht)
                    src_j = int(j * sh1 / Wt)
                    arr[i,j] = cam[min(src_i, sh0-1), min(src_j, sh1-1)]
            return arr

def generate_gradcam_for_slices(nifti_path: str,
                                cif_model: Optional[Any] = None,
                                out_dir: Optional[str] = None,
                                num_slices: int = 8) -> Optional[str]:
    import torch
    import torch.nn as nn
    import numpy as np
    import nibabel as nib
    from PIL import Image

    try:
        model, class_names = _load_tumor_clf_model()
        if model is None:
            return None

        # prefer deep-most conv block
        try:
            target_layer = model.layer4[-1]
        except Exception:
            # fallback to last conv found
            target_layer = None
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    target_layer = m

        if target_layer is None:
            logger.warning("No target layer found for Grad-CAM")
            return None

        gradcam_fn = _make_gradcam(model, target_layer)

        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)
        _, _, Z = data.shape
        num_slices = min(num_slices, Z)
        step = max(1, Z // num_slices)
        center = Z // 2
        half = (num_slices // 2) * step
        start = max(0, center - half)
        zs = list(range(start, min(Z, start + num_slices * step), step))

        device = _tumor_clf_device
        tensors = []
        pil_slices = []
        for z in zs:
            slice_img = data[:, :, z]
            p1, p99 = np.percentile(slice_img, [1, 99])
            slice_clipped = np.clip(slice_img, p1, p99)
            slice_norm = (slice_clipped - p1) / (p99 - p1 + 1e-8)
            slice_uint8 = (slice_norm * 255).astype(np.uint8)
            rgb = np.stack([slice_uint8] * 3, axis=-1)
            pil = Image.fromarray(rgb).convert("RGB")
            t = _resnet_eval_transform(pil).unsqueeze(0).to(device)
            tensors.append(t)
            pil_slices.append(pil)

        cams_list = []
        model.to(device)
        model.eval()
        for t in tensors:
            try:
                cam_np = gradcam_fn(t)
                cams_list.append(cam_np)
            except Exception as e:
                logger.exception("Grad-CAM per-slice failed: %s", e)

        if len(cams_list) == 0:
            return None

        cam_avg = np.mean(np.stack([np.array(c) for c in cams_list]), axis=0)

        center_idx = len(pil_slices)//2
        center_pil = pil_slices[center_idx]
        W, H = center_pil.size  # PIL (W,H)
        cam_resized = _resize_cam_to_shape(cam_avg, (H, W))
        cam_resized = cam_resized - cam_resized.min()
        cam_resized = cam_resized / (cam_resized.max() + 1e-8)

        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
        heat = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
        rgb = np.asarray(center_pil).astype(np.uint8)

        overlay = (0.6 * heat.astype(float) + 0.4 * rgb.astype(float)).astype(np.uint8)

        out_base = _ensure_media_subdir("results/gradcam")
        os.makedirs(out_base, exist_ok=True)
        fname = f"gradcam_study_{uuid.uuid4().hex}.png"
        full = os.path.join(out_base, fname)
        Image.fromarray(overlay).save(full)
        rel = _safe_relpath(full)
        return rel
    except Exception as e:
        logger.exception("generate_gradcam_for_slices failed: %s", e)
        return None

# -----------------------
# Uncertainty proxy map
# -----------------------
def generate_uncertainty_map(mask_nifti_path: str,
                             nifti_path: str,
                             out_path: Optional[str] = None,
                             n_samples: int = 8) -> Optional[str]:
    try:
        import nibabel as nib
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        mask_nii = nib.load(mask_nifti_path)
        mask = (mask_nii.get_fdata() > 0.5).astype(np.uint8)

        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask)
        inv_dist = distance_transform_edt(1 - mask)
        boundary = np.minimum(dist, inv_dist)
        # higher score near boundary
        score = (boundary.max() - boundary) / (boundary.max() + 1e-8)
        score2d = score.max(axis=2)
        score_img = (np.clip(score2d, 0, 1) * 255).astype(np.uint8)
        heat = plt.get_cmap("hot")(score_img / 255.0)[:, :, :3]
        heat_rgb = (heat * 255).astype(np.uint8)

        out_dir = _ensure_media_subdir("results/uncertainty")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"uncertainty_{uuid.uuid4().hex}.png"
        full = os.path.join(out_dir, fname)
        Image.fromarray(heat_rgb).save(full)
        rel = _safe_relpath(full)
        return rel
    except Exception as e:
        logger.exception("generate_uncertainty_map failed: %s", e)
        return None

# -----------------------
# New: compute volumes by label groups (core/enhancing/whole)
# -----------------------
def clean_mask_keep_labels(mask_nifti_path: str, labels_to_keep: List[int], out_path: Optional[str] = None, keep_largest_cc: bool = True) -> Optional[str]:
    import nibabel as nib
    import numpy as np
    import scipy.ndimage as ndi
    try:
        nii = nib.load(mask_nifti_path)
        m = nii.get_fdata().astype(int)
        keep = np.isin(m, labels_to_keep).astype(np.uint8)

        if keep_largest_cc and keep.sum() > 0:
            l, n = ndi.label(keep)
            if n > 0:
                sizes = np.bincount(l.ravel())
                sizes[0] = 0
                largest = int(np.argmax(sizes))
                keep = (l == largest).astype(np.uint8)

        if out_path is None:
            out_dir = os.path.join(os.path.dirname(mask_nifti_path), "cleaned")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}_clean.nii.gz")

        new_nii = nib.Nifti1Image(keep.astype(np.uint8), affine=nii.affine, header=nii.header)
        nib.save(new_nii, out_path)
        return out_path
    except Exception as e:
        logger.exception("clean_mask_keep_labels failed: %s", e)
        return None

def compute_volumes_by_label_groups(mask_nifti_path: str,
                                   label_groups: Optional[Dict[str, List[int]]] = None,
                                   keep_largest_cc: bool = True,
                                   save_filtered_masks_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    if label_groups is None:
        label_groups = {
            "whole_tumor": [1, 2, 4],
            "tumor_core": [1, 4],
            "enhancing": [4],
        }

    try:
        nii = nib.load(mask_nifti_path)
        m = nii.get_fdata().astype(int)
        sx, sy, sz = map(float, nii.header.get_zooms()[:3])
        voxel_vol = sx * sy * sz

        results: Dict[str, Dict[str, Any]] = {}
        for group_name, labels in label_groups.items():
            mask_bin = np.isin(m, labels).astype(np.uint8)
            if keep_largest_cc and mask_bin.sum() > 0:
                l, n = ndi.label(mask_bin)
                if n > 0:
                    sizes = np.bincount(l.ravel())
                    sizes[0] = 0
                    largest = int(np.argmax(sizes))
                    mask_bin = (l == largest).astype(np.uint8)

            vox = int(mask_bin.sum())
            vol_mm3 = float(vox * voxel_vol)
            vol_cm3 = vol_mm3 / 1000.0

            approx_max_diam_mm = None
            if vox > 0:
                coords = np.array(np.where(mask_bin)).T.astype(float)
                coords[:,0] *= sx; coords[:,1] *= sy; coords[:,2] *= sz
                if coords.shape[0] > 20000:
                    idx = np.random.choice(coords.shape[0], size=2000, replace=False)
                    sample = coords[idx]
                else:
                    sample = coords
                try:
                    from scipy.spatial.distance import cdist
                    d = cdist(sample, sample)
                    approx_max_diam_mm = float(d.max())
                except Exception:
                    mins = coords.min(axis=0)
                    maxs = coords.max(axis=0)
                    approx_max_diam_mm = float(np.linalg.norm(maxs - mins))

            saved_mask = None
            if save_filtered_masks_dir:
                os.makedirs(save_filtered_masks_dir, exist_ok=True)
                fname = f"{group_name}_mask_{uuid.uuid4().hex[:8]}.nii.gz"
                saved_mask = os.path.join(save_filtered_masks_dir, fname)
                _save_nifti_from_data(mask_bin, nii, saved_mask)

            results[group_name] = {
                "labels": labels,
                "voxels": vox,
                "volume_mm3": vol_mm3,
                "volume_cm3": vol_cm3,
                "approx_max_diameter_mm": approx_max_diam_mm,
                "saved_mask": saved_mask,
            }

        return results
    except Exception as e:
        logger.exception("compute_volumes_by_label_groups failed: %s", e)
        return {}

# -----------------------
# End of module
# -----------------------

if __name__ == "__main__":
    # Quick CLI test - run from project root: python tumor/ai.py /abs/path/mask.nii.gz
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tumor/ai.py /path/to/mask.nii.gz [--save-masks]")
        sys.exit(0)
    mask_p = sys.argv[1]
    save = "--save-masks" in sys.argv
    out_dir = None
    if save:
        out_dir = os.path.join(os.path.dirname(mask_p), "by_group")
        os.makedirs(out_dir, exist_ok=True)

    print("Computing groups (whole/core/enhancing)...")
    res = compute_volumes_by_label_groups(mask_p, save_filtered_masks_dir=out_dir)
    print(json.dumps(res, indent=2))
