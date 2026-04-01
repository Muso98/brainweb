# tumor/ai_groq.py
"""
Groq vision API orqali MRI/CT tahlili.
NIfTI -> 3 ta kesim (axial, coronal, sagittal) -> Groq Llama 4 Vision -> natija
"""
import os
import io
import re
import json
import base64
import logging
from typing import Optional, Dict, Any

import nibabel as nib
import numpy as np
from PIL import Image
from django.conf import settings

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "") or getattr(settings, "GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY topilmadi. settings yoki environment da sozlang.")
    return key


def _extract_slice_b64(data: np.ndarray, axis: int) -> Optional[str]:
    """NIfTI volume dan o'rtadagi kesimni ajratib, base64 PNG qaytaradi."""
    try:
        mid = data.shape[axis] // 2
        if axis == 0:
            sl = data[mid, :, :]
        elif axis == 1:
            sl = data[:, mid, :]
        else:
            sl = data[:, :, mid]

        mn, mx = sl.min(), sl.max()
        if mx > mn:
            sl_norm = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            sl_norm = np.zeros_like(sl, dtype=np.uint8)

        sl_norm = np.rot90(sl_norm)

        img = Image.fromarray(sl_norm, mode='L').convert('RGB')
        img = img.resize((320, 320), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error("Kesim ajratishda xato (axis=%s): %s", axis, e)
        return None


def save_slices_png(nifti_path: str, study_id: int) -> Optional[str]:
    """
    NIfTI dan 3 ta kesim (axial, coronal, sagittal) ni yonma-yon PNG ga saqlaydi.
    MEDIA_ROOT/results/groq_slices/study_{id}.png ga yozadi.
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)

        slices = []
        for axis in [2, 1, 0]:  # axial, coronal, sagittal
            mid = data.shape[axis] // 2
            if axis == 0:
                sl = data[mid, :, :]
            elif axis == 1:
                sl = data[:, mid, :]
            else:
                sl = data[:, :, mid]

            mn, mx = sl.min(), sl.max()
            if mx > mn:
                sl_norm = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                sl_norm = np.zeros_like(sl, dtype=np.uint8)

            sl_norm = np.rot90(sl_norm)
            slices.append(Image.fromarray(sl_norm, mode='L').convert('RGB').resize((256, 256), Image.LANCZOS))

        # 3 ta kesimni yonma-yon joylashtirish
        combined = Image.new('RGB', (768, 256), color=(0, 0, 0))
        labels = ['Axial', 'Coronal', 'Sagittal']
        for i, (sl_img, label) in enumerate(zip(slices, labels)):
            combined.paste(sl_img, (i * 256, 0))

        out_dir = os.path.join(settings.MEDIA_ROOT, "results", "groq_slices")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"study_{study_id}.png")
        combined.save(out_path, format='PNG')
        return out_path
    except Exception as e:
        logger.error("PNG saqlashda xato: %s", e)
        return None


def analyze_study(nifti_path: str, modality: str = "MRI") -> Dict[str, Any]:
    """
    NIfTI faylni Groq Llama 4 Vision bilan tahlil qiladi.
    Qaytariladi: dict (tumor_detected, predicted_class, findings, ...)
    """
    api_key = _get_api_key()

    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)

    # 3 ta kesim: axial(2), coronal(1), sagittal(0)
    slices_b64 = []
    for axis in [2, 1, 0]:
        b64 = _extract_slice_b64(data, axis)
        if b64:
            slices_b64.append(b64)

    if not slices_b64:
        raise RuntimeError("NIfTI fayldan kesimlar ajratib bo'lmadi.")

    axis_names = ['axial', 'coronal', 'sagittal']

    # Groq API uchun xabar tuzish
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert neuroradiology AI. Analyze these {modality} brain scan slices "
                f"({', '.join(axis_names[:len(slices_b64)])} views) for brain tumor detection.\n\n"
                "Respond ONLY with a valid JSON object, no extra text:\n"
                "{\n"
                '  "tumor_detected": true or false,\n'
                '  "confidence": 0.0 to 1.0,\n'
                '  "predicted_class": "glioma" or "meningioma" or "pituitary" or "no_tumor",\n'
                '  "tumor_volume_estimate_cm3": number or null,\n'
                '  "location": "brain region description" or null,\n'
                '  "severity": "none" or "low" or "moderate" or "high",\n'
                '  "findings": "2-3 sentences of radiological findings",\n'
                '  "recommendation": "1-2 sentences of clinical recommendation"\n'
                "}"
            )
        }
    ]

    for b64 in slices_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 600,
    }

    try:
        import requests as _requests
        resp = _requests.post(
            GROQ_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; BrainWebAI/1.0)",
            },
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Groq API xatosi {resp.status_code}: {resp.text[:400]}")
        resp_json = resp.json()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Groq API bilan aloqa xatosi: {e}")

    raw_text = resp_json["choices"][0]["message"]["content"].strip()
    logger.info("Groq raw response: %s", raw_text[:500])

    # JSON ajratish
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        result = json.loads(match.group()) if match else json.loads(raw_text)
    except Exception:
        result = {
            "tumor_detected": None,
            "confidence": None,
            "predicted_class": "unknown",
            "tumor_volume_estimate_cm3": None,
            "location": None,
            "severity": "unknown",
            "findings": raw_text[:500],
            "recommendation": "Qo'shimcha ko'rik tavsiya etiladi.",
        }

    result["groq_model"] = GROQ_VISION_MODEL
    result["slices_analyzed"] = len(slices_b64)
    return result
