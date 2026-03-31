# tumor/utils/qrgen.py
import os
import uuid
from typing import Optional

import qrcode
from django.conf import settings


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def generate_qr(url: str, out_path: Optional[str] = None, box_size: int = 4, border: int = 2) -> str:
    """
    Generate QR PNG for given URL and return relative path (to MEDIA_ROOT).
    If out_path is None -> MEDIA_ROOT/results/qr/<uuid>.png
    """
    base = os.path.join(settings.MEDIA_ROOT, "results", "qr")
    _ensure_dir(base)

    if out_path is None:
        fname = f"qr_{uuid.uuid4().hex}.png"
        out_path = os.path.join(base, fname)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = qrcode.make(url)
    img.save(out_path)
    rel = os.path.relpath(out_path, settings.MEDIA_ROOT)
    return rel
