# tumor/tasks.py
import os

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist

from .models import Study, AIResult
from . import ai


@shared_task
def process_study(study_id: int):
    """
    Background task:
    - Study ni oladi
    - preprocess -> nnUNet -> metrics -> overlay -> multi-slice stack -> classification
    - AIResult yaratadi/yangi qiladi
    - Study.status ni update qiladi
    """
    try:
        study = Study.objects.get(pk=study_id)
    except ObjectDoesNotExist:
        return f"Study {study_id} not found"

    try:
        with transaction.atomic():
            # 0) Status: processing
            study.status = "processing"
            study.error_message = ""
            study.save(update_fields=["status", "error_message"])

            # 1) Preprocess -> NIfTI
            uploaded_abs_path = study.uploaded_file.path
            preprocessed_nifti_abs = ai.preprocess_to_nifti(uploaded_abs_path)

            nifti_rel = os.path.relpath(preprocessed_nifti_abs, settings.MEDIA_ROOT)
            study.nifti_file.name = nifti_rel
            study.save(update_fields=["nifti_file"])

            # 2) Segmentation (nnU-Net)
            mask_abs_path = ai.run_segmentation(preprocessed_nifti_abs)
            mask_rel = os.path.relpath(mask_abs_path, settings.MEDIA_ROOT)

            # 3) Metrics (volume, bbox, diameter)
            metrics = ai.compute_bbox_and_metrics(mask_abs_path)
            bbox = metrics["bbox"]

            # 4) Bitta preview overlay PNG
            overlay_rel_path = ai.generate_overlay_png(
                preprocessed_nifti_abs,
                mask_abs_path,
            )

            # 4b) Multi-slice overlay stack (viewer uchun)
            # Fayllar MEDIA_ROOT/results/stacks/ ichiga yoziladi
            ai.generate_overlay_stack(
                preprocessed_nifti_abs,
                mask_abs_path,
                study_id=study.id,
                num_slices=12,
            )

            # 5) Classification (agar model bo'lsa)
            # classify_tumor() model bo'lmasa ("", None) qaytaradigan qilib yozilgan
            predicted_class, predicted_conf = ai.classify_tumor(
                preprocessed_nifti_abs
            )

            # 6) AIResult: eski bo'lsa o'chirib, yangidan yaratamiz
            AIResult.objects.filter(study=study).delete()

            ai_result = AIResult.objects.create(
                study=study,
                tumor_volume_mm3=metrics["tumor_volume_mm3"],
                tumor_max_diameter_mm=metrics["tumor_max_diameter_mm"],
                bbox_x_min=bbox["x_min"],
                bbox_y_min=bbox["y_min"],
                bbox_z_min=bbox["z_min"],
                bbox_x_max=bbox["x_max"],
                bbox_y_max=bbox["y_max"],
                bbox_z_max=bbox["z_max"],
                predicted_class=predicted_class or "",
                predicted_confidence=predicted_conf,
            )

            ai_result.mask_nifti.name = mask_rel
            if overlay_rel_path:
                ai_result.bbox_png.name = overlay_rel_path
            ai_result.save()

            # 7) Status: done
            study.status = "done"
            study.save(update_fields=["status"])

        return f"Study {study_id} processed successfully"

    except Exception as e:
        study.status = "error"
        study.error_message = str(e)
        study.save(update_fields=["status", "error_message"])
        return f"Error processing study {study_id}: {e}"
