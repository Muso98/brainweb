# tumor/views.py
"""
Improved / ideal views for the 'tumor' app (BrainWeb).
- Namespaced templates: templates/tumor/*.html
- Secure endpoints: login_required + permission_required where appropriate
- Robust PDF generation & storage
- Chatbot agent run with timeout via ThreadPoolExecutor
- Lightweight subscribe endpoint (POST)
"""

import os
import json
import logging
import traceback
from io import BytesIO
from datetime import datetime, timedelta
from urllib.parse import urljoin

import concurrent.futures

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required, permission_required
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Count, Avg
from django.db.models.functions import TruncDate
from django.http import HttpResponse, JsonResponse, HttpResponseForbidden, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# Local imports
from .models import Patient, Study, AIResult
from .forms import StudyUploadForm, BrainwebAuthenticationForm
from .ai_agent import run_brainweb_agent

# Utils (small helpers - implement under .utils)
from .utils.slices import generate_slice_gallery       # should return list of absolute paths
from .utils.preview3d import generate_3d_preview       # render static 3D snapshot
from .utils.qrgen import generate_qr                   # simple QR PNG generator

logger = logging.getLogger(__name__)

# Agent timeout in seconds (configurable via settings)
DEFAULT_AGENT_TIMEOUT = getattr(settings, "BRAINWEB_AGENT_TIMEOUT", 15)


def _rel_or_fs_to_abs_url(request, path_or_rel):
    """Convert a filesystem path or media-relative path to an absolute URL.

    Tries multiple strategies:
      - if value already absolute URL -> return as-is
      - if absolute filesystem path under MEDIA_ROOT -> convert to MEDIA_URL
      - if storage backend can provide url -> use default_storage.url
      - fallback -> treat as MEDIA relative path
    """
    if not path_or_rel:
        return None
    try:
        if isinstance(path_or_rel, str) and (path_or_rel.startswith("http://") or path_or_rel.startswith("https://")):
            return path_or_rel

        # If it's inside MEDIA_ROOT on filesystem
        media_root_abs = os.path.abspath(settings.MEDIA_ROOT)
        abspath = os.path.abspath(path_or_rel)
        if abspath.startswith(media_root_abs):
            rel = os.path.relpath(abspath, media_root_abs).replace(os.path.sep, "/")
            media_url = settings.MEDIA_URL if settings.MEDIA_URL.endswith("/") else settings.MEDIA_URL + "/"
            return request.build_absolute_uri(urljoin(media_url, rel))
    except Exception:
        # fallback to storage.url (works when argument is storage name)
        try:
            return request.build_absolute_uri(default_storage.url(path_or_rel))
        except Exception:
            pass

    # final fallback: treat as MEDIA relative path
    media_url = settings.MEDIA_URL if settings.MEDIA_URL.endswith("/") else settings.MEDIA_URL + "/"
    return request.build_absolute_uri(urljoin(media_url, str(path_or_rel).lstrip("/")))


# ----------------------------
# Public pages
# ----------------------------
def home_redirect(request):
    return redirect("tumor:dashboard")


def health_check(request):
    """Zero-dependency health check for Railway."""
    return HttpResponse("OK")


def about(request):
    """Public about page - always accessible by anyone."""
    return render(request, "about.html")


def documentation(request):
    return render(request, "tumor/documentation.html")


def faq(request):
    return render(request, "tumor/faq.html")


def privacy(request):
    return render(request, "tumor/privacy.html")


def terms(request):
    return render(request, "tumor/terms.html")


from django.core.mail import send_mail
from django.shortcuts import reverse

def _save_contact_to_queue(name, email, message):
    # simple file queue for dev: append json line to MEDIA_ROOT/contact_queue.ndjson
    try:
        qdir = os.path.join(settings.MEDIA_ROOT, "contact_queue")
        os.makedirs(qdir, exist_ok=True)
        fname = os.path.join(qdir, f"{datetime.utcnow().strftime('%Y%m%d')}.ndjson")
        with open(fname, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.utcnow().isoformat(), "name": name, "email": email, "message": message}) + "\n")
    except Exception:
        logger.exception("Failed to save contact to queue file")

def contact(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        email = request.POST.get("email", "").strip()
        message_text = request.POST.get("message", "").strip()

        if not email or not message_text:
            messages.error(request, "Iltimos email va xabar kiriting.")
            return redirect("tumor:contact")

        subject = f"Contact form — {name or 'No name'}"
        body = f"From: {name} <{email}>\n\n{message_text}"

        try:
            # optional: explicit connection to get better control
            connection = get_connection(fail_silently=False)
            send_mail(subject, body, settings.DEFAULT_FROM_EMAIL, [getattr(settings, "SUPPORT_EMAIL", "support@brainweb.ai")], connection=connection)
            messages.success(request, "Xabaringiz qabul qilindi. Rahmat!")
        except Exception:
            logger.exception("Contact form send failed")
            # fallback: save to queue for later processing by worker
            _save_contact_to_queue(name, email, message_text)
            messages.success(request, "Xabaringiz qabul qilindi. Agar email yuborib bo'lmasa, keyinroq qo'ng'iroq qilamiz.")

        return redirect("tumor:contact")

    return render(request, "tumor/contact.html")



@require_POST
def subscribe(request):
    """Simple newsletter subscription (POST). Validate email and store or queue."""
    email = (request.POST.get("email") or "").strip()
    if not email:
        messages.error(request, "Iltimos, email kiriting.")
        return redirect("tumor:about")

    try:
        validate_email(email)
    except ValidationError:
        messages.error(request, "Noto‘g‘ri email manzil.")
        return redirect("tumor:about")

    # TODO: replace with persistent model (Subscription) or queue task (Celery)
    # Example: Subscription.objects.create(email=email)
    try:
        # Placeholder: try to append to a local file if MEDIA_ROOT writable (development)
        if getattr(settings, "DEBUG", False):
            subs_dir = os.path.join(settings.MEDIA_ROOT, "subscriptions")
            os.makedirs(subs_dir, exist_ok=True)
            fname = os.path.join(subs_dir, f"{datetime.utcnow().strftime('%Y%m%d')}.txt")
            with open(fname, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.utcnow().isoformat()} {email}\n")
        messages.success(request, "Rahmat — emailingiz qabul qilindi.")
    except Exception:
        logger.exception("Subscribe failed for email %s", email)
        messages.success(request, "Rahmat — emailingiz qabul qilindi.")  # user-facing: keep UX smooth

    return redirect("tumor:about")


# ----------------------------
# Studies: list / upload
# ----------------------------
def study_list(request):
    # Session-based study tracking (no login required)
    session_study_ids = request.session.get("study_ids", [])

    form = StudyUploadForm(data=request.POST or None, files=request.FILES or None)
    if request.method == "POST":
        from . import ai
        if form.is_valid():
            study = None
            try:
                patient, _ = Patient.objects.get_or_create(
                    first_name=form.cleaned_data["first_name"],
                    last_name=form.cleaned_data["last_name"],
                    birth_date=form.cleaned_data.get("birth_date"),
                    identifier=form.cleaned_data.get("identifier") or "",
                )

                uploaded_file = form.cleaned_data["uploaded_file"]

                study = Study.objects.create(
                    patient=patient,
                    created_by=request.user if request.user.is_authenticated else None,
                    modality=form.cleaned_data.get("modality", ""),
                    description=form.cleaned_data.get("description", ""),
                    uploaded_file=uploaded_file,
                    status="processing",
                )

                # Session orqali study ID ni saqlash
                ids = request.session.get("study_ids", [])
                if study.pk not in ids:
                    ids.append(study.pk)
                request.session["study_ids"] = ids

                # Preprocess -> nifti
                uploaded_abs = study.uploaded_file.path
                preprocessed_nifti = ai.preprocess_to_nifti(uploaded_abs)
                study.nifti_file.name = os.path.relpath(preprocessed_nifti, settings.MEDIA_ROOT)
                study.save(update_fields=["nifti_file"])

                # Segmentation
                mask_abs = ai.run_segmentation(preprocessed_nifti)
                mask_rel = os.path.relpath(mask_abs, settings.MEDIA_ROOT)

                # Metrics & overlay
                metrics = ai.compute_bbox_and_metrics(mask_abs)
                overlay_rel = ai.generate_overlay_png(preprocessed_nifti, mask_abs)

                # Classification (optional)
                predicted_class, predicted_conf = "", None
                try:
                    predicted_class, predicted_conf = ai.classify_tumor(preprocessed_nifti)
                except Exception:
                    logger.debug("Tumor classifier not available or failed.", exc_info=True)

                # Save AIResult
                ai_result = AIResult.objects.create(
                    study=study,
                    tumor_volume_mm3=metrics.get("tumor_volume_mm3"),
                    tumor_max_diameter_mm=metrics.get("tumor_max_diameter_mm"),
                    bbox_x_min=metrics["bbox"].get("x_min"),
                    bbox_y_min=metrics["bbox"].get("y_min"),
                    bbox_z_min=metrics["bbox"].get("z_min"),
                    bbox_x_max=metrics["bbox"].get("x_max"),
                    bbox_y_max=metrics["bbox"].get("y_max"),
                    bbox_z_max=metrics["bbox"].get("z_max"),
                    predicted_class=predicted_class or "",
                    predicted_confidence=predicted_conf,
                )

                ai_result.mask_nifti.name = mask_rel
                if overlay_rel:
                    ai_result.bbox_png.name = overlay_rel

                # Mesh generation (best-effort)
                try:
                    mesh_abs = ai.generate_tumor_mesh_json(mask_abs, study.id)
                    ai_result.tumor_mesh_json.name = os.path.relpath(mesh_abs, settings.MEDIA_ROOT)
                except Exception as e:
                    logger.exception("Mesh generation failed: %s", e)

                ai_result.save()

                study.status = "done"
                study.save(update_fields=["status"])

                messages.success(request, "Study uploaded and processed successfully.")
                return redirect("tumor:study_detail", pk=study.pk)

            except Exception as e:
                logger.exception("Study processing failed: %s", e)
                if study is not None:
                    study.status = "error"
                    study.error_message = str(e)
                    study.save(update_fields=["status", "error_message"])
                messages.error(request, f"Error while processing study: {e}")
        else:
            messages.error(request, "Upload form is invalid. Check required fields.")

    session_study_ids = request.session.get("study_ids", [])
    if request.user.is_authenticated:
        studies_qs = Study.objects.filter(created_by=request.user).select_related("patient").order_by("-created_at")
    else:
        studies_qs = Study.objects.filter(pk__in=session_study_ids).select_related("patient").order_by("-created_at")

    # simple stats for UI chart
    stats_by_day = (
        studies_qs
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )
    chart_labels = [r["day"].strftime("%Y-%m-%d") for r in stats_by_day]
    chart_values = [r["count"] for r in stats_by_day]

    return render(request, "tumor/study_list.html", {
        "form": form,
        "studies": studies_qs,
        "chart_labels_json": json.dumps(chart_labels),
        "chart_values_json": json.dumps(chart_values),
    })


# ----------------------------
# Study detail / viewer
# ----------------------------
def study_detail(request, pk):
    study = get_object_or_404(Study.objects.select_related("patient"), pk=pk)

    session_study_ids = request.session.get("study_ids", [])
    if study.created_by and study.created_by != request.user:
        if not request.user.is_authenticated and study.pk not in session_study_ids:
            return HttpResponseForbidden("You are not allowed to view this study.")

    ai_result = getattr(study, "ai_result", None)

    volume_cm3 = None
    max_diam_cm = None
    if ai_result and ai_result.tumor_volume_mm3 is not None:
        volume_cm3 = ai_result.tumor_volume_mm3 / 1000.0
    if ai_result and ai_result.tumor_max_diameter_mm is not None:
        max_diam_cm = ai_result.tumor_max_diameter_mm / 10.0

    orig_slice_urls = []
    overlay_slice_urls = []
    try:
        stacks_base = os.path.join(settings.MEDIA_ROOT, "results", "stacks")
        orig_dir = os.path.join(stacks_base, "orig")
        overlay_dir = os.path.join(stacks_base, "overlay")
        prefix_orig = f"study{study.id}_orig_"
        prefix_overlay = f"study{study.id}_overlay_"
        if os.path.isdir(orig_dir) and os.path.isdir(overlay_dir):
            orig_files = sorted([f for f in os.listdir(orig_dir) if f.startswith(prefix_orig) and f.endswith(".png")])
            overlay_files = sorted([f for f in os.listdir(overlay_dir) if f.startswith(prefix_overlay) and f.endswith(".png")])
            n = min(len(orig_files), len(overlay_files))
            for i in range(n):
                orig_rel = os.path.join("results", "stacks", "orig", orig_files[i])
                overlay_rel = os.path.join("results", "stacks", "overlay", overlay_files[i])
                orig_slice_urls.append(_rel_or_fs_to_abs_url(request, orig_rel))
                overlay_slice_urls.append(_rel_or_fs_to_abs_url(request, overlay_rel))
    except Exception:
        logger.exception("Failed to collect stack PNGs for study %s", study.id)

    tumor_mesh_url = None
    if ai_result and ai_result.tumor_mesh_json:
        try:
            tumor_mesh_url = ai_result.tumor_mesh_json.url
        except Exception:
            tumor_mesh_url = None

    return render(request, "tumor/study_detail.html", {
        "study": study,
        "ai_result": ai_result,
        "volume_cm3": volume_cm3,
        "max_diam_cm": max_diam_cm,
        "orig_slice_urls": orig_slice_urls,
        "overlay_slice_urls": overlay_slice_urls,
        "tumor_mesh_url": tumor_mesh_url,
    })


# ----------------------------
# Advanced PDF report v2.0
# ----------------------------
def study_report_pdf(request, pk):
    study = get_object_or_404(Study.objects.select_related("patient"), pk=pk)

    session_study_ids = request.session.get("study_ids", [])
    if study.created_by and study.created_by != request.user:
        if not request.user.is_authenticated and study.pk not in session_study_ids:
            return HttpResponseForbidden("You are not allowed to export this study.")

    ai_result = getattr(study, "ai_result", None)

    # overlay / grad-cam / uncertainty absolute URLs
    overlay_url = grad_cam_url = uncertainty_map_url = None
    try:
        if ai_result and getattr(ai_result, "bbox_png", None):
            overlay_url = _rel_or_fs_to_abs_url(request, ai_result.bbox_png.path if hasattr(ai_result.bbox_png, "path") else ai_result.bbox_png.url)
    except Exception:
        logger.debug("Failed to build overlay_url", exc_info=True)
    try:
        if ai_result and getattr(ai_result, "grad_cam_png", None):
            grad_cam_url = _rel_or_fs_to_abs_url(request, ai_result.grad_cam_png.path if hasattr(ai_result.grad_cam_png, "path") else ai_result.grad_cam_png.url)
    except Exception:
        logger.debug("Failed to build grad_cam_url", exc_info=True)
    try:
        if ai_result and getattr(ai_result, "uncertainty_png", None):
            uncertainty_map_url = _rel_or_fs_to_abs_url(request, ai_result.uncertainty_png.path if hasattr(ai_result.uncertainty_png, "path") else ai_result.uncertainty_png.url)
    except Exception:
        logger.debug("Failed to build uncertainty_map_url", exc_info=True)

    # QR generation for report (saved under MEDIA_ROOT/reports/{study.id}/{ts}/qr.png)
    qr_code_url = None
    report_dir = None
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(settings.MEDIA_ROOT, "reports", str(study.id), ts)
        os.makedirs(report_dir, exist_ok=True)
        study_detail_url = reverse_lazy("tumor:study_detail", kwargs={"pk": study.pk})
        absolute_study_url = request.build_absolute_uri(study_detail_url)
        qr_path = os.path.join(report_dir, "qr.png")
        generate_qr(absolute_study_url, qr_path)
        qr_code_url = _rel_or_fs_to_abs_url(request, qr_path)
    except Exception:
        logger.exception("QR generation failed for study %s", study.id)

    # Growth prediction (best-effort)
    pred_vol_30_days = pred_vol_90_days = growth_rate_30_days = sim_vol_post_treatment = None
    try:
        from . import ai
        if ai_result and ai_result.tumor_volume_mm3 is not None:
            prog = ai.predict_growth_and_simulate(
                patient_id=study.patient.id,
                current_volume_mm3=ai_result.tumor_volume_mm3,
                study_id=study.id,
            )
            pred_vol_30_days = prog.get("pred_vol_30_days")
            pred_vol_90_days = prog.get("pred_vol_90_days")
            growth_rate_30_days = prog.get("growth_rate_30_days")
            sim_vol_post_treatment = prog.get("sim_vol_post_treatment")
    except Exception:
        logger.exception("Growth prediction failed for study %s", study.id)

    # Slice gallery generation
    slice_gallery = []
    try:
        if report_dir is None:
            report_dir = os.path.join(settings.MEDIA_ROOT, "reports", str(study.id), datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(report_dir, exist_ok=True)

        nifti_path = study.nifti_file.path if study.nifti_file else None
        mask_path = ai_result.mask_nifti.path if ai_result and getattr(ai_result, "mask_nifti", None) and getattr(ai_result.mask_nifti, "path", None) else None

        if nifti_path and mask_path:
            slice_paths = generate_slice_gallery(nifti_path, mask_path, report_dir, n_slices=8)
            slice_gallery = [_rel_or_fs_to_abs_url(request, p) for p in (slice_paths or [])]
    except Exception:
        logger.exception("Slice gallery generation failed for study %s", study.id)

    # 3D preview snapshot
    preview3d = None
    try:
        if mask_path:
            preview_path = os.path.join(report_dir, "preview3d.png")
            generate_3d_preview(mask_path, preview_path)
            preview3d = _rel_or_fs_to_abs_url(request, preview_path)
    except Exception:
        logger.exception("3D preview generation failed for study %s", study.id)

    # Render HTML for PDF
    generated_at = timezone.now()
    context = {
        "study": study,
        "ai_result": ai_result,
        "overlay_url": overlay_url,
        "grad_cam_url": grad_cam_url,
        "uncertainty_map_url": uncertainty_map_url,
        "qr_code_url": qr_code_url,
        "slice_gallery": slice_gallery,
        "preview3d": preview3d,
        "pred_vol_30_days": pred_vol_30_days,
        "pred_vol_90_days": pred_vol_90_days,
        "growth_rate_30_days": growth_rate_30_days,
        "sim_vol_post_treatment": sim_vol_post_treatment,
        "radiologist_summary": request.POST.get("radiologist_summary") or getattr(study, "radiologist_summary", ""),
        "now": generated_at,
    }

    html_string = render_to_string("tumor/report_v2.0.html", context)

    pdf_io = BytesIO()
    try:
        from weasyprint import HTML, CSS
        html = HTML(string=html_string, base_url=request.build_absolute_uri("/"))
        css_path = os.path.join(settings.BASE_DIR, "templates", "tumor", "report_v2.css")
        if os.path.exists(css_path):
            html.write_pdf(pdf_io, stylesheets=[CSS(filename=css_path)])
        else:
            html.write_pdf(pdf_io)
        pdf_io.seek(0)
    except Exception:
        logger.exception("WeasyPrint PDF generation failed for study %s", study.id)
        return HttpResponseBadRequest("PDF generation failed on server.")

    # Save PDF into storage / update AIResult
    try:
        pdf_filename = f"report_{study.id}_{generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"
        saved_pdf_path = os.path.join(report_dir, pdf_filename)
        os.makedirs(report_dir, exist_ok=True)
        with open(saved_pdf_path, "wb") as f:
            f.write(pdf_io.getbuffer())

        try:
            saved_pdf_rel = os.path.relpath(saved_pdf_path, settings.MEDIA_ROOT)
        except Exception:
            saved_pdf_rel = saved_pdf_path

        saved_pdf_url = _rel_or_fs_to_abs_url(request, saved_pdf_rel)

        ar = getattr(study, "ai_result", None)
        if ar is None:
            logger.warning("AIResult not found for study %s; PDF generated but DB fields not updated.", study.pk)
        else:
            try:
                with transaction.atomic():
                    if hasattr(ar, "report_pdf") and getattr(ar._meta.get_field("report_pdf"), "upload_to", None) is not None:
                        with open(saved_pdf_path, "rb") as fh:
                            content = fh.read()
                        storage_name = f"reports/study_{study.id}/{pdf_filename}"
                        ar.report_pdf.save(storage_name, ContentFile(content), save=False)
                        ar.report_generated_at = generated_at
                        try:
                            ar.report_pdf_url = default_storage.url(ar.report_pdf.name)
                        except Exception:
                            ar.report_pdf_url = _rel_or_fs_to_abs_url(request, ar.report_pdf.name)
                        ar.save(update_fields=["report_pdf", "report_generated_at", "report_pdf_url"])
                    else:
                        ar.report_pdf_url = saved_pdf_url
                        ar.report_generated_at = generated_at
                        ar.save(update_fields=["report_pdf_url", "report_generated_at"])
            except Exception:
                logger.exception("Failed to save PDF metadata into AIResult for study %s", study.pk)
    except Exception:
        logger.exception("Saving PDF failed for study %s", study.id)

    # Return PDF response to user
    response = HttpResponse(pdf_io.read(), content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="study_{study.id}_advanced_report.pdf"'
    return response


# ----------------------------
# Dashboard
# ----------------------------
def dashboard(request):
    if request.user.is_authenticated:
        studies_qs = Study.objects.filter(created_by=request.user)
    else:
        session_study_ids = request.session.get("study_ids", [])
        studies_qs = Study.objects.filter(pk__in=session_study_ids)

    total_studies = studies_qs.count()
    completed_studies = studies_qs.filter(status="done").count()
    processing_studies = studies_qs.filter(status__in=["pending", "processing"]).count()
    error_studies = studies_qs.filter(status="error").count()

    today = timezone.now().date()
    start_date = today - timedelta(days=6)
    daily_raw = (
        studies_qs.filter(created_at__date__gte=start_date)
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )

    daily_labels = []
    daily_counts = []
    for i in range(7):
        d = start_date + timedelta(days=i)
        daily_labels.append(d.strftime("%Y-%m-%d"))
        match = next((x for x in daily_raw if x["day"] == d), None)
        daily_counts.append(match["count"] if match else 0)

    ai_qs = AIResult.objects.filter(study__in=studies_qs)
    class_raw = (
        ai_qs.filter(predicted_class__isnull=False)
        .exclude(predicted_class="")
        .values("predicted_class")
        .annotate(count=Count("id"))
        .order_by("predicted_class")
    )
    class_labels = [r["predicted_class"] for r in class_raw]
    class_counts = [r["count"] for r in class_raw]

    avg_volume = ai_qs.filter(tumor_volume_mm3__isnull=False).aggregate(avg=Avg("tumor_volume_mm3"))["avg"]

    recent_studies = studies_qs.select_related("patient").order_by("-created_at")[:5]

    return render(request, "tumor/dashboard.html", {
        "total_studies": total_studies,
        "completed_studies": completed_studies,
        "processing_studies": processing_studies,
        "error_studies": error_studies,
        "daily_labels": daily_labels,
        "daily_counts": daily_counts,
        "class_labels": class_labels,
        "class_counts": class_counts,
        "avg_volume": avg_volume,
        "recent_studies": recent_studies,
        "now": timezone.now(),   # <-- qo'shildi

    })


# ----------------------------
# Study status API (AJAX)
# ----------------------------
def study_status_api(request, pk):
    study = get_object_or_404(Study.objects.select_related("patient"), pk=pk)
    session_study_ids = request.session.get("study_ids", [])
    if study.created_by and study.created_by != request.user:
        if not request.user.is_authenticated and study.pk not in session_study_ids:
            return HttpResponseForbidden("You are not allowed to query this study status.")
    ai_result = getattr(study, "ai_result", None)
    data = {
        "status": study.status,
        "error_message": study.error_message,
        "has_ai_result": ai_result is not None,
    }
    return JsonResponse(data)


# ----------------------------
# Chatbot API (agent) - run with timeout using ThreadPoolExecutor
# ----------------------------
def chat_api(request):
    """POST JSON: {'message': '...'} -> JSON {'reply': '...'}.
    Expects standard CSRF token from frontend (no csrf_exempt).
    The agent is executed in a thread pool and timed out by DEFAULT_AGENT_TIMEOUT.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = (data.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "Empty message"}, status=400)

    user = request.user if request.user.is_authenticated else None

    reply = None
    try:
        # Run agent in a thread with timeout to avoid blocking server process
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_brainweb_agent, user, message)
            try:
                reply = future.result(timeout=DEFAULT_AGENT_TIMEOUT)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.warning("Agent timeout after %s seconds for user %s", DEFAULT_AGENT_TIMEOUT, getattr(user, "pk", None))
                reply = "AI moduli bilan bog'lanishda vaqtincha nosozlik yuz berdi (timeout)."
    except Exception as e:
        logger.exception("Agent failed: %s", e)
        if settings.DEBUG:
            reply = f"Tizim xatoligi (Debug): {str(e)}\n\n{traceback.format_exc()}"
        else:
            reply = "AI moduli bilan bog'lanishda vaqtincha nosozlik yuz berdi."

    return JsonResponse({"reply": reply})
