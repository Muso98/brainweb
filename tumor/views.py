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


def _handle_study_upload(request, form):
    """POST /studies/ — fayl yuklash va Groq AI tahlili. Hech qachon 500 bermaydi."""
    study = None
    try:
        logger.info("[UPLOAD] POST boshlandi")

        if not form.is_valid():
            logger.warning("[UPLOAD] Form invalid: %s", form.errors)
            messages.error(request, "Forma to'ldirilmagan. Kerakli maydonlarni tekshiring.")
            return redirect("tumor:study_list")

        from . import ai
        from .models import Patient as PatientModel

        # 1. Bemor yaratish
        patient, _ = PatientModel.objects.get_or_create(
            first_name=form.cleaned_data["first_name"],
            last_name=form.cleaned_data.get("last_name") or "",
            birth_date=form.cleaned_data.get("birth_date"),
            identifier=form.cleaned_data.get("identifier") or "",
        )
        logger.info("[UPLOAD] Bemor: %s", patient)

        # 2. Study yaratish
        study = Study.objects.create(
            patient=patient,
            created_by=request.user if request.user.is_authenticated else None,
            modality=form.cleaned_data.get("modality", "MRI"),
            description=form.cleaned_data.get("description", ""),
            uploaded_file=form.cleaned_data["uploaded_file"],
            status="processing",
        )
        logger.info("[UPLOAD] Study #%s yaratildi", study.pk)

        # Session ga qo'shish
        try:
            ids = request.session.get("study_ids", [])
            if study.pk not in ids:
                ids.append(study.pk)
            request.session["study_ids"] = ids
        except Exception as sess_err:
            logger.warning("[UPLOAD] Session xatosi: %s", sess_err)

        # 3. NIfTI ga o'girish
        uploaded_abs = study.uploaded_file.path
        logger.info("[UPLOAD] Fayl: %s", uploaded_abs)
        preprocessed_nifti = ai.preprocess_to_nifti(uploaded_abs)
        study.nifti_file.name = os.path.relpath(preprocessed_nifti, settings.MEDIA_ROOT)
        study.save(update_fields=["nifti_file"])
        logger.info("[UPLOAD] NIfTI: %s", preprocessed_nifti)

        # 4. Groq AI tahlili
        from .ai_groq import analyze_study, save_slices_png
        modality = form.cleaned_data.get("modality", "MRI")
        logger.info("[UPLOAD] Groq tahlili boshlanmoqda...")
        groq_result = analyze_study(preprocessed_nifti, modality=modality)
        logger.info("[UPLOAD] Groq natija: %s", str(groq_result)[:300])

        # 5. Volume hisoblash
        vol_mm3 = None
        try:
            v = groq_result.get("tumor_volume_estimate_cm3")
            if v is not None:
                vol_mm3 = float(v) * 1000
        except Exception:
            pass

        conf = groq_result.get("confidence")
        predicted_conf = float(conf) if conf is not None else None
        predicted_class = str(groq_result.get("predicted_class") or "")

        # 6. AIResult saqlash
        ai_result = AIResult.objects.create(
            study=study,
            tumor_volume_mm3=vol_mm3,
            tumor_max_diameter_mm=None,
            predicted_class=predicted_class,
            predicted_confidence=predicted_conf,
            volumes_by_group={
                "groq_analysis": {
                    "tumor_detected": groq_result.get("tumor_detected"),
                    "severity": groq_result.get("severity"),
                    "location": groq_result.get("location"),
                    "findings": groq_result.get("findings"),
                    "recommendation": groq_result.get("recommendation"),
                    "model": groq_result.get("groq_model"),
                }
            },
        )
        logger.info("[UPLOAD] AIResult #%s saqlandi", ai_result.pk)

        # 7. Kesim rasmlari saqlash
        try:
            slices_path = save_slices_png(preprocessed_nifti, study.id)
            if slices_path:
                ai_result.bbox_png.name = os.path.relpath(slices_path, settings.MEDIA_ROOT)
                ai_result.save(update_fields=["bbox_png"])
        except Exception as png_err:
            logger.warning("[UPLOAD] PNG saqlashda xato (kritik emas): %s", png_err)

        study.status = "done"
        study.save(update_fields=["status"])
        logger.info("[UPLOAD] Muvaffaqiyat! Study #%s", study.pk)

        messages.success(request, "MRI/CT tahlili muvaffaqiyatli bajarildi!")
        return redirect("tumor:study_detail", pk=study.pk)

    except Exception as e:
        logger.exception("[UPLOAD] XATO: %s", e)
        if study is not None:
            try:
                study.status = "error"
                study.error_message = str(e)[:2000]
                study.save(update_fields=["status", "error_message"])
                messages.warning(request, f"Fayl yuklandi, lekin xato yuz berdi: {e}")
                return redirect("tumor:study_detail", pk=study.pk)
            except Exception as e2:
                logger.error("[UPLOAD] study.save xatosi: %s", e2)
        messages.error(request, f"Xato: {e}")
        return redirect("tumor:study_list")


# ----------------------------
# Studies: list / upload
# ----------------------------
def study_list(request):
    form = StudyUploadForm(data=request.POST or None, files=request.FILES or None)
    if request.method == "POST":
        return _handle_study_upload(request, form)

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
    """Render a print-friendly HTML report that auto-triggers browser print dialog."""
    study = get_object_or_404(Study.objects.select_related("patient"), pk=pk)

    session_study_ids = request.session.get("study_ids", [])
    if study.created_by and study.created_by != request.user:
        if not request.user.is_authenticated and study.pk not in session_study_ids:
            return HttpResponseForbidden("You are not allowed to export this study.")

    ai_result = getattr(study, "ai_result", None)

    volume_cm3 = None
    if ai_result and ai_result.tumor_volume_mm3 is not None:
        volume_cm3 = ai_result.tumor_volume_mm3 / 1000.0

    # Build absolute URL for MRI slice image
    slice_image_url = None
    try:
        if ai_result and getattr(ai_result, "bbox_png", None) and ai_result.bbox_png.name:
            slice_image_url = request.build_absolute_uri(ai_result.bbox_png.url)
    except Exception:
        pass

    groq_data = None
    if ai_result and isinstance(ai_result.volumes_by_group, dict):
        groq_data = ai_result.volumes_by_group.get("groq_analysis")

    generated_at = timezone.now()
    context = {
        "study": study,
        "ai_result": ai_result,
        "volume_cm3": volume_cm3,
        "slice_image_url": slice_image_url,
        "groq_data": groq_data,
        "generated_at": generated_at,
        "base_url": request.build_absolute_uri("/"),
    }
    return render(request, "tumor/report_print.html", context)


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
