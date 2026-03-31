# tumor/admin.py
from django.contrib import admin
from .models import Patient, Study, AIResult


# =========================================
# Patient admin
# =========================================

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("first_name", "last_name", "identifier", "birth_date")
    search_fields = ("first_name", "last_name", "identifier")
    list_filter = ("birth_date",)
    ordering = ("first_name", "last_name")

    fieldsets = (
        (None, {
            "fields": ("first_name", "last_name")
        }),
        ("Additional info", {
            "fields": ("birth_date", "identifier"),
            "classes": ("collapse",),
        }),
    )


# =========================================
# AIResult inline – Study ichida ko‘rsatamiz
# =========================================

class AIResultInline(admin.StackedInline):
    model = AIResult
    can_delete = False
    extra = 0

    readonly_fields = (
        "created_at",
        "mask_nifti",
        "bbox_png",
        "preview_png",
        "tumor_volume_mm3",
        "tumor_max_diameter_mm",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_z_min",
        "bbox_x_max",
        "bbox_y_max",
        "bbox_z_max",
        "predicted_class",
        "predicted_confidence",
        "tumor_volume_cm3",
    )

    fieldsets = (
        ("Files", {
            "fields": ("mask_nifti", "bbox_png", "preview_png"),
        }),
        ("Tumor metrics", {
            "fields": (
                "tumor_volume_mm3",
                "tumor_volume_cm3",
                "tumor_max_diameter_mm",
            ),
        }),
        ("Bounding box", {
            "fields": (
                ("bbox_x_min", "bbox_x_max"),
                ("bbox_y_min", "bbox_y_max"),
                ("bbox_z_min", "bbox_z_max"),
            ),
            "classes": ("collapse",),
        }),
        ("Classification", {
            "fields": ("predicted_class", "predicted_confidence"),
        }),
        ("Meta", {
            "fields": ("created_at",),
        }),
    )

    def has_add_permission(self, request, obj=None):
        # AIResult faqat pipeline orqali yaratiladi, admin qo‘shmasin
        return False


# =========================================
# Study admin
# =========================================

@admin.register(Study)
class StudyAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "patient",
        "modality",
        "status",
        "created_at",
        "created_by",
    )
    list_filter = ("modality", "status", "created_at")
    search_fields = (
        "patient__first_name",
        "patient__last_name",
        "patient__identifier",
        "description",
    )
    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    readonly_fields = (
        "created_at",
        "created_by",
        "nifti_file",
        "error_message",
    )

    inlines = [AIResultInline]

    fieldsets = (
        ("Patient", {
            "fields": ("patient",),
        }),
        ("Study info", {
            "fields": (
                "modality",
                "description",
            ),
        }),
        ("Files", {
            "fields": (
                "uploaded_file",
                "nifti_file",
            ),
        }),
        ("Status", {
            "fields": (
                "status",
                "error_message",
            ),
        }),
        ("Meta", {
            "fields": ("created_at", "created_by"),
        }),
    )

    def save_model(self, request, obj, form, change):
        # Yangi Study yaratilsa, created_by ni avtomatik to‘ldiramiz
        if not change and not obj.created_by:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


# =========================================
# AIResult admin – alohida ham ko‘rish mumkin, lekin faqat readonly
# =========================================

@admin.register(AIResult)
class AIResultAdmin(admin.ModelAdmin):
    list_display = ("study", "created_at", "predicted_class", "predicted_confidence")
    list_filter = ("created_at", "predicted_class")
    search_fields = (
        "study__patient__first_name",
        "study__patient__last_name",
        "study__patient__identifier",
        "predicted_class",
    )
    readonly_fields = (
        "study",
        "created_at",
        "mask_nifti",
        "bbox_png",
        "preview_png",
        "tumor_volume_mm3",
        "tumor_max_diameter_mm",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_z_min",
        "bbox_x_max",
        "bbox_y_max",
        "bbox_z_max",
        "predicted_class",
        "predicted_confidence",
        "tumor_volume_cm3",
    )

    fieldsets = (
        (None, {
            "fields": ("study", "created_at"),
        }),
        ("Files", {
            "fields": ("mask_nifti", "bbox_png", "preview_png"),
        }),
        ("Tumor metrics", {
            "fields": ("tumor_volume_mm3", "tumor_volume_cm3", "tumor_max_diameter_mm"),
        }),
        ("Bounding box", {
            "fields": (
                ("bbox_x_min", "bbox_x_max"),
                ("bbox_y_min", "bbox_y_max"),
                ("bbox_z_min", "bbox_z_max"),
            ),
            "classes": ("collapse",),
        }),
        ("Classification", {
            "fields": ("predicted_class", "predicted_confidence"),
        }),
    )

    def has_add_permission(self, request):
        # Yangi AIResult admin orqali yaratilmasin
        return False

    def has_change_permission(self, request, obj=None):
        # Faqat ko‘rish, o‘zgartirish mumkin emas
        if request.user.is_superuser:
            return True  # agar hohlasangiz superuserga ruxsat qoldirishingiz mumkin
        return False
