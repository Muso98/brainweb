# tumor/models.py
from django.db import models
from django.conf import settings


class Patient(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    identifier = models.CharField(
        max_length=100,
        blank=True,
        help_text="Clinic / Hospital ID (optional)",
    )

    def __str__(self):
        full_name = f"{self.first_name} {self.last_name}".strip()
        if self.identifier:
            return f"{full_name} ({self.identifier})"
        return full_name


class Study(models.Model):
    MODALITY_CHOICES = [
        ("MRI", "MRI"),
        ("CT", "CT"),
    ]

    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="studies",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="studies",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    modality = models.CharField(max_length=10, choices=MODALITY_CHOICES)
    description = models.CharField(max_length=255, blank=True)

    # User uploaded original file (nii / nii.gz / zip)
    uploaded_file = models.FileField(upload_to="uploads/")

    # Preprocessed NIfTI (model input)
    nifti_file = models.FileField(
        upload_to="nifti/",
        null=True,
        blank=True,
    )

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("done", "Done"),
        ("error", "Error"),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending",
    )
    error_message = models.TextField(blank=True)

    def __str__(self):
        return f"Study #{self.id} - {self.modality} - {self.patient}"

    @property
    def has_ai_result(self):
        return hasattr(self, "ai_result")

    class Meta:
        permissions = [
            ("upload_study", "Can upload MRI/CT study"),
            ("review_study", "Can medically review AI results")
        ]


class AIResult(models.Model):
    study = models.OneToOneField(
        Study,
        on_delete=models.CASCADE,
        related_name="ai_result",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    # 3D mask (nii.gz) - nullable when using Groq AI (no nnUNet)
    mask_nifti = models.FileField(upload_to="results/masks/", null=True, blank=True)

    # BBOX overlay image (png)
    bbox_png = models.ImageField(
        upload_to="results/overlays/",
        null=True,
        blank=True,
    )

    # Additional PNG slices (optional)
    preview_png = models.ImageField(
        upload_to="results/previews/",
        null=True,
        blank=True,
    )

    # Analytic numbers
    tumor_volume_mm3 = models.FloatField(null=True, blank=True)
    tumor_max_diameter_mm = models.FloatField(null=True, blank=True)

    # bounding-box corners (voxel indices)
    bbox_x_min = models.IntegerField(null=True, blank=True)
    bbox_y_min = models.IntegerField(null=True, blank=True)
    bbox_z_min = models.IntegerField(null=True, blank=True)
    bbox_x_max = models.IntegerField(null=True, blank=True)
    bbox_y_max = models.IntegerField(null=True, blank=True)
    bbox_z_max = models.IntegerField(null=True, blank=True)

    # bounding box diagonal (mm) — new: useful as technical metric
    bbox_diag_mm = models.FloatField(null=True, blank=True)

    # Tumor type prediction (if classifier present)
    predicted_class = models.CharField(max_length=100, blank=True)
    predicted_confidence = models.FloatField(null=True, blank=True)

    # New: structured volumes by groups (core/whole/enhancing) — JSONField
    # Django >= 3.1: use models.JSONField (DB-agnostic). If older, use
    # django.contrib.postgres.fields.JSONField for Postgres only.
    volumes_by_group = models.JSONField(null=True, blank=True)

    # Mesh and report fields
    tumor_mesh_json = models.FileField(
        upload_to="results/meshes/",
        blank=True,
        null=True,
    )

    report_pdf_url = models.CharField(max_length=1024, null=True, blank=True)
    report_generated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"AIResult for Study #{self.study_id}"

    @property
    def tumor_volume_cm3(self):
        if self.tumor_volume_mm3 is None:
            return None
        return self.tumor_volume_mm3 / 1000.0
