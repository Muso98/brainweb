# tumor/forms.py
from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from .models import Study


class StudyUploadForm(forms.Form):
    # Patient info
    first_name = forms.CharField(
        max_length=100,
        label="First name",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Patient first name",
            }
        ),
    )
    last_name = forms.CharField(
        max_length=100,
        required=False,
        label="Last name",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Patient last name (optional)",
            }
        ),
    )
    birth_date = forms.DateField(
        required=False,
        widget=forms.DateInput(
            attrs={
                "type": "date",
                "class": "form-control",
            }
        ),
        label="Birth date",
    )
    identifier = forms.CharField(
        max_length=100,
        required=False,
        label="Patient ID (clinic)",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Internal clinic ID (optional)",
            }
        ),
    )

    # Study info
    modality = forms.ChoiceField(
        choices=Study.MODALITY_CHOICES,
        label="Modality",
        widget=forms.Select(
            attrs={
                "class": "form-select",
            }
        ),
    )
    description = forms.CharField(
        max_length=255,
        required=False,
        widget=forms.Textarea(
            attrs={
                "rows": 2,
                "class": "form-control",
                "placeholder": "Short clinical description (optional)",
            }
        ),
        label="Description",
    )

    # File
    uploaded_file = forms.FileField(
        label="MRI/CT file (.nii, .nii.gz, or .zip with DICOM)",
        widget=forms.ClearableFileInput(
            attrs={
                "class": "form-control",
            }
        ),
    )


class BrainwebAuthenticationForm(AuthenticationForm):
    """
    Login form:
    - Bootstrap classlari qo‘shilgan
    - Faqat active va staff foydalanuvchilarga ruxsat beradi
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields["username"].widget.attrs.update(
            {
                "class": "form-control",
                "placeholder": "Username",
                "autocomplete": "username",
            }
        )
        self.fields["password"].widget.attrs.update(
            {
                "class": "form-control",
                "placeholder": "Password",
                "autocomplete": "current-password",
            }
        )

    def confirm_login_allowed(self, user):
        # Django'ning default tekshiruvlari (is_active va h.k.)
        super().confirm_login_allowed(user)

        # Faqat staff bo'lgan userlarga ruxsat
        if not user.is_staff:
            raise forms.ValidationError(
                "Access is restricted to clinical staff. Please contact the system administrator.",
                code="not_staff",
            )


class BrainwebUserCreationForm(UserCreationForm):
    """
    Bu formani siz faqat ichki maqsadlar uchun (masalan,
    alohida admin UI bo‘lsa) ishlatishingiz mumkin.
    Hozir self-signup o‘chirilgan, shuning uchun odatdagi
    foydalanuvchi registratsiyasi uchun ishlatilmaydi.
    """

    class Meta:
        model = User
        fields = ("username", "email")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            css_class = "form-control"
            if field_name in ("password1", "password2"):
                placeholder = "Password" if field_name == "password1" else "Repeat password"
            elif field_name == "username":
                placeholder = "Username"
            elif field_name == "email":
                placeholder = "Email (optional)"
            else:
                placeholder = ""

            field.widget.attrs.update(
                {
                    "class": css_class,
                    "placeholder": placeholder,
                }
            )
