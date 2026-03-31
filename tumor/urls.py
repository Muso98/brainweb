# tumor/urls.py
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .forms import BrainwebAuthenticationForm

app_name = "tumor"

urlpatterns = [
    # 1) Entry point - asosiy '/' ga yo'naltirish
    # - home_redirect: agar user autentifikatsiyadan o'tgan bo'lsa -> dashboard
    #   aks holda -> about sahifasi
    path("", views.home_redirect, name="home"),

    # 2) Public / informational pages (hamma uchun ochiq)
    path("about/", views.about, name="about"),
    path("health-check/", views.health_check, name="health_check"),
    path("documentation/", views.documentation, name="documentation"),
    path("faq/", views.faq, name="faq"),
    path("privacy/", views.privacy, name="privacy"),
    path("terms/", views.terms, name="terms"),
    path("contact/", views.contact, name="contact"),

    # 3) Footer form - newsletter / subscribe (POST)
    path("subscribe/", views.subscribe, name="subscribe"),

    # 4) Dashboard - require login (view ichida @login_required yoki view logic bilan)
    path("dashboard/", views.dashboard, name="dashboard"),

    # 5) Studies - list, detail, PDF report, API status endpoint
    path("studies/", views.study_list, name="study_list"),
    path("studies/<int:pk>/", views.study_detail, name="study_detail"),
    path("studies/<int:pk>/report.pdf", views.study_report_pdf, name="study_report_pdf"),
    path("api/studies/<int:pk>/status/", views.study_status_api, name="study_status_api"),

    # 6) Authentication (Login / Logout). Signup commented out as optional.
    #    LoginView uses custom BrainwebAuthenticationForm
    path(
        "accounts/login/",
        auth_views.LoginView.as_view(
            template_name="registration/login.html",
            authentication_form=BrainwebAuthenticationForm,
        ),
        name="login",
    ),
    path(
        "accounts/logout/",
        auth_views.LogoutView.as_view(),
        name="logout",
    ),
    # Uncomment / implement if you add a signup view
    # path("accounts/signup/", views.SignUpView.as_view(), name="signup"),

    # 7) Chatbot / AI API endpoint (internal use by frontend)
    path("api/chat/", views.chat_api, name="chat_api"),

    # 8) Other API endpoints (misol uchun batch analyze) - qo'shish mumkin
    # path("api/analyze/", views.api_analyze, name="api_analyze"),
]

# Eslatma:
# - Har bir nom (name="...") template va view ichida reverse/redirect bilan 'tumor:NAME' sifatida chaqiriladi.
# - views.py ichidagi view funksiyalar (home_redirect, about, documentation, faq, privacy, terms,
#   contact, subscribe, dashboard, study_list, study_detail, study_report_pdf, study_status_api,
#   chat_api) mavjud bo'lishi kerak.
