import os
import json
import logging
import google.generativeai as genai
from django.conf import settings
from .models import Study, AIResult

# ---------------------------------------------------------
# 1. SOZLAMALAR VA KONFIGURATSIYA
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


def get_gemini_model():
    """
    API kalitni xavfsiz oladi va modelni sozlaydi.
    """
    # 1. Kalitni olish (Settings yoki Environmentdan)
    api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

    if not api_key:
        logger.error("GEMINI_API_KEY topilmadi! Chat ishlamaydi.")
        return None

    # 2. Konfiguratsiya
    try:
        genai.configure(api_key=api_key)
        # Sizda aniq ishlagan versiya:
        return genai.GenerativeModel('gemini-flash-latest')
    except Exception as e:
        logger.exception("Gemini konfiguratsiyasida xatolik: %s", e)
        return None


# ---------------------------------------------------------
# 2. TOOL FUNKSIYALARI (Ma'lumot yig'uvchilar)
# ---------------------------------------------------------

def tool_list_user_studies(user) -> str:
    """Foydalanuvchining oxirgi 5 ta MRI tekshiruvini ro'yxat qiladi."""
    if not user or not user.is_authenticated:
        return "Tizimga kirmagan foydalanuvchi."

    qs = Study.objects.filter(created_by=user).order_by("-created_at")[:5]
    if not qs.exists():
        return "Sizda yuklangan MRI fayllar yo'q."

    lines = ["Foydalanuvchining oxirgi tekshiruvlari:"]
    for s in qs:
        lines.append(f"- ID: {s.id} | Sana: {s.created_at.strftime('%Y-%m-%d')} | Holati: {s.status}")
    return "\n".join(lines)


def tool_get_study_brief(study_id: int) -> str:
    """Bitta MRI tekshiruvi haqida qisqacha ma'lumot."""
    try:
        s = Study.objects.get(id=study_id)
        return (f"Study ID: {s.id}\nBemor: {s.patient}\n"
                f"Sana: {s.created_at.strftime('%Y-%m-%d')}\n"
                f"Izoh: {s.description}")
    except Study.DoesNotExist:
        return f"Study ID={study_id} bazadan topilmadi."


def tool_get_latest_study_results(user) -> str:
    """Eng oxirgi yuklangan MRI natijalarini oladi."""
    if not user or not user.is_authenticated:
        return "Tizimga kirmagan foydalanuvchi."

    s = Study.objects.filter(created_by=user).order_by("-created_at").first()
    if not s:
        return "MRI tekshiruvi topilmadi."

    lines = [f"Oxirgi MRI (ID: {s.id})", f"Bemor: {s.patient}", f"Holati: {s.status}"]

    ai_res = getattr(s, "ai_result", None)
    if ai_res:
        lines.append("--- AI TAHLILI ---")
        if ai_res.predicted_class:
            lines.append(f"Tashxis (Taxminiy): {ai_res.predicted_class}")
        if ai_res.tumor_volume_mm3:
            lines.append(f"O'sma hajmi: {ai_res.tumor_volume_mm3:.2f} mm3")
        if ai_res.predicted_confidence:
            lines.append(f"Ishonchlilik: {ai_res.predicted_confidence:.2f}")
    else:
        lines.append("AI natijalari hali generatsiya qilinmagan.")

    return "\n".join(lines)


# ---------------------------------------------------------
# 3. ASOSIY AGENT MANTIQI (RAG: Intent -> Action -> Answer)
# ---------------------------------------------------------

def run_brainweb_agent(user, user_message: str) -> str:
    """
    Foydalanuvchi so'rovini qabul qiladi, kerakli toolni ishlatadi
    va tibbiy kontekstda javob qaytaradi.
    """
    model = get_gemini_model()
    if not model:
        return "Tizim xatoligi: AI moduli sozlanmagan."

    # --- 1-BOSQICH: NIYATNI ANIQLASH (INTENT DETECTION) ---
    system_prompt_intent = """
    You are the brain of a medical MRI platform called 'BrainWeb'.
    Your task: Analyze the user's message and choose the correct action.

    Available Actions (Reply with JSON only):
    1. {"action": "list_user_studies", "args": {}} 
       -> User asks for history, list of files, "my scans".

    2. {"action": "get_latest_study_results", "args": {}} 
       -> User asks for latest result, diagnosis, analysis of recent file.

    3. {"action": "get_study_brief", "args": {"study_id": 123}} 
       -> User refers to a specific ID (e.g., "Study 5", "result of ID 10").

    4. {"action": "answer", "args": {"reply": "..."}} 
       -> General greetings, medical questions unrelated to specific data, or if you can't help.

    CRITICAL: Output MUST be valid JSON. Do not write explanations.
    """

    try:
        response = model.generate_content(f"{system_prompt_intent}\nUser Message: {user_message}")

        # JSONni tozalash
        cleaned_json = response.text.replace("```json", "").replace("```", "").strip()
        intent_data = json.loads(cleaned_json)

        action = intent_data.get("action")
        args = intent_data.get("args", {})

    except Exception as e:
        logger.warning(f"Intent parsing error: {e}")
        return "Uzr, so'rovingizni to'liq tushunmadim. Iltimos, qayta yozing."

    # --- 2-BOSQICH: AMALNI BAJARISH (TOOL EXECUTION) ---
    if action == "answer":
        return args.get("reply", "Javob yo'q.")

    tool_output = ""
    if action == "list_user_studies":
        tool_output = tool_list_user_studies(user)
    elif action == "get_latest_study_results":
        tool_output = tool_get_latest_study_results(user)
    elif action == "get_study_brief":
        s_id = args.get("study_id")
        if s_id:
            tool_output = tool_get_study_brief(int(s_id))
        else:
            tool_output = "ID raqami topilmadi."
    else:
        tool_output = "Ma'lumot topilmadi."

    # --- 3-BOSQICH: YAKUNIY JAVOB (FINAL GENERATION) ---
    final_prompt = f"""
    Context: You are an AI assistant for a brain tumor detection system.

    User Question: {user_message}
    System Data (Retrieved Info):
    {tool_output}

    INSTRUCTIONS:
    1. Language: Answer in the SAME language as the User Question (Uzbek/Russian/English). Default to Uzbek.
    2. Tone: Professional, Empathetic, Medical but Cautious.
    3. Disclaimer: YOU MUST STATE that you are an AI and this is PRELIMINARY analysis. Users must consult a doctor.
    4. Content: Summarize the 'System Data' clearly for the user. If 'predicted_class' suggests a tumor, use words like "ehtimol", "taxminan", "indicates possible".
    """

    try:
        final_resp = model.generate_content(final_prompt)
        return final_resp.text
    except Exception as e:
        logger.error(f"Final generation error: {e}")
        return "Javob tayyorlashda texnik xatolik yuz berdi."