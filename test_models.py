import google.generativeai as genai

# O'zingizning API kalitingizni qo'ying
API_KEY = "AIzaSyA0wiBhXOcpR2dg3XTHndV96NORTAtBRTI"

genai.configure(api_key=API_KEY)

print("🔍 Mavjud modellar qidirilmoqda...\n")

try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Topildi: {m.name}")
            available_models.append(m.name)

    if not available_models:
        print("\n❌ Hech qanday model topilmadi. API kalitda muammo bo'lishi mumkin.")
    else:
        print("\n🎉 Model nomlari aniqlandi! Shulardan birini kodda ishlatish kerak.")

except Exception as e:
    print(f"\n❌ XATOLIK: {e}")