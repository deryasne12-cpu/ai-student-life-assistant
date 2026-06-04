
import io
import sqlite3
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


st.set_page_config(
    page_title="AI Student Performance Tracker",
    page_icon="🎓",
    layout="wide",
)


TRANSLATIONS = {
    "Türkçe": {
        "app_title": "🎓 Yapay Zeka Öğrenci Performansı ve Sağlık Takipçisi",
        "subtitle": "Öğrenci verimliliği, uyku, sağlık, beslenme, egzersiz, haftalık raporlar, yapay zeka koçluğu, veritabanı takibi ve gelecekteki akıllı entegrasyonlar için tam kapsamlı bir yapay zeka paneli.",
        "concept_title": "Proje Konsepti",
        "concept_text": "Bu platform, öğrenci performansını zaman içinde takip eder. Sadece günlük bir girdiyi analiz etmiyor. Akademik performans, uyku alışkanlıkları, stres seviyesi, beslenme, egzersiz ve yapay zeka tabanlı önerileri birleştirir.",
        "sidebar_title": "Kontrol Paneli",
        "login": "Giriş / Login",
        "settings": "Ayarlar",
        "db": "Veritabanı Geçmişi",
        "language": "Dil / Language",
        "active_language": "Aktif Dil",
        "motivation": "Disiplin bugün, başarı yarın.",
        "you_can": "Sen yaparsın.",
        "logout": "Çıkış / Logout",
        "profile_title": "👤 Öğrenci Girişi ve Profili",
        "full_name": "Tam isim",
        "student_id": "Öğrenci Kimliği",
        "faculty": "Fakülte",
        "semester": "Dönem",
        "age": "Yaş",
        "main_goal": "Ana Hedef",
        "height": "Boy (cm)",
        "weight": "Ağırlık (kg)",
        "save_profile": "💾 Öğrenci Profilini Kaydet",
        "profile_saved": "Öğrenci profili başarıyla kaydedildi ve SQLite veritabanına işlendi.",
        "student_status": "Öğrenci Durumu",
        "daily_motivation": "✨ Günlük Motivasyon",
        "daily_tracking": "📅 Günlük",
        "nutrition": "🥗 Beslenme",
        "exercise": "🎯 Egzersiz",
        "analytics": "📊 Analiz",
        "coach": "🤖 AI Koç",
        "report": "📄 Rapor",
        "smart": "⌚ Akıllı",
        "theme": "Dashboard Temasını Seç",
        "background": "Arka Plan Modu",
        "background_soft": "Yumuşak Çoklu Renk",
        "background_dark": "Koyu",
        "background_light": "Açık / Beyaz",
        "background_green": "Yeşil Sağlık",
        "background_orange": "Turuncu Enerji",
        "settings_title": "⚙️ Ayarlar ve Dashboard Kontrolü",
        "settings_text": "Buradan tema, arka plan, dil ve oturum kontrolü yönetilir.",
        "daily_title": "📅 Günlük Öğrenci Takibi",
        "mood_question": "Bugün nasıl hissediyorsun?",
        "sleep": "Uyku Saatleri",
        "study": "Çalışma Saatleri",
        "task": "Görev Tamamlama (%)",
        "focus": "Odak Seviyesi",
        "stress": "Stres Seviyesi",
        "exercise_min": "Egzersiz Dakikaları",
        "water": "Su Tüketimi (Litre)",
        "nutrition_quality": "Beslenme Kalitesi",
        "save_day": "🚀 Bugünü Kaydet ve Yapay Zeka Analizi Oluştur",
        "no_records": "Henüz veritabanında kayıt yok. Önce günlük kayıt kaydet.",
    },
    "English": {
        "app_title": "🎓 AI Student Performance & Wellness Tracker",
        "subtitle": "A complete AI dashboard for productivity, sleep, health, nutrition, exercise, weekly reports, AI coaching, database tracking and future smart integrations.",
        "concept_title": "Project Concept",
        "concept_text": "This platform tracks student performance over time. It does not only analyze one daily input. It combines academic performance, sleep habits, stress level, nutrition, exercise and AI-based recommendations.",
        "sidebar_title": "Control Panel",
        "login": "Login / Profile",
        "settings": "Settings",
        "db": "Database History",
        "language": "Language",
        "active_language": "Active Language",
        "motivation": "Discipline today, success tomorrow.",
        "you_can": "You can do it.",
        "logout": "Logout",
        "profile_title": "👤 Student Login & Profile",
        "full_name": "Full Name",
        "student_id": "Student ID",
        "faculty": "Faculty",
        "semester": "Semester",
        "age": "Age",
        "main_goal": "Main Goal",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "save_profile": "💾 Save Student Profile",
        "profile_saved": "Student profile saved successfully to SQLite database.",
        "student_status": "Student Status",
        "daily_motivation": "✨ Daily Motivation",
        "daily_tracking": "📅 Daily",
        "nutrition": "🥗 Nutrition",
        "exercise": "🎯 Exercise",
        "analytics": "📊 Insights",
        "coach": "🤖 AI Coach",
        "report": "📄 Report",
        "smart": "⌚ Smart",
        "theme": "Choose Dashboard Theme",
        "background": "Background Mode",
        "background_soft": "Soft Multi Color",
        "background_dark": "Dark",
        "background_light": "Light / White",
        "background_green": "Green Health",
        "background_orange": "Orange Energy",
        "settings_title": "⚙️ Settings & Dashboard Control",
        "settings_text": "Manage theme, background, language and session control here.",
        "daily_title": "📅 Daily Student Tracking",
        "mood_question": "How do you feel today?",
        "sleep": "Sleep Hours",
        "study": "Study Hours",
        "task": "Task Completion (%)",
        "focus": "Focus Level",
        "stress": "Stress Level",
        "exercise_min": "Exercise Minutes",
        "water": "Water Intake (Liters)",
        "nutrition_quality": "Nutrition Quality",
        "save_day": "🚀 Save Today and Generate AI Analysis",
        "no_records": "No saved records found in the database yet. Save a daily record first.",
    },
    "Deutsch": {
        "app_title": "🎓 KI-Tracker für Studentenleistung und Gesundheit",
        "subtitle": "Ein vollständiges KI-Dashboard für Produktivität, Schlaf, Gesundheit, Ernährung, Training, Wochenberichte, KI-Coaching, Datenbankverfolgung und zukünftige Smart-Integrationen.",
        "concept_title": "Projektkonzept",
        "concept_text": "Diese Plattform verfolgt die Leistung von Studierenden über die Zeit. Sie analysiert nicht nur einen täglichen Eintrag, sondern kombiniert akademische Leistung, Schlaf, Stress, Ernährung, Training und KI-Empfehlungen.",
        "sidebar_title": "Kontrollpanel",
        "login": "Login / Profil",
        "settings": "Einstellungen",
        "db": "Datenbankverlauf",
        "language": "Sprache",
        "active_language": "Aktive Sprache",
        "motivation": "Disziplin heute, Erfolg morgen.",
        "you_can": "Du schaffst das.",
        "logout": "Abmelden",
        "profile_title": "👤 Studentenlogin und Profil",
        "full_name": "Vollständiger Name",
        "student_id": "Studenten-ID",
        "faculty": "Fakultät",
        "semester": "Semester",
        "age": "Alter",
        "main_goal": "Hauptziel",
        "height": "Größe (cm)",
        "weight": "Gewicht (kg)",
        "save_profile": "💾 Profil speichern",
        "profile_saved": "Profil erfolgreich gespeichert.",
        "student_status": "Studentenstatus",
        "daily_motivation": "✨ Tägliche Motivation",
        "daily_tracking": "📅 Alltag",
        "nutrition": "🥗 Ernährung",
        "exercise": "🎯 Training",
        "analytics": "📊 Analyse",
        "coach": "🤖 KI-Coach",
        "report": "📄 Bericht",
        "smart": "⌚ Smart",
        "theme": "Dashboard-Thema wählen",
        "background": "Hintergrundmodus",
        "background_soft": "Sanfte Mehrfarben",
        "background_dark": "Dunkel",
        "background_light": "Hell / Weiß",
        "background_green": "Grüne Gesundheit",
        "background_orange": "Orange Energie",
        "settings_title": "⚙️ Einstellungen und Dashboard-Kontrolle",
        "settings_text": "Hier werden Thema, Hintergrund, Sprache und Sitzung gesteuert.",
        "daily_title": "📅 Tägliches Studententracking",
        "mood_question": "Wie fühlst du dich heute?",
        "sleep": "Schlafstunden",
        "study": "Lernstunden",
        "task": "Aufgabenerfüllung (%)",
        "focus": "Fokuslevel",
        "stress": "Stresslevel",
        "exercise_min": "Trainingsminuten",
        "water": "Wasseraufnahme (Liter)",
        "nutrition_quality": "Ernährungsqualität",
        "save_day": "🚀 Heute speichern und KI-Analyse erstellen",
        "no_records": "Keine gespeicherten Einträge gefunden. Speichere zuerst einen Tagesdatensatz.",
    },
    "Русский": {
        "app_title": "🎓 ИИ-трекер успеваемости и здоровья студента",
        "subtitle": "Полная ИИ-панель для продуктивности, сна, здоровья, питания, тренировок, еженедельных отчетов, ИИ-коучинга, базы данных и будущих smart-интеграций.",
        "concept_title": "Концепция проекта",
        "concept_text": "Эта платформа отслеживает прогресс студента во времени. Она анализирует не только один дневной ввод, а объединяет учебу, сон, стресс, питание, упражнения и рекомендации ИИ.",
        "sidebar_title": "Панель управления",
        "login": "Вход / Профиль",
        "settings": "Настройки",
        "db": "История базы данных",
        "language": "Язык",
        "active_language": "Активный язык",
        "motivation": "Дисциплина сегодня, успех завтра.",
        "you_can": "Ты справишься.",
        "logout": "Выйти",
        "profile_title": "👤 Вход и профиль студента",
        "full_name": "Полное имя",
        "student_id": "ID студента",
        "faculty": "Факультет",
        "semester": "Семестр",
        "age": "Возраст",
        "main_goal": "Главная цель",
        "height": "Рост (см)",
        "weight": "Вес (кг)",
        "save_profile": "💾 Сохранить профиль",
        "profile_saved": "Профиль успешно сохранен.",
        "student_status": "Статус студента",
        "daily_motivation": "✨ Мотивация дня",
        "daily_tracking": "📅 День",
        "nutrition": "🥗 Питание",
        "exercise": "🎯 Тренировка",
        "analytics": "📊 Анализ",
        "coach": "🤖 ИИ-коуч",
        "report": "📄 Отчет",
        "smart": "⌚ Smart",
        "theme": "Выберите тему панели",
        "background": "Режим фона",
        "background_soft": "Мягкие цвета",
        "background_dark": "Темный",
        "background_light": "Светлый / Белый",
        "background_green": "Зеленое здоровье",
        "background_orange": "Оранжевая энергия",
        "settings_title": "⚙️ Настройки и управление панелью",
        "settings_text": "Здесь можно управлять темой, фоном, языком и сессией.",
        "daily_title": "📅 Ежедневный трекинг студента",
        "mood_question": "Как ты себя чувствуешь сегодня?",
        "sleep": "Часы сна",
        "study": "Часы учебы",
        "task": "Выполнение задач (%)",
        "focus": "Уровень фокуса",
        "stress": "Уровень стресса",
        "exercise_min": "Минуты упражнений",
        "water": "Вода (литры)",
        "nutrition_quality": "Качество питания",
        "save_day": "🚀 Сохранить день и создать ИИ-анализ",
        "no_records": "Записей в базе пока нет. Сначала сохраните дневную запись.",
    },
    "Español": {
        "app_title": "🎓 Rastreador de rendimiento y salud estudiantil con IA",
        "subtitle": "Un panel completo de IA para productividad, sueño, salud, nutrición, ejercicio, informes semanales, coaching con IA, base de datos e integraciones inteligentes futuras.",
        "concept_title": "Concepto del proyecto",
        "concept_text": "Esta plataforma sigue el rendimiento del estudiante a lo largo del tiempo. No analiza solo una entrada diaria; combina rendimiento académico, sueño, estrés, nutrición, ejercicio y recomendaciones de IA.",
        "sidebar_title": "Panel de control",
        "login": "Login / Perfil",
        "settings": "Configuración",
        "db": "Historial de base de datos",
        "language": "Idioma",
        "active_language": "Idioma activo",
        "motivation": "Disciplina hoy, éxito mañana.",
        "you_can": "Tú puedes.",
        "logout": "Cerrar sesión",
        "profile_title": "👤 Login y perfil del estudiante",
        "full_name": "Nombre completo",
        "student_id": "ID del estudiante",
        "faculty": "Facultad",
        "semester": "Semestre",
        "age": "Edad",
        "main_goal": "Objetivo principal",
        "height": "Altura (cm)",
        "weight": "Peso (kg)",
        "save_profile": "💾 Guardar perfil",
        "profile_saved": "Perfil guardado correctamente.",
        "student_status": "Estado del estudiante",
        "daily_motivation": "✨ Motivación diaria",
        "daily_tracking": "📅 Diario",
        "nutrition": "🥗 Nutrición",
        "exercise": "🎯 Ejercicio",
        "analytics": "📊 Análisis",
        "coach": "🤖 Coach IA",
        "report": "📄 Informe",
        "smart": "⌚ Smart",
        "theme": "Elegir tema del dashboard",
        "background": "Modo de fondo",
        "background_soft": "Colores suaves",
        "background_dark": "Oscuro",
        "background_light": "Claro / Blanco",
        "background_green": "Salud verde",
        "background_orange": "Energía naranja",
        "settings_title": "⚙️ Configuración y control del dashboard",
        "settings_text": "Gestiona tema, fondo, idioma y sesión aquí.",
        "daily_title": "📅 Seguimiento diario del estudiante",
        "mood_question": "¿Cómo te sientes hoy?",
        "sleep": "Horas de sueño",
        "study": "Horas de estudio",
        "task": "Tareas completadas (%)",
        "focus": "Nivel de enfoque",
        "stress": "Nivel de estrés",
        "exercise_min": "Minutos de ejercicio",
        "water": "Agua (litros)",
        "nutrition_quality": "Calidad de nutrición",
        "save_day": "🚀 Guardar hoy y generar análisis IA",
        "no_records": "No hay registros guardados. Guarda primero un registro diario.",
    },
}


EXTRA_TRANSLATIONS = {
    "Türkçe": {
        "home": "🏠 Ana Sayfa",
        "current_student_status": "Mevcut Öğrenci Durumu",
        "daily_motivation_quote": "✨ Günlük Motivasyon Sözü",
        "excellent": "Mükemmel",
        "stable": "Dengeli",
        "needs_attention": "Dikkat Gerekiyor",
        "improving": "Gelişiyor",
        "no_tracking_data": "Henüz takip verisi yok. Lütfen önce günlük kayıt kaydet.",
        "productivity_strong": "verimlilik seviyen güçlü",
        "productivity_moderate": "verimlilik seviyen orta düzeyde",
        "productivity_needs": "verimlilik seviyen geliştirme gerektiriyor",
        "stress_high": "stres seviyen yüksek, bu yüzden toparlanma süresi artırılmalı",
        "stress_watch": "stres seviyen yönetilebilir ama yine de takip edilmeli",
        "stress_under_control": "stres seviyen şu anda kontrol altında",
        "sleep_low_note": "uyku süren düşük ve odaklanmanı azaltabilir",
        "sleep_ok_note": "uyku süren kabul edilebilir görünüyor",
        "exercise_low_note": "egzersiz aktiviten düşük, hafif yürüyüş veya mobilite çalışması önerilir",
        "exercise_ok_note": "egzersiz aktiviten sağlığını destekliyor",
        "water_low_note": "su tüketimin düşük, gün içinde daha fazla su iç",
        "water_ok_note": "su tüketimin kabul edilebilir görünüyor",
        "status_note_template": "{name}, kaydedilen takip verilerine göre {productivity_text}. Ortalama uyku süren {avg_sleep:.1f} saat, ortalama çalışma süren {avg_study:.1f} saat ve ortalama sağlık skorun {avg_wellness:.1f}/100. Şu anda {sleep_text}; {stress_text}; {exercise_text}; ve {water_text}. Önerim: düzenli bir günlük rutin kur, aşırı yüklenmeden kaçın ve gelişimini istikrarlı şekilde takip et.",
        "quote_productive": "Disiplin, iyi günleri ilerlemeye; kötü günleri derse çevirir.",
        "quote_stress": "Yavaşla, toparlan ve devam et. Sürdürülebilir ilerleme tükenmişlikten daha değerlidir.",
        "quote_sleep": "Daha iyi uyku boşa geçen zaman değildir; daha iyi performansın yakıtıdır.",
        "quote_default": "Küçük ama istikrarlı adımlar güçlü uzun vadeli sonuçlar üretir.",
        "sleep_low": "Uyku düşük. En az 1 saat artırmaya çalış.",
        "sleep_ok": "Uyku süresi kabul edilebilir.",
        "stress_high_msg": "Stres trendi yüksek. Toparlanma süresi ekle ve iş yükü yoğunluğunu azalt.",
        "stress_ok": "Stres seviyesi yönetilebilir.",
        "nutrition_low": "Beslenme kalitesi düşük. Protein, meyve, sebze ve düzenli öğün ekle.",
        "nutrition_ok": "Beslenme kalitesi kabul edilebilir.",
        "exercise_low": "Egzersiz aktivitesi düşük. Yürüyüş veya mobilite seansları ekle.",
        "exercise_ok": "Egzersiz sağlık ve odağı destekliyor.",
        "water_low": "Su tüketimi düşük. Daha iyi odak için hidrasyonu artır.",
        "water_ok": "Hidrasyon seviyesi kabul edilebilir.",
        "productivity_strong_msg": "Verimlilik trendi güçlü. Derin çalışma seansları artırılabilir.",
        "productivity_moderate_msg": "Verimlilik trendi orta düzeyde. Daha yapılandırılmış haftalık plan kullan.",
        "productivity_low_msg": "Verimlilik geliştirilmeli. Daha küçük görevler ve kısa çalışma bloklarıyla başla.",
        "model_feature_importance": "Model Özellik Önemi",
        "current_student_status_short": "Mevcut Öğrenci Durumu",
        "records_loaded": "{count} kayıt SQLite veritabanından yüklendi.",
    },
    "English": {
        "home": "🏠 Home",
        "current_student_status": "Current Student Status",
        "daily_motivation_quote": "✨ Daily Motivation Quote",
        "excellent": "Excellent",
        "stable": "Stable",
        "needs_attention": "Needs Attention",
        "improving": "Improving",
        "no_tracking_data": "No tracking data is available yet. Please save daily records first.",
        "productivity_strong": "your productivity level is strong",
        "productivity_moderate": "your productivity level is moderate",
        "productivity_needs": "your productivity level needs improvement",
        "stress_high": "your stress level is high, so recovery time should be increased",
        "stress_watch": "your stress level is manageable but should still be watched",
        "stress_under_control": "your stress level is currently under control",
        "sleep_low_note": "your sleep duration is low and may reduce focus",
        "sleep_ok_note": "your sleep duration looks acceptable",
        "exercise_low_note": "your exercise activity is low, so light walking or mobility work is recommended",
        "exercise_ok_note": "your exercise activity supports your wellness",
        "water_low_note": "your hydration is low, so drink more water during the day",
        "water_ok_note": "your hydration looks acceptable",
        "status_note_template": "{name}, based on your saved tracking data, {productivity_text}. Your average sleep is {avg_sleep:.1f} hours, your average study time is {avg_study:.1f} hours, and your average wellness score is {avg_wellness:.1f}/100. Currently, {sleep_text}; {stress_text}; {exercise_text}; and {water_text}. My recommendation is to keep a stable daily routine, avoid overload, and track your progress consistently.",
        "quote_productive": "Discipline turns good days into progress and bad days into lessons.",
        "quote_stress": "Slow down, reset, and continue. Sustainable progress beats burnout.",
        "quote_sleep": "Better sleep is not wasted time; it is fuel for better performance.",
        "quote_default": "Small consistent steps create strong long-term results.",
        "sleep_low": "Sleep is low. Try to increase sleep by at least 1 hour.",
        "sleep_ok": "Sleep duration is acceptable.",
        "stress_high_msg": "Stress trend is high. Add recovery time and reduce workload intensity.",
        "stress_ok": "Stress level is manageable.",
        "nutrition_low": "Nutrition quality is low. Add protein, fruit, vegetables and stable meals.",
        "nutrition_ok": "Nutrition quality is acceptable.",
        "exercise_low": "Exercise activity is low. Add walking or mobility sessions.",
        "exercise_ok": "Exercise supports wellness and focus.",
        "water_low": "Water intake is low. Increase hydration for better focus.",
        "water_ok": "Hydration level is acceptable.",
        "productivity_strong_msg": "Productivity trend is strong. Deep work sessions can be increased.",
        "productivity_moderate_msg": "Productivity trend is moderate. Use a more structured weekly plan.",
        "productivity_low_msg": "Productivity needs improvement. Start with smaller tasks and shorter study blocks.",
        "model_feature_importance": "Model Feature Importance",
        "current_student_status_short": "Current Student Status",
        "records_loaded": "{count} records loaded from SQLite database.",
    },
    "Deutsch": {
        "home": "🏠 Startseite",
        "current_student_status": "Aktueller Studentenstatus",
        "daily_motivation_quote": "✨ Tägliches Motivationszitat",
        "excellent": "Ausgezeichnet",
        "stable": "Stabil",
        "needs_attention": "Braucht Aufmerksamkeit",
        "improving": "Verbessert sich",
        "no_tracking_data": "Noch keine Tracking-Daten verfügbar. Bitte zuerst einen Tagesdatensatz speichern.",
        "productivity_strong": "dein Produktivitätsniveau ist stark",
        "productivity_moderate": "dein Produktivitätsniveau ist mittelmäßig",
        "productivity_needs": "dein Produktivitätsniveau muss verbessert werden",
        "stress_high": "dein Stressniveau ist hoch, daher sollte die Erholungszeit erhöht werden",
        "stress_watch": "dein Stressniveau ist kontrollierbar, sollte aber weiter beobachtet werden",
        "stress_under_control": "dein Stressniveau ist derzeit unter Kontrolle",
        "sleep_low_note": "deine Schlafdauer ist niedrig und kann den Fokus verringern",
        "sleep_ok_note": "deine Schlafdauer sieht akzeptabel aus",
        "exercise_low_note": "deine Trainingsaktivität ist niedrig, leichtes Gehen oder Mobilitätsarbeit wird empfohlen",
        "exercise_ok_note": "deine Trainingsaktivität unterstützt dein Wohlbefinden",
        "water_low_note": "deine Wasseraufnahme ist niedrig, trinke tagsüber mehr Wasser",
        "water_ok_note": "deine Wasseraufnahme sieht akzeptabel aus",
        "status_note_template": "{name}, basierend auf deinen gespeicherten Tracking-Daten: {productivity_text}. Deine durchschnittliche Schlafdauer beträgt {avg_sleep:.1f} Stunden, deine durchschnittliche Lernzeit {avg_study:.1f} Stunden und dein durchschnittlicher Wellness-Score {avg_wellness:.1f}/100. Aktuell gilt: {sleep_text}; {stress_text}; {exercise_text}; und {water_text}. Meine Empfehlung: Halte eine stabile Tagesroutine ein, vermeide Überlastung und verfolge deinen Fortschritt konsequent.",
        "quote_productive": "Disziplin verwandelt gute Tage in Fortschritt und schlechte Tage in Lektionen.",
        "quote_stress": "Verlangsame, setze zurück und mach weiter. Nachhaltiger Fortschritt ist besser als Burnout.",
        "quote_sleep": "Besserer Schlaf ist keine verlorene Zeit; er ist Treibstoff für bessere Leistung.",
        "quote_default": "Kleine konsequente Schritte schaffen starke langfristige Ergebnisse.",
        "sleep_low": "Der Schlaf ist niedrig. Versuche, ihn um mindestens 1 Stunde zu erhöhen.",
        "sleep_ok": "Die Schlafdauer ist akzeptabel.",
        "stress_high_msg": "Der Stresstrend ist hoch. Füge Erholungszeit hinzu und reduziere die Arbeitsintensität.",
        "stress_ok": "Das Stressniveau ist kontrollierbar.",
        "nutrition_low": "Die Ernährungsqualität ist niedrig. Ergänze Protein, Obst, Gemüse und regelmäßige Mahlzeiten.",
        "nutrition_ok": "Die Ernährungsqualität ist akzeptabel.",
        "exercise_low": "Die Trainingsaktivität ist niedrig. Füge Geh- oder Mobilitätseinheiten hinzu.",
        "exercise_ok": "Training unterstützt Wohlbefinden und Fokus.",
        "water_low": "Die Wasseraufnahme ist niedrig. Erhöhe die Hydration für besseren Fokus.",
        "water_ok": "Die Hydration ist akzeptabel.",
        "productivity_strong_msg": "Der Produktivitätstrend ist stark. Deep-Work-Einheiten können erhöht werden.",
        "productivity_moderate_msg": "Der Produktivitätstrend ist mittelmäßig. Nutze einen strukturierteren Wochenplan.",
        "productivity_low_msg": "Die Produktivität muss verbessert werden. Beginne mit kleineren Aufgaben und kürzeren Lernblöcken.",
        "model_feature_importance": "Modell-Merkmalswichtigkeit",
        "current_student_status_short": "Aktueller Studentenstatus",
        "records_loaded": "{count} Datensätze aus der SQLite-Datenbank geladen.",
    },
    "Русский": {
        "home": "🏠 Главная",
        "current_student_status": "Текущий статус студента",
        "daily_motivation_quote": "✨ Ежедневная мотивационная цитата",
        "excellent": "Отлично",
        "stable": "Стабильно",
        "needs_attention": "Требует внимания",
        "improving": "Улучшается",
        "no_tracking_data": "Данных отслеживания пока нет. Сначала сохраните дневную запись.",
        "productivity_strong": "уровень продуктивности высокий",
        "productivity_moderate": "уровень продуктивности средний",
        "productivity_needs": "уровень продуктивности требует улучшения",
        "stress_high": "уровень стресса высокий, нужно увеличить время восстановления",
        "stress_watch": "уровень стресса управляемый, но его нужно отслеживать",
        "stress_under_control": "уровень стресса сейчас под контролем",
        "sleep_low_note": "сон недостаточный и может снижать концентрацию",
        "sleep_ok_note": "длительность сна выглядит приемлемой",
        "exercise_low_note": "физическая активность низкая, рекомендуется легкая ходьба или мобильность",
        "exercise_ok_note": "физическая активность поддерживает самочувствие",
        "water_low_note": "гидратация низкая, пейте больше воды в течение дня",
        "water_ok_note": "гидратация выглядит приемлемой",
        "status_note_template": "{name}, по сохраненным данным отслеживания {productivity_text}. Средний сон: {avg_sleep:.1f} ч, среднее время учебы: {avg_study:.1f} ч, средний wellness-score: {avg_wellness:.1f}/100. Сейчас: {sleep_text}; {stress_text}; {exercise_text}; и {water_text}. Моя рекомендация: держать стабильный режим, избегать перегрузки и регулярно отслеживать прогресс.",
        "quote_productive": "Дисциплина превращает хорошие дни в прогресс, а плохие — в уроки.",
        "quote_stress": "Замедлись, восстановись и продолжай. Устойчивый прогресс лучше выгорания.",
        "quote_sleep": "Хороший сон — не потерянное время; это топливо для лучшей работы.",
        "quote_default": "Маленькие последовательные шаги создают сильные долгосрочные результаты.",
        "sleep_low": "Сна мало. Постарайтесь увеличить сон хотя бы на 1 час.",
        "sleep_ok": "Длительность сна приемлемая.",
        "stress_high_msg": "Тренд стресса высокий. Добавьте время восстановления и снизьте нагрузку.",
        "stress_ok": "Уровень стресса управляемый.",
        "nutrition_low": "Качество питания низкое. Добавьте белок, фрукты, овощи и стабильные приемы пищи.",
        "nutrition_ok": "Качество питания приемлемое.",
        "exercise_low": "Физическая активность низкая. Добавьте прогулки или мобильность.",
        "exercise_ok": "Упражнения поддерживают самочувствие и фокус.",
        "water_low": "Потребление воды низкое. Увеличьте гидратацию для лучшей концентрации.",
        "water_ok": "Уровень гидратации приемлемый.",
        "productivity_strong_msg": "Тренд продуктивности сильный. Можно увеличить сессии глубокой работы.",
        "productivity_moderate_msg": "Тренд продуктивности средний. Используйте более структурированный недельный план.",
        "productivity_low_msg": "Продуктивность нужно улучшить. Начните с маленьких задач и коротких учебных блоков.",
        "model_feature_importance": "Важность признаков модели",
        "current_student_status_short": "Текущий статус студента",
        "records_loaded": "{count} записей загружено из базы SQLite.",
    },
    "Español": {
        "home": "🏠 Inicio",
        "current_student_status": "Estado actual del estudiante",
        "daily_motivation_quote": "✨ Frase de motivación diaria",
        "excellent": "Excelente",
        "stable": "Estable",
        "needs_attention": "Necesita atención",
        "improving": "Mejorando",
        "no_tracking_data": "Todavía no hay datos de seguimiento. Guarda primero un registro diario.",
        "productivity_strong": "tu nivel de productividad es fuerte",
        "productivity_moderate": "tu nivel de productividad es moderado",
        "productivity_needs": "tu nivel de productividad necesita mejorar",
        "stress_high": "tu nivel de estrés es alto, por eso debe aumentarse el tiempo de recuperación",
        "stress_watch": "tu nivel de estrés es manejable, pero debe seguir vigilándose",
        "stress_under_control": "tu nivel de estrés está actualmente bajo control",
        "sleep_low_note": "tu duración de sueño es baja y puede reducir el enfoque",
        "sleep_ok_note": "tu duración de sueño parece aceptable",
        "exercise_low_note": "tu actividad física es baja, se recomienda caminar suave o movilidad",
        "exercise_ok_note": "tu actividad física apoya tu bienestar",
        "water_low_note": "tu hidratación es baja, bebe más agua durante el día",
        "water_ok_note": "tu hidratación parece aceptable",
        "status_note_template": "{name}, según tus datos guardados, {productivity_text}. Tu sueño promedio es {avg_sleep:.1f} horas, tu tiempo promedio de estudio es {avg_study:.1f} horas y tu puntuación promedio de bienestar es {avg_wellness:.1f}/100. Actualmente: {sleep_text}; {stress_text}; {exercise_text}; y {water_text}. Mi recomendación es mantener una rutina diaria estable, evitar la sobrecarga y seguir tu progreso de forma constante.",
        "quote_productive": "La disciplina convierte los buenos días en progreso y los malos en lecciones.",
        "quote_stress": "Baja el ritmo, reinicia y continúa. El progreso sostenible supera al agotamiento.",
        "quote_sleep": "Dormir mejor no es tiempo perdido; es combustible para rendir mejor.",
        "quote_default": "Pequeños pasos constantes crean grandes resultados a largo plazo.",
        "sleep_low": "El sueño es bajo. Intenta aumentarlo al menos 1 hora.",
        "sleep_ok": "La duración del sueño es aceptable.",
        "stress_high_msg": "La tendencia de estrés es alta. Añade recuperación y reduce la intensidad de trabajo.",
        "stress_ok": "El nivel de estrés es manejable.",
        "nutrition_low": "La calidad nutricional es baja. Añade proteína, fruta, verduras y comidas estables.",
        "nutrition_ok": "La calidad nutricional es aceptable.",
        "exercise_low": "La actividad física es baja. Añade caminatas o sesiones de movilidad.",
        "exercise_ok": "El ejercicio apoya el bienestar y el enfoque.",
        "water_low": "La ingesta de agua es baja. Aumenta la hidratación para mejorar el enfoque.",
        "water_ok": "El nivel de hidratación es aceptable.",
        "productivity_strong_msg": "La tendencia de productividad es fuerte. Se pueden aumentar las sesiones de trabajo profundo.",
        "productivity_moderate_msg": "La tendencia de productividad es moderada. Usa un plan semanal más estructurado.",
        "productivity_low_msg": "La productividad necesita mejorar. Empieza con tareas pequeñas y bloques cortos.",
        "model_feature_importance": "Importancia de características del modelo",
        "current_student_status_short": "Estado actual del estudiante",
        "records_loaded": "{count} registros cargados desde la base de datos SQLite.",
    },
}

for _lang, _extra in EXTRA_TRANSLATIONS.items():
    TRANSLATIONS[_lang].update(_extra)


EXTRA_TRANSLATIONS_ALL_TABS = {
    "Türkçe": {
        "date_label":"Tarih", "default_mood":"Bugün odaklanmış ve çalışmaya hazırım.", "ai_analysis":"Yapay Zeka Analizi",
        "mood_metric":"Ruh Hali", "status_metric":"Durum", "productivity_score":"Verimlilik Skoru", "wellness_score":"Sağlık Skoru", "risk_level":"Risk Seviyesi",
        "positive":"Pozitif", "negative":"Negatif", "neutral":"Nötr", "high_risk":"Yüksek Risk", "medium_risk":"Orta Risk", "low_risk":"Düşük Risk",
        "ai_low_advice":"AI Koçu: İş yükünü azalt, su iç, sağlıklı bir öğün ye ve küçük bir görevi tamamla.",
        "ai_medium_advice":"AI Koçu: Orta düzey görevlerle devam et, çoklu görevden kaçın ve sabit ritmini koru.",
        "ai_high_advice":"AI Koçu: Verimlilik durumun güçlü. Zor görevlerle başla ve derin çalışma yap.",
        "plan_low":"25 dk hafif çalışma -> 5 dk mola -> 20 dk tekrar", "plan_medium":"45 dk çalışma -> 10 dk mola -> 45 dk pratik", "plan_high":"60 dk derin çalışma -> 10 dk mola -> 60 dk odaklı çalışma",
        "suggested_daily_plan":"Önerilen Günlük Plan", "chart_value":"Değer", "chart_sleep":"Uyku", "chart_study":"Çalışma", "chart_focus":"Odak", "chart_stress":"Stres", "chart_exercise":"Egzersiz / 10", "chart_tasks":"Görevler / 10", "chart_water":"Su", "chart_nutrition":"Beslenme", "chart_sentiment":"Duygu x10",
        "bmi_status":"BMI Durumu", "goal":"Hedef", "food_category":"Kategori", "food_examples":"Örnekler", "food_purpose":"Amaç", "protein":"Protein", "carbohydrate":"Karbonhidrat", "healthy_fat":"Sağlıklı Yağ", "fruit":"Meyve", "vegetable":"Sebze", "hydration":"Hidrasyon",
        "protein_examples":"Yumurta, tavuk, balık, yoğurt, mercimek", "carb_examples":"Pirinç, yulaf, patates, tam tahıllı ekmek", "fat_examples":"Zeytinyağı, avokado, kuruyemiş, fıstık ezmesi", "fruit_examples":"Muz, elma, orman meyveleri, portakal", "veg_examples":"Brokoli, ıspanak, salata, havuç", "hydration_examples":"Su, maden suyu, şekersiz çay",
        "protein_purpose":"Kas onarımı ve tokluk", "carb_purpose":"Ders ve antrenman için enerji", "fat_purpose":"Hormonal sağlık ve uzun süreli enerji", "fruit_purpose":"Vitaminler ve hızlı enerji", "veg_purpose":"Mikro besinler ve sindirim", "hydration_purpose":"Odak, toparlanma ve duygu dengesi",
        "exercise_goal":"Egzersiz Hedefi", "general_health":"Genel Sağlık", "weight_gain_muscle":"Kilo Alma / Kas", "fat_loss":"Yağ Kaybı", "stress_reduction":"Stres Azaltma", "posture_mobility":"Duruş ve Mobilite", "exercise_plan_text":"Bu egzersiz planı fiziksel sağlık, odak, toparlanma ve öğrenci verimliliğini destekler.",
        "plan_general":["Pazartesi: 30 dk yürüyüş + 10 dk esneme", "Çarşamba: Tüm vücut ağırlık antrenmanı", "Cuma: 30 dk bisiklet veya yürüyüş", "Pazar: Hafif mobilite ve toparlanma"],
        "plan_muscle":["Pazartesi: Push antrenmanı", "Salı: Pull antrenmanı", "Perşembe: Bacak", "Cumartesi: Tüm vücut güç antrenmanı"],
        "plan_fat":["Pazartesi: 40 dk tempolu yürüyüş", "Çarşamba: Tüm vücut circuit", "Cuma: Interval kardiyo", "Pazar: Uzun yürüyüş"],
        "plan_stress":["Pazartesi: 20 dk yürüyüş + nefes", "Çarşamba: Mobilite seansı", "Cuma: Hafif kardiyo", "Pazar: Esneme ve toparlanma"],
        "plan_posture":["Her gün: 5 dk boyun mobilitesi", "Her gün: 5 dk omuz mobilitesi", "Haftada 3x: Core stabilite", "Haftada 3x: Sırt güçlendirme"],
        "avg_sleep":"Ort. Uyku", "avg_study":"Ort. Çalışma", "avg_stress":"Ort. Stres", "avg_nutrition":"Ort. Beslenme", "avg_productivity":"Ort. Verimlilik", "sleep_trend":"Uyku Trendi", "stress_trend":"Stres Trendi", "nutrition_trend":"Beslenme Kalitesi Trendi", "study_trend":"Çalışma Trendi", "productivity_trend":"Verimlilik Trendi", "wellness_trend":"Sağlık Trendi", "complete_tracking_data":"Tam Takip Verisi", "training_accuracy":"Eğitim Doğruluğu",
        "metric":"Metrik", "value":"Değer", "average_sleep":"Ortalama Uyku", "average_study":"Ortalama Çalışma", "average_focus":"Ortalama Odak", "average_stress":"Ortalama Stres", "average_exercise":"Ortalama Egzersiz", "average_nutrition":"Ortalama Beslenme", "average_water":"Ortalama Su Tüketimi", "average_task":"Ortalama Görev Tamamlama", "average_productivity":"Ortalama Verimlilik", "average_wellness":"Ortalama Sağlık", "student_weekly_status_note":"Öğrenci Haftalık Durum Notu", "download_csv":"Haftalık Raporu CSV Olarak İndir", "download_pdf":"Haftalık Raporu PDF Olarak İndir", "pdf_not_active":"PDF dışa aktarma aktif değil. requirements.txt dosyasına reportlab ekleyip kur.",
        "smart_intro":"Bu bölüm sistemin manuel girişten otomatik öğrenci takibine nasıl evrilebileceğini gösterir.", "smartwatch_sleep_tracking":"Akıllı Saat Uyku Takibi", "smart_sleep_text":"Gelecek sürümler Apple Watch, Fitbit, Garmin veya uyku takip uygulamalarından gerçek uyku döngüsü verisi toplayabilir.", "deep_sleep_duration":"Derin uyku süresi", "rem_sleep":"REM uykusu", "sleep_quality_score":"Uyku kalite skoru", "recovery_score":"Toparlanma skoru", "exercise_tracking":"Egzersiz Takibi", "exercise_tracking_text":"Sistem egzersiz ve aktivite verilerini verimlilik analiziyle bağlayabilir.", "step_count":"Adım sayısı", "training_duration":"Antrenman süresi", "activity_intensity":"Aktivite yoğunluğu", "energy_score":"Enerji skoru", "weekly_reports":"Haftalık Raporlar", "weekly_reports_text":"Sistem otomatik akademik, sağlık, beslenme ve egzersiz raporları üretebilir.", "long_term_behavior_analysis":"Uzun Vadeli Davranış Analizi", "long_term_behavior_text":"Sistem uyku-verimlilik ilişkisi, stres etkisi ve çalışma istikrarı gibi örüntüleri tespit edebilir.",
        "sleep_quality":"Uyku Kalitesi", "deep_sleep":"Derin Uyku", "recovery":"Toparlanma", "exercise_minutes_label":"Egzersiz Dakikaları", "steps_1000":"Adım / 1000", "mon":"Pzt", "tue":"Sal", "wed":"Çar", "thu":"Per", "fri":"Cum", "sat":"Cmt", "sun":"Paz",
        "underweight":"Zayıf", "normal":"Normal", "overweight":"Fazla Kilolu", "high_range":"Yüksek Aralık", "bmi_under_advice":"Sağlıklı kalori, protein alımı ve güç antrenmanını artır.", "bmi_normal_advice":"Dengeli beslenme ve düzenli egzersizi koru.", "bmi_over_advice":"Kalori kontrolü, yürüyüş ve düzenli antrenmana odaklan.", "bmi_high_advice":"Yapılandırılmış plan ve profesyonel destek önerilir."
    },
    "English": {
        "date_label":"Date", "default_mood":"I feel focused and ready to study.", "ai_analysis":"AI Analysis", "mood_metric":"Mood", "status_metric":"Status", "productivity_score":"Productivity Score", "wellness_score":"Wellness Score", "risk_level":"Risk Level", "positive":"Positive", "negative":"Negative", "neutral":"Neutral", "high_risk":"High Risk", "medium_risk":"Medium Risk", "low_risk":"Low Risk", "ai_low_advice":"AI Coach: Reduce workload, drink water, eat a healthy meal, and complete one small task.", "ai_medium_advice":"AI Coach: Continue with moderate tasks, avoid multitasking, and keep a stable rhythm.", "ai_high_advice":"AI Coach: Strong productivity state. Start with difficult tasks and use deep work.", "plan_low":"25 min light study -> 5 min break -> 20 min review", "plan_medium":"45 min study -> 10 min break -> 45 min practice", "plan_high":"60 min deep work -> 10 min break -> 60 min focused study", "suggested_daily_plan":"Suggested Daily Plan", "chart_value":"Value", "chart_sleep":"Sleep", "chart_study":"Study", "chart_focus":"Focus", "chart_stress":"Stress", "chart_exercise":"Exercise / 10", "chart_tasks":"Tasks / 10", "chart_water":"Water", "chart_nutrition":"Nutrition", "chart_sentiment":"Sentiment x10", "bmi_status":"BMI Status", "goal":"Goal", "food_category":"Category", "food_examples":"Examples", "food_purpose":"Purpose", "protein":"Protein", "carbohydrate":"Carbohydrate", "healthy_fat":"Healthy Fat", "fruit":"Fruit", "vegetable":"Vegetable", "hydration":"Hydration", "protein_examples":"Eggs, chicken, fish, yogurt, lentils", "carb_examples":"Rice, oats, potatoes, whole grain bread", "fat_examples":"Olive oil, avocado, nuts, peanut butter", "fruit_examples":"Banana, apple, berries, orange", "veg_examples":"Broccoli, spinach, salad, carrots", "hydration_examples":"Water, mineral water, unsweetened tea", "protein_purpose":"Muscle repair and satiety", "carb_purpose":"Energy for studying and training", "fat_purpose":"Hormonal health and long-term energy", "fruit_purpose":"Vitamins and quick energy", "veg_purpose":"Micronutrients and digestion", "hydration_purpose":"Focus, recovery, and mood stability", "exercise_goal":"Exercise Goal", "general_health":"General Health", "weight_gain_muscle":"Weight Gain / Muscle", "fat_loss":"Fat Loss", "stress_reduction":"Stress Reduction", "posture_mobility":"Posture & Mobility", "exercise_plan_text":"This exercise plan supports physical health, focus, recovery and student productivity.", "plan_general":["Monday: 30 min walking + 10 min stretching", "Wednesday: Full body bodyweight training", "Friday: 30 min cycling or walking", "Sunday: Light mobility and recovery"], "plan_muscle":["Monday: Push training", "Tuesday: Pull training", "Thursday: Legs", "Saturday: Full body strength training"], "plan_fat":["Monday: 40 min brisk walking", "Wednesday: Full body circuit", "Friday: Interval cardio", "Sunday: Long walk"], "plan_stress":["Monday: 20 min walk + breathing", "Wednesday: Mobility session", "Friday: Light cardio", "Sunday: Stretching and recovery"], "plan_posture":["Daily: 5 min neck mobility", "Daily: 5 min shoulder mobility", "3x/week: Core stability", "3x/week: Back strengthening"], "avg_sleep":"Avg Sleep", "avg_study":"Avg Study", "avg_stress":"Avg Stress", "avg_nutrition":"Avg Nutrition", "avg_productivity":"Avg Productivity", "sleep_trend":"Sleep Trend", "stress_trend":"Stress Trend", "nutrition_trend":"Nutrition Quality Trend", "study_trend":"Study Trend", "productivity_trend":"Productivity Trend", "wellness_trend":"Wellness Trend", "complete_tracking_data":"Complete Tracking Data", "training_accuracy":"Training Accuracy", "metric":"Metric", "value":"Value", "average_sleep":"Average Sleep", "average_study":"Average Study", "average_focus":"Average Focus", "average_stress":"Average Stress", "average_exercise":"Average Exercise", "average_nutrition":"Average Nutrition", "average_water":"Average Water Intake", "average_task":"Average Task Completion", "average_productivity":"Average Productivity", "average_wellness":"Average Wellness", "student_weekly_status_note":"Student Weekly Status Note", "download_csv":"Download Weekly Report as CSV", "download_pdf":"Download Weekly Report as PDF", "pdf_not_active":"PDF export is not active. Add reportlab to requirements.txt and install it.", "smart_intro":"This section shows how the system can evolve from manual input into automated student tracking.", "smartwatch_sleep_tracking":"Smartwatch Sleep Tracking", "smart_sleep_text":"Future versions can collect real sleep cycle data from Apple Watch, Fitbit, Garmin, or sleep tracking apps.", "deep_sleep_duration":"Deep sleep duration", "rem_sleep":"REM sleep", "sleep_quality_score":"Sleep quality score", "recovery_score":"Recovery score", "exercise_tracking":"Exercise Tracking", "exercise_tracking_text":"The system can connect exercise and activity data with productivity analysis.", "step_count":"Step count", "training_duration":"Training duration", "activity_intensity":"Activity intensity", "energy_score":"Energy score", "weekly_reports":"Weekly Reports", "weekly_reports_text":"The system can automatically generate weekly academic, wellness, nutrition and exercise reports.", "long_term_behavior_analysis":"Long-Term Behavior Analysis", "long_term_behavior_text":"The system can detect patterns such as sleep-productivity relationship, stress impact, and study consistency.", "sleep_quality":"Sleep Quality", "deep_sleep":"Deep Sleep", "recovery":"Recovery", "exercise_minutes_label":"Exercise Minutes", "steps_1000":"Steps / 1000", "mon":"Mon", "tue":"Tue", "wed":"Wed", "thu":"Thu", "fri":"Fri", "sat":"Sat", "sun":"Sun", "underweight":"Underweight", "normal":"Normal", "overweight":"Overweight", "high_range":"High Range", "bmi_under_advice":"Increase healthy calories, protein intake, and strength training.", "bmi_normal_advice":"Maintain balanced nutrition and consistent exercise.", "bmi_over_advice":"Focus on calorie control, walking, and regular training.", "bmi_high_advice":"A structured plan and professional guidance are recommended."
    },
    "Deutsch": {
        "date_label":"Datum", "default_mood":"Ich fühle mich fokussiert und bereit zu lernen.", "ai_analysis":"KI-Analyse", "mood_metric":"Stimmung", "status_metric":"Status", "productivity_score":"Produktivitätswert", "wellness_score":"Wellness-Wert", "risk_level":"Risikostufe", "positive":"Positiv", "negative":"Negativ", "neutral":"Neutral", "high_risk":"Hohes Risiko", "medium_risk":"Mittleres Risiko", "low_risk":"Niedriges Risiko", "ai_low_advice":"KI-Coach: Reduziere die Arbeitslast, trinke Wasser, iss eine gesunde Mahlzeit und erledige eine kleine Aufgabe.", "ai_medium_advice":"KI-Coach: Mach mit mittleren Aufgaben weiter, vermeide Multitasking und halte einen stabilen Rhythmus.", "ai_high_advice":"KI-Coach: Starker Produktivitätszustand. Beginne mit schwierigen Aufgaben und nutze Deep Work.", "plan_low":"25 Min. leichtes Lernen -> 5 Min. Pause -> 20 Min. Wiederholung", "plan_medium":"45 Min. Lernen -> 10 Min. Pause -> 45 Min. Übung", "plan_high":"60 Min. Deep Work -> 10 Min. Pause -> 60 Min. fokussiertes Lernen", "suggested_daily_plan":"Empfohlener Tagesplan", "chart_value":"Wert", "chart_sleep":"Schlaf", "chart_study":"Lernen", "chart_focus":"Fokus", "chart_stress":"Stress", "chart_exercise":"Training / 10", "chart_tasks":"Aufgaben / 10", "chart_water":"Wasser", "chart_nutrition":"Ernährung", "chart_sentiment":"Stimmung x10", "bmi_status":"BMI-Status", "goal":"Ziel", "food_category":"Kategorie", "food_examples":"Beispiele", "food_purpose":"Zweck", "protein":"Protein", "carbohydrate":"Kohlenhydrate", "healthy_fat":"Gesunde Fette", "fruit":"Obst", "vegetable":"Gemüse", "hydration":"Hydration", "protein_examples":"Eier, Hähnchen, Fisch, Joghurt, Linsen", "carb_examples":"Reis, Hafer, Kartoffeln, Vollkornbrot", "fat_examples":"Olivenöl, Avocado, Nüsse, Erdnussbutter", "fruit_examples":"Banane, Apfel, Beeren, Orange", "veg_examples":"Brokkoli, Spinat, Salat, Karotten", "hydration_examples":"Wasser, Mineralwasser, ungesüßter Tee", "protein_purpose":"Muskelreparatur und Sättigung", "carb_purpose":"Energie fürs Lernen und Training", "fat_purpose":"Hormonelle Gesundheit und langfristige Energie", "fruit_purpose":"Vitamine und schnelle Energie", "veg_purpose":"Mikronährstoffe und Verdauung", "hydration_purpose":"Fokus, Erholung und Stimmungsstabilität", "exercise_goal":"Trainingsziel", "general_health":"Allgemeine Gesundheit", "weight_gain_muscle":"Gewichtszunahme / Muskelaufbau", "fat_loss":"Fettverlust", "stress_reduction":"Stressreduktion", "posture_mobility":"Haltung & Mobilität", "exercise_plan_text":"Dieser Trainingsplan unterstützt körperliche Gesundheit, Fokus, Erholung und studentische Produktivität.", "plan_general":["Montag: 30 Min. Gehen + 10 Min. Dehnen", "Mittwoch: Ganzkörpertraining mit Körpergewicht", "Freitag: 30 Min. Radfahren oder Gehen", "Sonntag: Leichte Mobilität und Erholung"], "plan_muscle":["Montag: Push-Training", "Dienstag: Pull-Training", "Donnerstag: Beine", "Samstag: Ganzkörper-Krafttraining"], "plan_fat":["Montag: 40 Min. zügiges Gehen", "Mittwoch: Ganzkörper-Zirkel", "Freitag: Intervall-Cardio", "Sonntag: Langer Spaziergang"], "plan_stress":["Montag: 20 Min. Gehen + Atmung", "Mittwoch: Mobilitätseinheit", "Freitag: Leichtes Cardio", "Sonntag: Dehnen und Erholung"], "plan_posture":["Täglich: 5 Min. Nackenmobilität", "Täglich: 5 Min. Schultermobilität", "3x/Woche: Core-Stabilität", "3x/Woche: Rückenkräftigung"], "avg_sleep":"Ø Schlaf", "avg_study":"Ø Lernen", "avg_stress":"Ø Stress", "avg_nutrition":"Ø Ernährung", "avg_productivity":"Ø Produktivität", "sleep_trend":"Schlaftrend", "stress_trend":"Stresstrend", "nutrition_trend":"Trend der Ernährungsqualität", "study_trend":"Lerntrend", "productivity_trend":"Produktivitätstrend", "wellness_trend":"Wellness-Trend", "complete_tracking_data":"Vollständige Tracking-Daten", "training_accuracy":"Trainingsgenauigkeit", "metric":"Metrik", "value":"Wert", "average_sleep":"Durchschnittlicher Schlaf", "average_study":"Durchschnittliches Lernen", "average_focus":"Durchschnittlicher Fokus", "average_stress":"Durchschnittlicher Stress", "average_exercise":"Durchschnittliches Training", "average_nutrition":"Durchschnittliche Ernährung", "average_water":"Durchschnittliche Wasseraufnahme", "average_task":"Durchschnittliche Aufgabenerfüllung", "average_productivity":"Durchschnittliche Produktivität", "average_wellness":"Durchschnittliches Wellness", "student_weekly_status_note":"Wöchentliche Statusnotiz des Studenten", "download_csv":"Wochenbericht als CSV herunterladen", "download_pdf":"Wochenbericht als PDF herunterladen", "pdf_not_active":"PDF-Export ist nicht aktiv. Füge reportlab zu requirements.txt hinzu und installiere es.", "smart_intro":"Dieser Abschnitt zeigt, wie sich das System von manueller Eingabe zu automatischem Studententracking entwickeln kann.", "smartwatch_sleep_tracking":"Smartwatch-Schlaftracking", "smart_sleep_text":"Zukünftige Versionen können echte Schlafzyklusdaten von Apple Watch, Fitbit, Garmin oder Schlaftracking-Apps sammeln.", "deep_sleep_duration":"Tiefschlafdauer", "rem_sleep":"REM-Schlaf", "sleep_quality_score":"Schlafqualitätswert", "recovery_score":"Erholungswert", "exercise_tracking":"Trainingstracking", "exercise_tracking_text":"Das System kann Trainings- und Aktivitätsdaten mit Produktivitätsanalyse verbinden.", "step_count":"Schrittzahl", "training_duration":"Trainingsdauer", "activity_intensity":"Aktivitätsintensität", "energy_score":"Energiewert", "weekly_reports":"Wochenberichte", "weekly_reports_text":"Das System kann automatisch akademische, Wellness-, Ernährungs- und Trainingsberichte erstellen.", "long_term_behavior_analysis":"Langfristige Verhaltensanalyse", "long_term_behavior_text":"Das System kann Muster wie Schlaf-Produktivitäts-Beziehung, Stresseffekt und Lernkonsistenz erkennen.", "sleep_quality":"Schlafqualität", "deep_sleep":"Tiefschlaf", "recovery":"Erholung", "exercise_minutes_label":"Trainingsminuten", "steps_1000":"Schritte / 1000", "mon":"Mo", "tue":"Di", "wed":"Mi", "thu":"Do", "fri":"Fr", "sat":"Sa", "sun":"So", "underweight":"Untergewicht", "normal":"Normal", "overweight":"Übergewicht", "high_range":"Hoher Bereich", "bmi_under_advice":"Erhöhe gesunde Kalorien, Proteinzufuhr und Krafttraining.", "bmi_normal_advice":"Behalte ausgewogene Ernährung und regelmäßiges Training bei.", "bmi_over_advice":"Konzentriere dich auf Kalorienkontrolle, Gehen und regelmäßiges Training.", "bmi_high_advice":"Ein strukturierter Plan und professionelle Beratung werden empfohlen."
    },
    "Русский": {},
    "Español": {}
}
# Fill remaining languages with English fallback for missing keys, then override key visible labels minimally.
EXTRA_TRANSLATIONS_ALL_TABS["Русский"].update(EXTRA_TRANSLATIONS_ALL_TABS["English"])
EXTRA_TRANSLATIONS_ALL_TABS["Русский"].update({"date_label":"Дата", "ai_analysis":"ИИ-анализ", "mood_metric":"Настроение", "status_metric":"Статус", "risk_level":"Уровень риска", "exercise_goal":"Цель упражнения", "avg_sleep":"Сред. сон", "avg_study":"Сред. учеба", "avg_stress":"Сред. стресс", "avg_nutrition":"Сред. питание", "avg_productivity":"Сред. продуктивность", "smart_intro":"Этот раздел показывает, как система может перейти от ручного ввода к автоматическому отслеживанию студента."})
EXTRA_TRANSLATIONS_ALL_TABS["Español"].update(EXTRA_TRANSLATIONS_ALL_TABS["English"])
EXTRA_TRANSLATIONS_ALL_TABS["Español"].update({"date_label":"Fecha", "ai_analysis":"Análisis IA", "mood_metric":"Estado de ánimo", "status_metric":"Estado", "risk_level":"Nivel de riesgo", "exercise_goal":"Objetivo de ejercicio", "avg_sleep":"Sueño prom.", "avg_study":"Estudio prom.", "avg_stress":"Estrés prom.", "avg_nutrition":"Nutrición prom.", "avg_productivity":"Productividad prom.", "smart_intro":"Esta sección muestra cómo el sistema puede evolucionar desde entrada manual hacia seguimiento automático del estudiante."})
for _lang, _extra in EXTRA_TRANSLATIONS_ALL_TABS.items():
    TRANSLATIONS[_lang].update(_extra)



# === V18 SAFETY LANGUAGE PATCH ===
_AVG_WELLNESS_LABELS = {
    "Türkçe": "Ort. Sağlık",
    "English": "Avg Wellness",
    "Deutsch": "Ø Wellness",
    "Русский": "Сред. здоровье",
    "Español": "Bienestar prom.",
}
for _lang, _label in _AVG_WELLNESS_LABELS.items():
    TRANSLATIONS[_lang]["avg_wellness"] = _label

# === FULL LANGUAGE PATCH V8: fix remembered inputs, exercise plans and smart integration cards ===
FULL_LANGUAGE_PATCH_V8 = {
    "Español": {
        "default_mood": "Me siento concentrado y listo para estudiar.",
        "general_health": "Salud general",
        "weight_gain_muscle": "Aumento de peso / músculo",
        "fat_loss": "Pérdida de grasa",
        "stress_reduction": "Reducción del estrés",
        "posture_mobility": "Postura y movilidad",
        "exercise_plan_text": "Este plan de ejercicio apoya la salud física, la concentración, la recuperación y la productividad del estudiante.",
        "plan_general": ["Lunes: 30 min caminata + 10 min estiramientos", "Miércoles: entrenamiento de cuerpo completo con peso corporal", "Viernes: 30 min bicicleta o caminata", "Domingo: movilidad ligera y recuperación"],
        "plan_muscle": ["Lunes: entrenamiento de empuje", "Martes: entrenamiento de tirón", "Jueves: piernas", "Sábado: entrenamiento de fuerza de cuerpo completo"],
        "plan_fat": ["Lunes: 40 min caminata rápida", "Miércoles: circuito de cuerpo completo", "Viernes: cardio por intervalos", "Domingo: caminata larga"],
        "plan_stress": ["Lunes: 20 min caminata + respiración", "Miércoles: sesión de movilidad", "Viernes: cardio ligero", "Domingo: estiramiento y recuperación"],
        "plan_posture": ["Diario: 5 min movilidad de cuello", "Diario: 5 min movilidad de hombros", "3 veces/semana: estabilidad del core", "3 veces/semana: fortalecimiento de espalda"],
        "smartwatch_sleep_tracking": "Seguimiento del sueño con smartwatch",
        "smart_sleep_text": "Las próximas versiones pueden recoger datos reales del ciclo de sueño desde Apple Watch, Fitbit, Garmin o apps de sueño.",
        "deep_sleep_duration": "Duración del sueño profundo",
        "rem_sleep": "Sueño REM",
        "sleep_quality_score": "Puntuación de calidad del sueño",
        "recovery_score": "Puntuación de recuperación",
        "exercise_tracking": "Seguimiento de ejercicio",
        "exercise_tracking_text": "El sistema puede conectar datos de ejercicio y actividad con el análisis de productividad.",
        "step_count": "Número de pasos",
        "training_duration": "Duración del entrenamiento",
        "activity_intensity": "Intensidad de actividad",
        "energy_score": "Puntuación de energía",
        "weekly_reports": "Informes semanales",
        "weekly_reports_text": "El sistema puede generar automáticamente informes académicos, de bienestar, nutrición y ejercicio.",
        "long_term_behavior_analysis": "Análisis de comportamiento a largo plazo",
        "long_term_behavior_text": "El sistema puede detectar patrones como relación sueño-productividad, efecto del estrés y constancia de estudio.",
        "sleep_quality": "Calidad del sueño", "deep_sleep": "Sueño profundo", "recovery": "Recuperación",
        "exercise_minutes_label": "Minutos de ejercicio", "steps_1000": "Pasos / 1000",
        "mon": "Lun", "tue": "Mar", "wed": "Mié", "thu": "Jue", "fri": "Vie", "sat": "Sáb", "sun": "Dom",
        "suggested_daily_plan": "Plan diario sugerido",
        "chart_sleep":"Sueño", "chart_study":"Estudio", "chart_focus":"Enfoque", "chart_stress":"Estrés", "chart_exercise":"Ejercicio / 10", "chart_tasks":"Tareas / 10", "chart_water":"Agua", "chart_nutrition":"Nutrición", "chart_sentiment":"Sentimiento x10",
    },
    "Русский": {
        "default_mood": "Я чувствую себя сосредоточенным и готовым учиться.",
        "general_health": "Общее здоровье",
        "weight_gain_muscle": "Набор веса / мышцы",
        "fat_loss": "Снижение жира",
        "stress_reduction": "Снижение стресса",
        "posture_mobility": "Осанка и мобильность",
        "exercise_plan_text": "Этот план тренировок поддерживает физическое здоровье, фокус, восстановление и продуктивность студента.",
        "plan_general": ["Понедельник: 30 мин ходьбы + 10 мин растяжки", "Среда: тренировка всего тела с собственным весом", "Пятница: 30 мин велосипед или ходьба", "Воскресенье: лёгкая мобильность и восстановление"],
        "plan_muscle": ["Понедельник: push-тренировка", "Вторник: pull-тренировка", "Четверг: ноги", "Суббота: силовая тренировка всего тела"],
        "plan_fat": ["Понедельник: 40 мин быстрой ходьбы", "Среда: круговая тренировка всего тела", "Пятница: интервальное кардио", "Воскресенье: длинная прогулка"],
        "plan_stress": ["Понедельник: 20 мин ходьбы + дыхание", "Среда: сессия мобильности", "Пятница: лёгкое кардио", "Воскресенье: растяжка и восстановление"],
        "plan_posture": ["Ежедневно: 5 мин мобильности шеи", "Ежедневно: 5 мин мобильности плеч", "3 раза/неделю: стабильность кора", "3 раза/неделю: укрепление спины"],
        "smartwatch_sleep_tracking": "Отслеживание сна через smartwatch",
        "smart_sleep_text": "Будущие версии смогут собирать реальные данные циклов сна с Apple Watch, Fitbit, Garmin или приложений сна.",
        "deep_sleep_duration": "Длительность глубокого сна", "rem_sleep": "REM-сон", "sleep_quality_score": "Оценка качества сна", "recovery_score": "Оценка восстановления",
        "exercise_tracking": "Отслеживание тренировок", "exercise_tracking_text": "Система может связывать данные тренировок и активности с анализом продуктивности.",
        "step_count": "Количество шагов", "training_duration": "Длительность тренировки", "activity_intensity": "Интенсивность активности", "energy_score": "Оценка энергии",
        "weekly_reports": "Еженедельные отчёты", "weekly_reports_text": "Система может автоматически создавать учебные, wellness, nutrition и exercise отчёты.",
        "long_term_behavior_analysis": "Долгосрочный анализ поведения", "long_term_behavior_text": "Система может обнаруживать паттерны: связь сна и продуктивности, влияние стресса и стабильность учёбы.",
        "sleep_quality": "Качество сна", "deep_sleep": "Глубокий сон", "recovery": "Восстановление", "exercise_minutes_label": "Минуты тренировки", "steps_1000": "Шаги / 1000",
        "mon": "Пн", "tue": "Вт", "wed": "Ср", "thu": "Чт", "fri": "Пт", "sat": "Сб", "sun": "Вс",
        "suggested_daily_plan": "Рекомендуемый план дня",
    },
}
for _lang, _patch in FULL_LANGUAGE_PATCH_V8.items():
    TRANSLATIONS[_lang].update(_patch)

# === V9 PRODUCT PATCH: daily rotating motivation, missions, streaks, cleaner sidebar ===
V9_PRODUCT_PATCH = {
    "Türkçe": {
        "welcome_title": "👋 Hoş geldin, {name}", "welcome_text": "Bugünün odak hedefi: küçük ama net ilerleme.", "ai_mission_title": "🤖 Bugünün AI Görevi",
        "mission_water": "2 litre su iç", "mission_study": "1 saat odaklı çalışma yap", "mission_walk": "20 dakika yürü veya esne", "mission_sleep": "Gece rutini kur ve erken uyu",
        "study_streak": "Çalışma Serisi", "water_streak": "Su Serisi", "exercise_streak": "Egzersiz Serisi", "day_unit": "gün",
        "productivity_kpi": "Verimlilik", "wellness_kpi": "Sağlık", "sleep_kpi": "Uyku", "stress_kpi": "Stres",
        "motivation_quotes": ["Bugün vazgeçmezsen yarın kendine teşekkür edeceksin.", "Disiplin, motivasyonun bittiği yerde oyuna girer.", "Küçük adımlar büyük hayatları sessizce inşa eder.", "Bugünkü emek, yarının özgürlüğüdür.", "Odaklan, sadeleş, ilerle.", "Bir saatlik gerçek çalışma, bir gün pişmanlıktan iyidir.", "Bugün sistem kur; yarın sistem seni taşır."],
    },
    "English": {
        "welcome_title": "👋 Welcome, {name}", "welcome_text": "Today's focus target: small but clear progress.", "ai_mission_title": "🤖 Today's AI Mission",
        "mission_water": "Drink 2 liters of water", "mission_study": "Complete 1 focused study hour", "mission_walk": "Walk or stretch for 20 minutes", "mission_sleep": "Build a night routine and sleep early",
        "study_streak": "Study Streak", "water_streak": "Water Streak", "exercise_streak": "Exercise Streak", "day_unit": "days",
        "productivity_kpi": "Productivity", "wellness_kpi": "Wellness", "sleep_kpi": "Sleep", "stress_kpi": "Stress",
        "motivation_quotes": ["Small steps create serious momentum.", "Discipline beats motivation when the day gets heavy.", "Your future self is built by today's routine.", "Do the next right thing, then repeat.", "Focus is a skill. Train it daily.", "One honest hour beats a full day of excuses.", "Build the system; the system will carry you."],
    },
    "Deutsch": {
        "welcome_title": "👋 Willkommen, {name}", "welcome_text": "Heutiges Fokusziel: kleiner, aber klarer Fortschritt.", "ai_mission_title": "🤖 Heutige KI-Mission",
        "mission_water": "2 Liter Wasser trinken", "mission_study": "1 Stunde fokussiert lernen", "mission_walk": "20 Minuten gehen oder dehnen", "mission_sleep": "Abendroutine aufbauen und früher schlafen",
        "study_streak": "Lernserie", "water_streak": "Wasserserie", "exercise_streak": "Trainingsserie", "day_unit": "Tage",
        "productivity_kpi": "Produktivität", "wellness_kpi": "Wellness", "sleep_kpi": "Schlaf", "stress_kpi": "Stress",
        "motivation_quotes": ["Kleine Schritte erzeugen starke Dynamik.", "Disziplin übernimmt, wenn Motivation nachlässt.", "Dein zukünftiges Ich entsteht durch deine heutige Routine.", "Mach den nächsten richtigen Schritt, dann wiederhole ihn.", "Fokus ist eine Fähigkeit. Trainiere sie täglich.", "Eine ehrliche Stunde schlägt einen ganzen Tag voller Ausreden.", "Baue das System; das System trägt dich."],
    },
    "Русский": {
        "welcome_title": "👋 Добро пожаловать, {name}", "welcome_text": "Фокус дня: небольшой, но ясный прогресс.", "ai_mission_title": "🤖 Миссия ИИ на сегодня",
        "mission_water": "Выпить 2 литра воды", "mission_study": "Сделать 1 час сфокусированной учёбы", "mission_walk": "20 минут ходьбы или растяжки", "mission_sleep": "Создать вечернюю рутину и лечь раньше",
        "study_streak": "Серия учёбы", "water_streak": "Серия воды", "exercise_streak": "Серия тренировок", "day_unit": "дн.",
        "productivity_kpi": "Продуктивность", "wellness_kpi": "Здоровье", "sleep_kpi": "Сон", "stress_kpi": "Стресс",
        "motivation_quotes": ["Маленькие шаги создают серьёзную инерцию.", "Дисциплина работает там, где мотивация устала.", "Твоё будущее строится сегодняшней рутиной.", "Сделай следующий правильный шаг и повтори.", "Фокус — это навык. Тренируй его каждый день.", "Один честный час лучше дня оправданий.", "Построй систему; система будет нести тебя."],
    },
    "Español": {
        "welcome_title": "👋 Bienvenido, {name}", "welcome_text": "Objetivo de enfoque de hoy: progreso pequeño pero claro.", "ai_mission_title": "🤖 Misión IA de hoy",
        "mission_water": "Beber 2 litros de agua", "mission_study": "Completar 1 hora de estudio enfocado", "mission_walk": "Caminar o estirar 20 minutos", "mission_sleep": "Crear una rutina nocturna y dormir temprano",
        "study_streak": "Racha de estudio", "water_streak": "Racha de agua", "exercise_streak": "Racha de ejercicio", "day_unit": "días",
        "productivity_kpi": "Productividad", "wellness_kpi": "Bienestar", "sleep_kpi": "Sueño", "stress_kpi": "Estrés",
        "motivation_quotes": ["Los pequeños pasos crean una gran inercia.", "La disciplina gana cuando la motivación se cansa.", "Tu futuro se construye con la rutina de hoy.", "Haz el siguiente paso correcto y repítelo.", "El enfoque es una habilidad. Entrénalo a diario.", "Una hora honesta vale más que un día de excusas.", "Construye el sistema; el sistema te sostendrá."],
    },
}
for _lang, _patch in V9_PRODUCT_PATCH.items():
    TRANSLATIONS[_lang].update(_patch)

# === V16 PATCH: richer welcome card text in all languages ===
V16_WELCOME_PATCH = {
    "Türkçe": {
        "welcome_focus_label": "Bugünkü öncelik",
        "welcome_focus_value": "Derin çalışma + enerji yönetimi",
        "welcome_next_step": "Sıradaki akıllı adım",
        "welcome_next_step_value": "Önce 25 dakikalık tek görev seç, sonra kısa mola ver.",
        "welcome_profile_hint": "Profiline göre kişisel öneriler güncellenir.",
        "quick_energy": "Enerji",
        "quick_routine": "Rutin",
        "quick_risk": "Risk",
        "energy_good": "Dengeli",
        "energy_watch": "Dikkat",
        "routine_good": "Stabil",
        "routine_build": "Kuruluyor",
        "risk_low": "Düşük",
        "risk_medium": "Orta",
        "risk_high": "Yüksek",
    },
    "English": {
        "welcome_focus_label": "Today’s priority",
        "welcome_focus_value": "Deep work + energy management",
        "welcome_next_step": "Next smart step",
        "welcome_next_step_value": "Pick one 25-minute task first, then take a short break.",
        "welcome_profile_hint": "Personal recommendations update based on your profile.",
        "quick_energy": "Energy",
        "quick_routine": "Routine",
        "quick_risk": "Risk",
        "energy_good": "Balanced",
        "energy_watch": "Watch",
        "routine_good": "Stable",
        "routine_build": "Building",
        "risk_low": "Low",
        "risk_medium": "Medium",
        "risk_high": "High",
    },
    "Deutsch": {
        "welcome_focus_label": "Heutige Priorität",
        "welcome_focus_value": "Deep Work + Energiemanagement",
        "welcome_next_step": "Nächster smarter Schritt",
        "welcome_next_step_value": "Wähle zuerst eine 25-Minuten-Aufgabe, dann kurze Pause.",
        "welcome_profile_hint": "Persönliche Empfehlungen werden anhand deines Profils aktualisiert.",
        "quick_energy": "Energie",
        "quick_routine": "Routine",
        "quick_risk": "Risiko",
        "energy_good": "Ausgeglichen",
        "energy_watch": "Achten",
        "routine_good": "Stabil",
        "routine_build": "Im Aufbau",
        "risk_low": "Niedrig",
        "risk_medium": "Mittel",
        "risk_high": "Hoch",
    },
    "Русский": {
        "welcome_focus_label": "Приоритет дня",
        "welcome_focus_value": "Глубокая работа + управление энергией",
        "welcome_next_step": "Следующий умный шаг",
        "welcome_next_step_value": "Сначала выбери задачу на 25 минут, затем короткий перерыв.",
        "welcome_profile_hint": "Личные рекомендации обновляются на основе профиля.",
        "quick_energy": "Энергия",
        "quick_routine": "Рутина",
        "quick_risk": "Риск",
        "energy_good": "Баланс",
        "energy_watch": "Внимание",
        "routine_good": "Стабильно",
        "routine_build": "Формируется",
        "risk_low": "Низкий",
        "risk_medium": "Средний",
        "risk_high": "Высокий",
    },
    "Español": {
        "welcome_focus_label": "Prioridad de hoy",
        "welcome_focus_value": "Trabajo profundo + gestión de energía",
        "welcome_next_step": "Siguiente paso inteligente",
        "welcome_next_step_value": "Elige primero una tarea de 25 minutos y luego toma una pausa corta.",
        "welcome_profile_hint": "Las recomendaciones personales se actualizan según tu perfil.",
        "quick_energy": "Energía",
        "quick_routine": "Rutina",
        "quick_risk": "Riesgo",
        "energy_good": "Equilibrada",
        "energy_watch": "Atención",
        "routine_good": "Estable",
        "routine_build": "En progreso",
        "risk_low": "Bajo",
        "risk_medium": "Medio",
        "risk_high": "Alto",
    },
}
for _lang, _patch in V16_WELCOME_PATCH.items():
    TRANSLATIONS[_lang].update(_patch)

# Small UI labels used in profile/PDF/model areas
for _lang in TRANSLATIONS:
    TRANSLATIONS[_lang].setdefault("faculty_software", "Software Engineering")
    TRANSLATIONS[_lang].setdefault("goal_improve_productivity", "Improve Productivity")
    TRANSLATIONS[_lang].setdefault("goal_gain_weight", "Gain Weight")
    TRANSLATIONS[_lang].setdefault("goal_lose_fat", "Lose Fat")
    TRANSLATIONS[_lang].setdefault("goal_reduce_stress", "Reduce Stress")
    TRANSLATIONS[_lang].setdefault("goal_improve_sleep", "Improve Sleep")
    TRANSLATIONS[_lang].setdefault("goal_build_discipline", "Build Discipline")
    TRANSLATIONS[_lang].setdefault("importance", "Importance")
    TRANSLATIONS[_lang].setdefault("pdf_title", "AI Student Weekly Performance Report")
    TRANSLATIONS[_lang].setdefault("student_label", "Student")
    TRANSLATIONS[_lang].setdefault("student_status_summary", "Student Status Summary")
    TRANSLATIONS[_lang].setdefault("weekly_metrics", "Weekly Metrics")
    TRANSLATIONS[_lang].setdefault("ai_coach_recommendations", "AI Coach Recommendations")
TRANSLATIONS["Türkçe"].update({"faculty_software":"Yazılım Mühendisliği", "goal_improve_productivity":"Verimliliği Artır", "goal_gain_weight":"Kilo Al", "goal_lose_fat":"Yağ Kaybet", "goal_reduce_stress":"Stresi Azalt", "goal_improve_sleep":"Uykuyu İyileştir", "goal_build_discipline":"Disiplin Kur", "importance":"Önem", "pdf_title":"AI Öğrenci Haftalık Performans Raporu", "student_label":"Öğrenci", "student_status_summary":"Öğrenci Durum Özeti", "weekly_metrics":"Haftalık Metrikler", "ai_coach_recommendations":"AI Koç Önerileri"})
TRANSLATIONS["Deutsch"].update({"faculty_software":"Softwaretechnik", "goal_improve_productivity":"Produktivität steigern", "goal_gain_weight":"Gewicht zunehmen", "goal_lose_fat":"Fett verlieren", "goal_reduce_stress":"Stress reduzieren", "goal_improve_sleep":"Schlaf verbessern", "goal_build_discipline":"Disziplin aufbauen", "importance":"Wichtigkeit", "pdf_title":"KI-Wochenbericht zur Studentenleistung", "student_label":"Student", "student_status_summary":"Statuszusammenfassung", "weekly_metrics":"Wöchentliche Metriken", "ai_coach_recommendations":"KI-Coach-Empfehlungen"})
TRANSLATIONS["Русский"].update({"faculty_software":"Программная инженерия", "goal_improve_productivity":"Повысить продуктивность", "goal_gain_weight":"Набрать вес", "goal_lose_fat":"Снизить жир", "goal_reduce_stress":"Снизить стресс", "goal_improve_sleep":"Улучшить сон", "goal_build_discipline":"Развить дисциплину", "importance":"Важность", "pdf_title":"Еженедельный отчет ИИ о результатах студента", "student_label":"Студент", "student_status_summary":"Сводка статуса студента", "weekly_metrics":"Недельные метрики", "ai_coach_recommendations":"Рекомендации ИИ-коуча"})
TRANSLATIONS["Español"].update({"faculty_software":"Ingeniería de software", "goal_improve_productivity":"Aumentar productividad", "goal_gain_weight":"Ganar peso", "goal_lose_fat":"Perder grasa", "goal_reduce_stress":"Reducir estrés", "goal_improve_sleep":"Mejorar sueño", "goal_build_discipline":"Crear disciplina", "importance":"Importancia", "pdf_title":"Informe semanal de rendimiento estudiantil con IA", "student_label":"Estudiante", "student_status_summary":"Resumen del estado del estudiante", "weekly_metrics":"Métricas semanales", "ai_coach_recommendations":"Recomendaciones del coach IA"})


THEMES = {
    "Neon Blue": {
        "primary": "#38bdf8",
        "secondary": "#2563eb",
        "accent": "#22c55e",
        "warning": "#f97316",
        "danger": "#ef4444",
        "soft": "#1e3a8a",
    },
    "Cyber Purple": {
        "primary": "#a855f7",
        "secondary": "#7c3aed",
        "accent": "#ec4899",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "soft": "#581c87",
    },
    "Emerald Health": {
        "primary": "#10b981",
        "secondary": "#059669",
        "accent": "#84cc16",
        "warning": "#f97316",
        "danger": "#dc2626",
        "soft": "#064e3b",
    },
    "Sunset Orange": {
        "primary": "#fb923c",
        "secondary": "#ea580c",
        "accent": "#facc15",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "soft": "#9a3412",
    },
}



# === V18 DAILY FORM COPY PATCH ===
_DAILY_FORM_COPY = {
    "Türkçe": {
        "daily_form_intro": "Önce günün tüm verilerini gir. Analiz ve grafikler sadece butona bastığında oluşur; sayfa veri girerken zıplamaz.",
        "daily_input_panel": "Günlük Veri Giriş Paneli",
        "daily_mode": "Günün Modu",
        "mode_tired": "😴 Yorgun",
        "mode_normal": "🙂 Normal",
        "mode_motivated": "🔥 Motive",
        "mode_deep_work": "🚀 Derin Çalışma",
        "daily_waiting_message": "Tüm verileri girip analiz butonuna basınca bugünün AI raporu burada oluşacak.",
    },
    "English": {
        "daily_form_intro": "Enter all daily data first. Analysis and charts are generated only after you press the button; the page will not jump while you edit values.",
        "daily_input_panel": "Daily Data Input Panel",
        "daily_mode": "Daily Mode",
        "mode_tired": "😴 Tired",
        "mode_normal": "🙂 Normal",
        "mode_motivated": "🔥 Motivated",
        "mode_deep_work": "🚀 Deep Work",
        "daily_waiting_message": "Fill all values and press the analysis button to generate today's AI report here.",
    },
    "Deutsch": {
        "daily_form_intro": "Gib zuerst alle Tagesdaten ein. Analyse und Diagramme entstehen erst nach dem Button-Klick; die Seite springt nicht während der Eingabe.",
        "daily_input_panel": "Tägliches Eingabepanel",
        "daily_mode": "Tagesmodus",
        "mode_tired": "😴 Müde",
        "mode_normal": "🙂 Normal",
        "mode_motivated": "🔥 Motiviert",
        "mode_deep_work": "🚀 Deep Work",
        "daily_waiting_message": "Gib alle Werte ein und klicke auf Analyse, dann erscheint hier der KI-Bericht des Tages.",
    },
    "Русский": {
        "daily_form_intro": "Сначала введи все дневные данные. Анализ и графики появятся только после нажатия кнопки; страница не будет прыгать при вводе.",
        "daily_input_panel": "Панель дневного ввода",
        "daily_mode": "Режим дня",
        "mode_tired": "😴 Устал",
        "mode_normal": "🙂 Нормально",
        "mode_motivated": "🔥 Мотивирован",
        "mode_deep_work": "🚀 Глубокая работа",
        "daily_waiting_message": "Заполни все значения и нажми кнопку анализа, чтобы создать сегодняшний AI-отчет.",
    },
    "Español": {
        "daily_form_intro": "Primero introduce todos los datos del día. El análisis y los gráficos se generan solo al pulsar el botón; la página no saltará mientras editas.",
        "daily_input_panel": "Panel de datos diarios",
        "daily_mode": "Modo del día",
        "mode_tired": "😴 Cansado",
        "mode_normal": "🙂 Normal",
        "mode_motivated": "🔥 Motivado",
        "mode_deep_work": "🚀 Trabajo profundo",
        "daily_waiting_message": "Completa todos los valores y pulsa el botón de análisis para generar aquí el informe IA de hoy.",
    },
}
for _lang, _extra in _DAILY_FORM_COPY.items():
    TRANSLATIONS[_lang].update(_extra)

if "language" not in st.session_state:
    st.session_state.language = "Türkçe"
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Sunset Orange"
if "background_mode" not in st.session_state:
    st.session_state.background_mode = "Yumuşak Çoklu Renk"
if "sidebar_page" not in st.session_state:
    st.session_state.sidebar_page = "home"


def get_text():
    return TRANSLATIONS[st.session_state.language]


t = get_text()


with st.sidebar:
    st.session_state.theme_name = st.selectbox(
        t["theme"],
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name),
        key="theme_select",
    )

    st.divider()
    st.markdown(
        f"""
        <div class="side-menu-title">{t["sidebar_title"]}</div>
        """,
        unsafe_allow_html=True,
    )

    home_label = {
        "Türkçe": "🏠 Ana Sayfa",
        "English": "🏠 Home",
        "Deutsch": "🏠 Startseite",
        "Русский": "🏠 Главная",
        "Español": "🏠 Inicio",
    }.get(st.session_state.language, "🏠 Home")

    page_labels = {
        "home": home_label,
        "login": t["login"],
        "settings": t["settings"],
        "database": t["db"],
    }

    page_order = ["home", "login", "settings", "database"]
    if st.session_state.sidebar_page not in page_order:
        st.session_state.sidebar_page = "home"

    selected_label = st.radio(
        "Navigation",
        [page_labels[key] for key in page_order],
        index=page_order.index(st.session_state.sidebar_page),
        label_visibility="collapsed",
        key="sidebar_radio_visible",
    )

    reverse_page_labels = {v: k for k, v in page_labels.items()}
    st.session_state.sidebar_page = reverse_page_labels[selected_label]

    st.divider()

    selected_language = st.selectbox(
        t["language"],
        ["Türkçe", "English", "Deutsch", "Русский", "Español"],
        index=["Türkçe", "English", "Deutsch", "Русский", "Español"].index(st.session_state.language),
        key="language_select",
    )

    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        # Reset language-sensitive input defaults so example texts switch language too.
        st.session_state.pop("daily_mood_text", None)
        st.rerun()

    t = get_text()

    st.markdown(
        f"""
        <div class="side-note">
            <div style="font-size:28px;">❝</div>
            <b>{t["motivation"]}</b><br><br>
            — {t["you_can"]} ✨
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(t["logout"], key="logout_button"):
        st.session_state.profile = {
            "name": "Student",
            "student_id": "0000",
            "faculty": "Yazılım Mühendisliği",
            "semester": 2,
            "age": 22,
            "height_cm": 175,
            "weight_kg": 70,
            "goal": "Verimliliği Artır",
        }
        st.success("Session cleared.")


theme = THEMES[st.session_state.theme_name]


def background_css(mode):
    if mode in ["Açık / Beyaz", "Light / White", "Hell / Weiß", "Светлый / Белый", "Claro / Blanco"]:
        return """
        background:
            radial-gradient(circle at top left, rgba(251,146,60,0.18), transparent 30%),
            radial-gradient(circle at bottom right, rgba(16,185,129,0.14), transparent 30%),
            #f8fafc;
        color: #111827;
        """
    if mode in ["Yeşil Sağlık", "Green Health", "Grüne Gesundheit", "Зеленое здоровье", "Salud verde"]:
        return """
        background:
            radial-gradient(circle at top left, rgba(16,185,129,0.28), transparent 32%),
            radial-gradient(circle at bottom right, rgba(132,204,22,0.20), transparent 30%),
            #061a14;
        color: #f8fafc;
        """
    if mode in ["Turuncu Enerji", "Orange Energy", "Orange Energie", "Оранжевая энергия", "Energía naranja"]:
        return """
        background:
            radial-gradient(circle at top left, rgba(251,146,60,0.30), transparent 30%),
            radial-gradient(circle at bottom right, rgba(250,204,21,0.20), transparent 32%),
            #1b1007;
        color: #f8fafc;
        """
    if mode in ["Koyu", "Dark", "Dunkel", "Темный", "Oscuro"]:
        return """
        background: #0b0f17;
        color: #f8fafc;
        """
    return """
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.20), transparent 30%),
        radial-gradient(circle at 80% 10%, rgba(251,146,60,0.18), transparent 30%),
        radial-gradient(circle at bottom left, rgba(16,185,129,0.16), transparent 32%),
        #0b0f17;
    color: #f8fafc;
    """


def apply_css(theme_dict, bg_mode):
    css = """
<style>
.stApp {{
    {background_style}
}}
.block-container {{
    padding-top: 0.8rem;
    max-width: 1500px;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 2.2rem !important;
}}
[data-testid="stToolbar"] {{
    right: 1rem !important;
    top: 0.2rem !important;
}}
.main-title {{
    font-size: 46px;
    font-weight: 950;
    margin-bottom: 10px;
    letter-spacing: -1.2px;
}}
.sub-title {{
    font-size: 18px;
    color: #cbd5e1;
    line-height: 1.6;
    margin-bottom: 28px;
}}
.hero-card {{
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, {secondary}, {primary});
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 18px 42px rgba(0,0,0,0.34);
    border: 1px solid rgba(255,255,255,0.12);
}}
.info-card {{
    padding: 22px;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.20);
    color: white;
    margin-bottom: 15px;
}}
.status-card {{
    padding: 24px;
    border-radius: 20px;
    background: linear-gradient(135deg, {soft}, #111827);
    border: 1px solid {primary};
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 12px 34px rgba(0,0,0,0.28);
}}
.quote-card {{
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, {secondary});
    color: white;
    margin-bottom: 18px;
    border-left: 6px solid {accent};
}}
.card-blue, .card-green, .card-purple, .card-orange {{
    padding: 22px;
    border-radius: 18px;
    color: white;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.12);
}}
.card-blue {{ background: linear-gradient(135deg, #1e3a8a, {primary}); }}
.card-green {{ background: linear-gradient(135deg, #064e3b, {accent}); }}
.card-purple {{ background: linear-gradient(135deg, #581c87, #a855f7); }}
.card-orange {{ background: linear-gradient(135deg, #9a3412, {warning}); }}

.side-menu-title {{
    font-size: 24px;
    font-weight: 900;
    margin: 12px 0 18px 0;
    color: #f8fafc;
}}
.language-card {{
    margin-top: 14px;
    padding: 16px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(15,23,42,0.72), rgba(30,41,59,0.72));
    border: 1px solid rgba(148,163,184,0.20);
    color: #f8fafc;
    font-size: 15px;
}}
.side-note {{
    padding: 16px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(30,41,59,0.76), rgba(49,46,129,0.58));
    border: 1px solid rgba(148, 163, 184, 0.18);
    color: #e5e7eb;
    line-height: 1.55;
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
}}

div[role="radiogroup"] {{
    display: flex;
    flex-direction: column;
    gap: 12px;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] {{
    width: 100% !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
    width: 100% !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] {{
    width: 100% !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    width: 100% !important;
}}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] {{
    width: 100% !important;
}}

div[role="radiogroup"] label {{
    width: 100% !important;
    min-width: 100% !important;
    min-height: 58px !important;
    display: flex !important;
    align-items: center !important;
    background: rgba(15, 23, 42, 0.50);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    padding: 14px 18px !important;
    margin: 0 !important;
    box-shadow: 0 8px 18px rgba(0,0,0,0.18);
    box-sizing: border-box !important;
}}
div[role="radiogroup"] label:hover {{
    background: linear-gradient(135deg, rgba(124,58,237,0.28), rgba(59,130,246,0.22));
    border: 1px solid rgba(255,255,255,0.20);
}}
div[role="radiogroup"] label p {{
    font-weight: 800 !important;
    font-size: 15px !important;
    margin: 0 !important;
}}

div.stButton > button {{
    background: linear-gradient(135deg, {secondary}, {primary});
    color: white;
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 0.85rem 1.45rem;
    font-weight: 900;
    box-shadow: 0 10px 22px rgba(0,0,0,0.28);
    transition: all 0.22s ease-in-out;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    filter: brightness(1.08);
    box-shadow: 0 0 16px rgba(148,163,184,0.24), 0 12px 25px rgba(0,0,0,0.32);
    color: white;
}}
.stDownloadButton > button {{
    background: linear-gradient(135deg, {secondary}, {primary});
    color: white;
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 0.85rem 1.45rem;
    font-weight: 900;
}}

div[data-testid="stTabs"] div[role="tablist"] {{
    gap: 14px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    padding-bottom: 16px;
    flex-wrap: wrap;
}}
button[data-baseweb="tab"],
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {{
    min-height: 48px !important;
    min-width: 132px !important;
    max-width: 220px !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
    margin: 0 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #f8fafc !important;
    font-weight: 800 !important;
    letter-spacing: -0.2px !important;
    overflow: hidden !important;
    position: relative !important;
    transition: all 0.22s ease-in-out !important;
    box-shadow: 0 10px 22px rgba(0,0,0,0.20) !important;
}}
button[data-baseweb="tab"]:nth-child(1), div[data-testid="stTabs"] [role="tab"]:nth-child(1) {{
    background: linear-gradient(135deg, rgba(37,99,235,0.62), rgba(14,165,233,0.52)) !important;
}}
button[data-baseweb="tab"]:nth-child(2), div[data-testid="stTabs"] [role="tab"]:nth-child(2) {{
    background: linear-gradient(135deg, rgba(22,163,74,0.62), rgba(20,184,166,0.50)) !important;
}}
button[data-baseweb="tab"]:nth-child(3), div[data-testid="stTabs"] [role="tab"]:nth-child(3) {{
    background: linear-gradient(135deg, rgba(13,148,136,0.60), rgba(6,182,212,0.46)) !important;
}}
button[data-baseweb="tab"]:nth-child(4), div[data-testid="stTabs"] [role="tab"]:nth-child(4) {{
    background: linear-gradient(135deg, rgba(99,102,241,0.62), rgba(168,85,247,0.48)) !important;
}}
button[data-baseweb="tab"]:nth-child(5), div[data-testid="stTabs"] [role="tab"]:nth-child(5) {{
    background: linear-gradient(135deg, rgba(202,138,4,0.62), rgba(234,179,8,0.48)) !important;
}}
button[data-baseweb="tab"]:nth-child(6), div[data-testid="stTabs"] [role="tab"]:nth-child(6) {{
    background: linear-gradient(135deg, rgba(190,24,93,0.58), rgba(244,114,182,0.44)) !important;
}}
button[data-baseweb="tab"]:nth-child(7), div[data-testid="stTabs"] [role="tab"]:nth-child(7) {{
    background: linear-gradient(135deg, rgba(14,116,144,0.60), rgba(56,189,248,0.44)) !important;
}}
button[data-baseweb="tab"]:hover,
div[data-testid="stTabs"] [role="tab"]:hover {{
    transform: translateY(-2px) !important;
    filter: brightness(1.10) saturate(1.05) !important;
    box-shadow: 0 0 16px rgba(148,163,184,0.22), 0 12px 25px rgba(0,0,0,0.30) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
}}
button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    transform: translateY(-2px) !important;
    border: 1px solid rgba(255,255,255,0.32) !important;
    box-shadow: 0 0 18px rgba(139,92,246,0.32), 0 14px 28px rgba(0,0,0,0.35) !important;
}}
button[data-baseweb="tab"],
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] div,
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] [role="tab"] div {{
    color: #ffffff !important;
    font-weight:600 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 13px !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
}}
div[data-baseweb="tab-highlight"],
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{
    background: transparent !important;
}}




/* === FINAL COMPACT NO-OVERFLOW TAB PATCH === */

/* Remove duplicate middle line and keep tabs close to project card */
div[data-testid="stTabs"] div[role="tablist"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    gap: 8px !important;
    flex-wrap: wrap !important;

    overflow-x: hidden !important;
    overflow-y: hidden !important;

    padding: 14px 0 12px 0 !important;
    margin-top: 12px !important;

    border-top: 1px solid rgba(148, 163, 184, 0.22) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.22) !important;
}}

/* Hide any tab scrollbar */
div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar {{
    height: 0 !important;
}}

/* Compact equal tabs: no right overflow */
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {{
    width: 165px !important;
    min-width: 165px !important;
    max-width: 165px !important;

    height: 50px !important;
    min-height: 50px !important;
    max-height: 50px !important;

    padding: 0 12px !important;
    margin: 0 !important;

    border-radius: 16px !important;
    box-sizing: border-box !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    overflow: hidden !important;
    line-height: 1 !important;
}}

/* Emoji + text vertical alignment */
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] [role="tab"] p {{
    margin: 0 !important;
    padding: 0 !important;

    line-height: 1 !important;
    height: auto !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    font-size:15px !important;
    font-weight: 900 !important;

    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}

/* Prevent wrappers from pushing emojis down */
div[data-testid="stTabs"] button[role="tab"] div,
div[data-testid="stTabs"] [role="tab"] div,
div[data-testid="stTabs"] button[role="tab"] span,
div[data-testid="stTabs"] [role="tab"] span {{
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

/* Remove Streamlit underline/highlight */
div[data-baseweb="tab-highlight"],
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{
    background: transparent !important;
    height: 0 !important;
}}

/* Sidebar buttons: same width as language select and active language card */
section[data-testid="stSidebar"] div[data-testid="stRadio"] {{
    width: 100% !important;
    max-width: 100% !important;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
    width: 100% !important;
    max-width: 100% !important;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] {{
    width: 100% !important;
    max-width: 100% !important;

    display: flex !important;
    flex-direction: column !important;
    gap: 14px !important;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;

    height: 60px !important;
    min-height: 60px !important;

    display: flex !important;
    align-items: center !important;

    padding: 0 18px !important;
    margin: 0 !important;

    border-radius: 18px !important;
    box-sizing: border-box !important;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    margin: 0 !important;
    padding: 0 !important;

    line-height: 1 !important;
    font-size: 15px !important;
    font-weight: 900 !important;
}}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"],
section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] .language-card,
section[data-testid="stSidebar"] .side-note {{
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    min-height: 58px !important;
    border-radius: 16px !important;
}}

section[data-testid="stSidebar"] .language-card,
section[data-testid="stSidebar"] .side-note {{
    border-radius: 18px !important;
}}


/* === LOGIN PAGE LAYOUT + CLEAN TAB POSITION PATCH === */
div[data-testid="stTabs"] {{
    margin-top: 10px !important;
    margin-bottom: 24px !important;
}}
div[data-testid="stTabs"] div[role="tablist"] {{
    margin-top: 0 !important;
}}
.login-profile-wrap {{
    margin-top: -6px;
}}



/* === V18 PREMIUM INPUT POLISH === */
div[data-testid="stForm"] {
    padding: 26px !important;
    border-radius: 24px !important;
    background: linear-gradient(135deg, rgba(15,23,42,.78), rgba(49,46,129,.25), rgba(251,146,60,.10)) !important;
    border: 1px solid rgba(148,163,184,.22) !important;
    box-shadow: 0 18px 46px rgba(0,0,0,.24) !important;
}
div[data-testid="stForm"] label p {
    font-weight: 750 !important;
    letter-spacing: -0.2px !important;
}
div[data-testid="stMetric"] {
    padding: 14px 16px !important;
    border-radius: 18px !important;
    background: rgba(15,23,42,.25) !important;
    border: 1px solid rgba(148,163,184,.10) !important;
}
div[data-testid="stForm"] textarea,
div[data-testid="stForm"] input {
    border-radius: 14px !important;
}

</style>
"""
    css = css.format(background_style=background_css(bg_mode), **theme_dict)
    st.markdown(css, unsafe_allow_html=True)


apply_css(theme, st.session_state.background_mode)

st.markdown("""
<style>

/* === V12 CLEAN TAB POSITION + ELEGANT LABELS === */
div[data-testid="stTabs"] {
    margin-top: 24px !important;
    margin-bottom: 24px !important;
}
div[data-testid="stTabs"] div[role="tablist"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
    padding: 14px 0 14px 0 !important;
    margin: 24px 0 26px 0 !important;
    border-top: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.24) !important;
    overflow: visible !important;
}
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {
    width: 150px !important;
    min-width: 150px !important;
    max-width: 150px !important;
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    padding: 0 12px !important;
    border-radius: 15px !important;
    opacity: 0.96 !important;
}
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] div,
div[data-testid="stTabs"] [role="tab"] div,
div[data-testid="stTabs"] button[role="tab"] span,
div[data-testid="stTabs"] [role="tab"] span {
    font-size: 14px !important;
    font-weight: 720 !important;
    letter-spacing: -0.15px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover,
div[data-testid="stTabs"] [role="tab"]:hover {
    transform: translateY(-1px) !important;
}
.hero-card {
    margin-bottom: 0 !important;
}

/* === V9 SIDEBAR POLISH + PRODUCT CARDS === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(31,35,48,0.98), rgba(24,28,39,0.98)) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    font-weight: 650 !important;
}
.side-menu-title {
    font-size: 27px !important;
    font-weight: 780 !important;
    letter-spacing: -0.5px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(49,46,129,0.55)) !important;
    border: 1px solid rgba(148,163,184,0.24) !important;
    box-shadow: 0 10px 24px rgba(0,0,0,0.18) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: linear-gradient(135deg, rgba(30,41,59,0.80), rgba(36,40,60,0.82)) !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px !important;
    font-weight: 720 !important;
}
.language-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(37,99,235,0.18)) !important;
}
.welcome-card {
    display: grid;
    grid-template-columns: 1.35fr 0.85fr;
    gap: 18px;
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(15,23,42,0.82), rgba(154,52,18,0.42));
    border: 1px solid rgba(251,146,60,0.30);
    box-shadow: 0 18px 42px rgba(0,0,0,0.26);
    margin: 10px 0 22px 0;
}
.welcome-card h3 { margin: 0 0 8px 0; font-size: 25px; }
.welcome-card p { margin: 0 0 8px 0; color: #e5e7eb; }
.welcome-quote { font-style: italic; color: #fde68a !important; }
.ai-avatar-card {
    border-radius: 18px;
    padding: 16px 18px;
    background: linear-gradient(135deg, rgba(37,99,235,0.32), rgba(16,185,129,0.24));
    border: 1px solid rgba(148,163,184,0.22);
}
.ai-avatar { font-size: 34px; margin-bottom: 4px; }
.ai-avatar-card ul { margin: 8px 0 0 18px; padding: 0; }
.ai-avatar-card li { margin: 5px 0; }
@media (max-width: 900px) {
    .welcome-card { grid-template-columns: 1fr; }
}

/* === V10 SIDEBAR CLEANUP === */
.side-menu-title {
    font-size: 24px !important;
    font-weight: 720 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px !important;
    font-weight: 680 !important;
}
section[data-testid="stSidebar"] .side-note {
    margin-top: 20px !important;
    padding: 22px 24px !important;
    background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(49,46,129,0.82)) !important;
    border: 1px solid rgba(129,140,248,0.28) !important;
    box-shadow: 0 14px 32px rgba(0,0,0,0.26) !important;
}
section[data-testid="stSidebar"] .side-note b {
    font-weight: 720 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(49,46,129,0.86)) !important;
    border: 1px solid rgba(129,140,248,0.30) !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* === V13 FINAL UI TUNE: SIDEBAR UP + WIDER TABS + CLEAN LINES === */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.45rem !important;
    padding-bottom: 0.65rem !important;
}
section[data-testid="stSidebar"] hr {
    margin-top: 18px !important;
    margin-bottom: 18px !important;
}
section[data-testid="stSidebar"] .side-menu-title {
    margin-top: 4px !important;
    margin-bottom: 14px !important;
    font-size: 23px !important;
    font-weight: 690 !important;
    letter-spacing: -0.35px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 10px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    height: 54px !important;
    min-height: 54px !important;
    border-radius: 16px !important;
    padding: 0 16px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 13.5px !important;
    font-weight: 650 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    min-height: 54px !important;
    border-radius: 16px !important;
}
section[data-testid="stSidebar"] .side-note {
    margin-top: 10px !important;
    padding: 18px 20px !important;
    border-radius: 20px !important;
}
section[data-testid="stSidebar"] .side-note b {
    font-size: 14px !important;
    font-weight: 650 !important;
    line-height: 1.35 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    padding: 0.75rem 1.25rem !important;
    border-radius: 15px !important;
}

/* Wider, calmer tab buttons */
div[data-testid="stTabs"] {
    margin-top: 18px !important;
    margin-bottom: 18px !important;
}
div[data-testid="stTabs"] div[role="tablist"] {
    gap: 12px !important;
    padding: 12px 0 12px 0 !important;
    margin: 18px 0 18px 0 !important;
    border-top: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.24) !important;
    box-shadow: none !important;
}
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {
    width: 172px !important;
    min-width: 172px !important;
    max-width: 172px !important;
    height: 50px !important;
    min-height: 50px !important;
    max-height: 50px !important;
    padding: 0 14px !important;
    border-radius: 16px !important;
}
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] div,
div[data-testid="stTabs"] [role="tab"] div,
div[data-testid="stTabs"] button[role="tab"] span,
div[data-testid="stTabs"] [role="tab"] span {
    font-size: 14.5px !important;
    font-weight: 680 !important;
    letter-spacing: -0.1px !important;
}

/* Remove the extra bottom separator after the tabs/welcome area */
div[data-testid="stTabs"] + hr,
div[data-testid="stTabs"] ~ hr:first-of-type {
    display: none !important;
}
.welcome-card {
    margin-top: 8px !important;
    margin-bottom: 18px !important;
}
</style>
""", unsafe_allow_html=True)


DB_NAME = "student_tracker.db"


def init_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            student_id TEXT,
            faculty TEXT,
            semester INTEGER,
            age INTEGER,
            height_cm INTEGER,
            weight_kg INTEGER,
            goal TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_date TEXT UNIQUE,
            mood_text TEXT,
            sleep_hours REAL,
            study_hours REAL,
            focus_level REAL,
            stress_level REAL,
            exercise_minutes REAL,
            task_completion REAL,
            water_liters REAL,
            nutrition_quality REAL,
            steps INTEGER,
            sentiment_score REAL,
            wellness_score REAL,
            productivity_score REAL,
            risk_score REAL
        )
        """
    )
    conn.commit()
    conn.close()


def save_profile_to_db(profile):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO student_profile (
            name, student_id, faculty, semester, age,
            height_cm, weight_kg, goal, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile["name"],
            profile["student_id"],
            profile["faculty"],
            profile["semester"],
            profile["age"],
            profile["height_cm"],
            profile["weight_kg"],
            profile["goal"],
            str(date.today()),
        ),
    )
    conn.commit()
    conn.close()


def load_records_from_db():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM daily_records ORDER BY record_date", conn)
    conn.close()
    return df


analyzer = SentimentIntensityAnalyzer()
init_database()

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "Student",
        "student_id": "0000",
        "faculty": "Yazılım Mühendisliği",
        "semester": 2,
        "age": 22,
        "height_cm": 175,
        "weight_kg": 70,
        "goal": "Verimliliği Artır",
    }

if "records" not in st.session_state:
    today = date.today()
    st.session_state.records = pd.DataFrame(
        {
            "date": [today - timedelta(days=i) for i in range(13, -1, -1)],
            "mood_text": [
                "I feel tired", "I am okay", "I feel stressed", "I feel focused",
                "I am motivated", "I feel good", "I feel tired but ready",
                "I am productive", "I feel calm", "I feel focused",
                "I feel low energy", "I feel disciplined", "I feel strong",
                "I feel balanced",
            ],
            "sleep_hours": [5, 6, 4, 7, 8, 7, 6, 8, 7, 8, 5, 7, 8, 7],
            "study_hours": [2, 3, 1, 4, 5, 4, 3, 6, 4, 5, 2, 5, 6, 4],
            "focus_level": [4, 5, 3, 7, 8, 7, 6, 9, 7, 8, 4, 8, 9, 7],
            "stress_level": [8, 7, 9, 5, 3, 4, 6, 2, 4, 3, 8, 4, 2, 4],
            "exercise_minutes": [0, 10, 0, 20, 30, 25, 15, 40, 25, 35, 5, 25, 45, 30],
            "task_completion": [30, 45, 25, 70, 85, 75, 60, 92, 78, 88, 35, 80, 95, 76],
            "water_liters": [1.0, 1.4, 0.8, 1.8, 2.0, 1.9, 1.5, 2.3, 2.0, 2.2, 1.1, 2.0, 2.5, 2.1],
            "nutrition_quality": [4, 5, 3, 7, 8, 7, 6, 9, 8, 8, 4, 8, 9, 7],
            "steps": [2500, 4200, 1800, 6500, 7200, 6900, 5000, 9800, 7600, 8500, 3100, 7000, 10200, 8200],
        }
    )


def calculate_scores(df):
    df = df.copy()
    df["sentiment_score"] = df["mood_text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["wellness_score"] = (
        (df["sleep_hours"] / 12 * 18)
        + (df["focus_level"] / 10 * 18)
        + ((10 - df["stress_level"]) / 10 * 18)
        + (df["exercise_minutes"].clip(0, 60) / 60 * 16)
        + (df["water_liters"].clip(0, 2.5) / 2.5 * 15)
        + (df["nutrition_quality"] / 10 * 15)
    ).round(2)
    df["productivity_score"] = (
        (df["study_hours"] / 10 * 25)
        + (df["task_completion"] / 100 * 35)
        + (df["focus_level"] / 10 * 25)
        + (df["sentiment_score"].clip(lower=0) * 15)
    ).round(2)
    df["risk_score"] = (
        (df["stress_level"] * 7)
        + ((12 - df["sleep_hours"]) * 3)
        + ((10 - df["focus_level"]) * 3)
        + ((2.5 - df["water_liters"].clip(0, 2.5)) * 5)
    ).round(2)
    return df


def save_daily_record_to_db(row):
    scored = calculate_scores(row).iloc[0]
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO daily_records (
            record_date, mood_text, sleep_hours, study_hours, focus_level,
            stress_level, exercise_minutes, task_completion, water_liters,
            nutrition_quality, steps, sentiment_score, wellness_score,
            productivity_score, risk_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(scored["date"]), scored["mood_text"], float(scored["sleep_hours"]),
            float(scored["study_hours"]), float(scored["focus_level"]),
            float(scored["stress_level"]), float(scored["exercise_minutes"]),
            float(scored["task_completion"]), float(scored["water_liters"]),
            float(scored["nutrition_quality"]), int(scored["steps"]),
            float(scored["sentiment_score"]), float(scored["wellness_score"]),
            float(scored["productivity_score"]), float(scored["risk_score"]),
        ),
    )
    conn.commit()
    conn.close()


def get_bmi_status(bmi):
    if bmi < 18.5:
        return t["underweight"], t["bmi_under_advice"]
    if bmi < 25:
        return t["normal"], t["bmi_normal_advice"]
    if bmi < 30:
        return t["overweight"], t["bmi_over_advice"]
    return t["high_range"], t["bmi_high_advice"]


def get_status_emoji(avg_productivity, avg_stress, avg_sleep):
    if avg_productivity >= 70 and avg_stress <= 5 and avg_sleep >= 7:
        return "🚀", t["excellent"]
    if avg_productivity >= 55 and avg_stress <= 6:
        return "🙂", t["stable"]
    if avg_stress >= 7 or avg_sleep < 6:
        return "⚠️", t["needs_attention"]
    return "🌱", t["improving"]


def get_motivation_quote(avg_productivity, avg_stress, avg_sleep):
    quotes = t.get("motivation_quotes", [t.get("quote_default", "Small consistent steps create strong long-term results.")])
    if not quotes:
        return t.get("quote_default", "Small consistent steps create strong long-term results.")
    today_index = date.today().toordinal() % len(quotes)
    return quotes[today_index]


def create_status_note(records, profile):
    if records.empty:
        return t["no_tracking_data"]

    avg_sleep = records["sleep_hours"].mean()
    avg_study = records["study_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_productivity = records["productivity_score"].mean()
    avg_wellness = records["wellness_score"].mean()
    avg_exercise = records["exercise_minutes"].mean()
    avg_water = records["water_liters"].mean()
    student_name = profile.get("name", "Student") or "Student"

    productivity_text = t["productivity_strong"] if avg_productivity >= 70 else t["productivity_moderate"] if avg_productivity >= 50 else t["productivity_needs"]
    stress_text = t["stress_high"] if avg_stress >= 7 else t["stress_watch"] if avg_stress >= 5 else t["stress_under_control"]
    sleep_text = t["sleep_low_note"] if avg_sleep < 6 else t["sleep_ok_note"]
    exercise_text = t["exercise_low_note"] if avg_exercise < 20 else t["exercise_ok_note"]
    water_text = t["water_low_note"] if avg_water < 1.8 else t["water_ok_note"]

    return t["status_note_template"].format(
        name=student_name,
        productivity_text=productivity_text,
        avg_sleep=avg_sleep,
        avg_study=avg_study,
        avg_wellness=avg_wellness,
        sleep_text=sleep_text,
        stress_text=stress_text,
        exercise_text=exercise_text,
        water_text=water_text,
    )


def get_ai_recommendations(records):
    avg_sleep = records["sleep_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_productivity = records["productivity_score"].mean()
    avg_nutrition = records["nutrition_quality"].mean()
    avg_exercise = records["exercise_minutes"].mean()
    avg_water = records["water_liters"].mean()

    messages = []
    messages.append(("warning", t["sleep_low"]) if avg_sleep < 6 else ("success", t["sleep_ok"]))
    messages.append(("error", t["stress_high_msg"]) if avg_stress > 6 else ("success", t["stress_ok"]))
    messages.append(("warning", t["nutrition_low"]) if avg_nutrition < 6 else ("success", t["nutrition_ok"]))
    messages.append(("warning", t["exercise_low"]) if avg_exercise < 20 else ("success", t["exercise_ok"]))
    messages.append(("warning", t["water_low"]) if avg_water < 1.8 else ("success", t["water_ok"]))

    if avg_productivity >= 70:
        messages.append(("success", t["productivity_strong_msg"]))
    elif avg_productivity >= 50:
        messages.append(("info", t["productivity_moderate_msg"]))
    else:
        messages.append(("warning", t["productivity_low_msg"]))

    return messages


def create_pdf_report(profile, weekly_summary, recommendations, status_note):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, t["pdf_title"])
    y -= 35
    pdf.setFont("Helvetica", 11)

    for line in [
        f"{t['student_label']}: {profile.get('name', 'Student')}",
        f"Student ID: {profile.get('student_id', '0000')}",
        f"{t['faculty']}: {profile.get('faculty', t['faculty_software'])}",
        f"{t['semester']}: {profile.get('semester', 2)}",
        f"{t['goal']}: {profile.get('goal', t['goal_improve_productivity'])}",
    ]:
        pdf.drawString(50, y, line)
        y -= 18

    y -= 20
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, t["student_status_summary"])
    y -= 22
    pdf.setFont("Helvetica", 10)

    for i in range(0, len(status_note), 90):
        pdf.drawString(60, y, status_note[i:i + 90])
        y -= 16

    y -= 15
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, t["weekly_metrics"])
    y -= 25
    pdf.setFont("Helvetica", 11)

    for _, row in weekly_summary.iterrows():
        pdf.drawString(60, y, f"{row[t['metric']]}: {row[t['value']]}")
        y -= 18

    y -= 20
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, t["ai_coach_recommendations"])
    y -= 25
    pdf.setFont("Helvetica", 10)

    for _, msg in recommendations:
        pdf.drawString(60, y, f"- {msg}"[:95])
        y -= 18
        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)

    pdf.save()
    buffer.seek(0)
    return buffer


training_data = pd.DataFrame(
    {
        "sleep_hours": [4, 5, 6, 7, 8, 3, 9, 2, 6, 7, 5, 8, 6, 4, 9, 7, 8, 5, 6, 3, 7, 8, 6, 5],
        "study_hours": [2, 3, 4, 5, 6, 1, 7, 1, 3, 6, 2, 5, 4, 2, 8, 5, 7, 3, 4, 1, 6, 7, 4, 2],
        "sentiment_score": [-0.6, -0.3, 0.0, 0.4, 0.7, -0.8, 0.8, -0.9, 0.2, 0.5, -0.1, 0.6, 0.1, -0.5, 0.9, 0.4, 0.7, -0.2, 0.1, -0.7, 0.6, 0.8, 0.2, -0.4],
        "focus_level": [3, 4, 5, 7, 8, 2, 9, 1, 5, 8, 4, 7, 6, 3, 9, 7, 9, 5, 6, 2, 8, 9, 6, 4],
        "stress_level": [8, 7, 5, 4, 2, 9, 1, 10, 5, 3, 6, 2, 4, 8, 1, 4, 2, 7, 5, 9, 3, 2, 5, 8],
        "exercise_minutes": [0, 10, 20, 25, 30, 0, 40, 0, 15, 35, 10, 30, 20, 5, 45, 25, 45, 10, 20, 0, 35, 50, 20, 5],
        "task_completion": [30, 40, 55, 70, 85, 20, 95, 15, 60, 80, 45, 75, 65, 35, 98, 72, 94, 45, 60, 20, 82, 96, 64, 38],
        "water_liters": [1.0, 1.2, 1.5, 1.8, 2.1, 0.8, 2.4, 0.6, 1.7, 2.0, 1.3, 2.1, 1.8, 1.0, 2.5, 1.9, 2.4, 1.2, 1.6, 0.7, 2.1, 2.5, 1.7, 1.0],
        "nutrition_quality": [4, 5, 6, 7, 8, 3, 9, 2, 6, 8, 5, 8, 7, 4, 9, 7, 9, 5, 6, 3, 8, 9, 6, 4],
        "productivity": [0, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 0, 2, 1, 2, 1, 1, 0, 2, 2, 1, 0],
    }
)

features = [
    "sleep_hours",
    "study_hours",
    "sentiment_score",
    "focus_level",
    "stress_level",
    "exercise_minutes",
    "task_completion",
    "water_liters",
    "nutrition_quality",
]

labels = {0: "Low", 1: "Medium", 2: "High"}

model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(training_data[features], training_data["productivity"])
accuracy = accuracy_score(training_data["productivity"], model.predict(training_data[features]))


def render_header():
    st.markdown(f'<div class="main-title">{t["app_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{t["subtitle"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero-card">
        <h2>{t["concept_title"]}</h2>
        <p>{t["concept_text"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def section_extra_texts():
    lang = st.session_state.language
    data = {
        "Türkçe": {
            "nutrition_input_title": "🍽️ Günlük Beslenme Girişi",
            "calorie_goal": "Günlük Kalori Hedefi",
            "calories_taken": "Bugün Alınan Kalori",
            "protein_g": "Protein (g)",
            "carb_g": "Karbonhidrat (g)",
            "fat_g": "Yağ (g)",
            "meal_count": "Öğün Sayısı",
            "macro_calories": "Makrolardan Hesaplanan Kalori",
            "remaining_calories": "Kalan Kalori",
            "nutrition_note": "Beslenme Notu",
            "nutrition_detail_note": "Kalori hedefini ve makroları düzenli takip etmek kilo, enerji ve odak yönetimini daha net hale getirir.",
            "protein_low": "Protein düşük görünüyor. Ana öğünlere yumurta, yoğurt, tavuk, balık veya bakliyat ekle.",
            "calorie_low": "Kalori hedefinin altındasın. Enerji düşüşü yaşamamak için dengeli bir ek öğün iyi olur.",
            "calorie_high": "Kalori hedefinin üstündesin. Bu sorun olmak zorunda değil; hedefin kilo almaksa mantıklı, yağ kaybıysa kontrol et.",
            "nutrition_good": "Bugünkü beslenme dengeli görünüyor.",
            "exercise_input_title": "🏃 Egzersiz Girişi",
            "exercise_type": "Egzersiz Türü",
            "duration_min": "Süre (dk)",
            "intensity": "Yoğunluk",
            "body_weight": "Vücut Ağırlığı (kg)",
            "estimated_burn": "Tahmini Yakılan Kalori",
            "exercise_note": "Egzersiz Notu",
            "walking": "Yürüyüş",
            "running": "Koşu",
            "strength": "Ağırlık / Kuvvet",
            "cycling": "Bisiklet",
            "mobility": "Mobilite / Esneme",
            "low": "Düşük", "medium": "Orta", "high": "Yüksek",
            "exercise_good": "Bugünkü hareket seviyesi iyi. Bunu haftalık rutine bağlarsan güçlü alışkanlık olur.",
            "exercise_low": "Egzersiz düşük. Bugün 15-20 dakika yürüyüş bile sistemi canlı tutar.",
        },
        "English": {
            "nutrition_input_title": "🍽️ Daily Nutrition Entry", "calorie_goal": "Daily Calorie Goal", "calories_taken": "Calories Taken Today", "protein_g": "Protein (g)", "carb_g": "Carbohydrates (g)", "fat_g": "Fat (g)", "meal_count": "Meal Count", "macro_calories": "Calories From Macros", "remaining_calories": "Remaining Calories", "nutrition_note": "Nutrition Note", "nutrition_detail_note": "Tracking calories and macros makes weight, energy and focus management more precise.", "protein_low": "Protein looks low. Add eggs, yogurt, chicken, fish or legumes to your meals.", "calorie_low": "You are below your calorie goal. A balanced extra meal can help prevent low energy.", "calorie_high": "You are above your calorie goal. This may be fine for weight gain, but check it if your goal is fat loss.", "nutrition_good": "Today's nutrition looks balanced.",
            "exercise_input_title": "🏃 Exercise Entry", "exercise_type": "Exercise Type", "duration_min": "Duration (min)", "intensity": "Intensity", "body_weight": "Body Weight (kg)", "estimated_burn": "Estimated Calories Burned", "exercise_note": "Exercise Note", "walking": "Walking", "running": "Running", "strength": "Strength Training", "cycling": "Cycling", "mobility": "Mobility / Stretching", "low": "Low", "medium": "Medium", "high": "High", "exercise_good": "Today's activity level is good. Connect it to a weekly routine and it becomes a strong habit.", "exercise_low": "Exercise is low. Even a 15-20 minute walk keeps the system active.",
        },
        "Deutsch": {
            "nutrition_input_title": "🍽️ Tägliche Ernährungseingabe", "calorie_goal": "Tägliches Kalorienziel", "calories_taken": "Heute aufgenommene Kalorien", "protein_g": "Protein (g)", "carb_g": "Kohlenhydrate (g)", "fat_g": "Fett (g)", "meal_count": "Anzahl der Mahlzeiten", "macro_calories": "Kalorien aus Makros", "remaining_calories": "Verbleibende Kalorien", "nutrition_note": "Ernährungsnotiz", "nutrition_detail_note": "Kalorien und Makros regelmäßig zu verfolgen macht Gewicht, Energie und Fokus besser steuerbar.", "protein_low": "Protein wirkt niedrig. Ergänze Eier, Joghurt, Hähnchen, Fisch oder Hülsenfrüchte.", "calorie_low": "Du liegst unter deinem Kalorienziel. Eine ausgewogene Zusatzmahlzeit kann Energieverlust verhindern.", "calorie_high": "Du liegst über deinem Kalorienziel. Für Gewichtszunahme kann das passen, bei Fettverlust solltest du es prüfen.", "nutrition_good": "Die heutige Ernährung wirkt ausgewogen.",
            "exercise_input_title": "🏃 Trainingseingabe", "exercise_type": "Trainingsart", "duration_min": "Dauer (Min.)", "intensity": "Intensität", "body_weight": "Körpergewicht (kg)", "estimated_burn": "Geschätzte verbrannte Kalorien", "exercise_note": "Trainingsnotiz", "walking": "Gehen", "running": "Laufen", "strength": "Krafttraining", "cycling": "Radfahren", "mobility": "Mobilität / Dehnen", "low": "Niedrig", "medium": "Mittel", "high": "Hoch", "exercise_good": "Das heutige Aktivitätsniveau ist gut. Als Wochenroutine wird daraus eine starke Gewohnheit.", "exercise_low": "Training ist niedrig. Schon 15-20 Minuten Gehen halten das System aktiv.",
        },
        "Русский": {
            "nutrition_input_title": "🍽️ Ежедневное питание", "calorie_goal": "Дневная цель калорий", "calories_taken": "Калории за сегодня", "protein_g": "Белок (г)", "carb_g": "Углеводы (г)", "fat_g": "Жиры (г)", "meal_count": "Количество приемов пищи", "macro_calories": "Калории по макроэлементам", "remaining_calories": "Осталось калорий", "nutrition_note": "Заметка по питанию", "nutrition_detail_note": "Отслеживание калорий и макроэлементов помогает точнее управлять весом, энергией и фокусом.", "protein_low": "Белка мало. Добавь яйца, йогурт, курицу, рыбу или бобовые.", "calorie_low": "Ты ниже цели калорий. Сбалансированный перекус поможет избежать упадка энергии.", "calorie_high": "Ты выше цели калорий. Для набора веса это может быть нормально, но при снижении жира стоит проверить.", "nutrition_good": "Сегодняшнее питание выглядит сбалансированным.",
            "exercise_input_title": "🏃 Ввод тренировки", "exercise_type": "Тип упражнения", "duration_min": "Длительность (мин)", "intensity": "Интенсивность", "body_weight": "Вес тела (кг)", "estimated_burn": "Примерно сожжено калорий", "exercise_note": "Заметка о тренировке", "walking": "Ходьба", "running": "Бег", "strength": "Силовая тренировка", "cycling": "Велосипед", "mobility": "Мобилити / растяжка", "low": "Низкая", "medium": "Средняя", "high": "Высокая", "exercise_good": "Сегодняшняя активность хорошая. Если привязать ее к неделе, получится сильная привычка.", "exercise_low": "Активность низкая. Даже 15-20 минут ходьбы поддержат систему.",
        },
        "Español": {
            "nutrition_input_title": "🍽️ Registro diario de nutrición", "calorie_goal": "Objetivo diario de calorías", "calories_taken": "Calorías consumidas hoy", "protein_g": "Proteína (g)", "carb_g": "Carbohidratos (g)", "fat_g": "Grasa (g)", "meal_count": "Número de comidas", "macro_calories": "Calorías por macros", "remaining_calories": "Calorías restantes", "nutrition_note": "Nota de nutrición", "nutrition_detail_note": "Seguir calorías y macros hace más claro el control de peso, energía y enfoque.", "protein_low": "La proteína parece baja. Añade huevos, yogur, pollo, pescado o legumbres.", "calorie_low": "Estás por debajo de tu objetivo calórico. Una comida extra equilibrada puede evitar baja energía.", "calorie_high": "Estás por encima del objetivo calórico. Puede estar bien si quieres subir de peso, pero revísalo si buscas perder grasa.", "nutrition_good": "La nutrición de hoy parece equilibrada.",
            "exercise_input_title": "🏃 Registro de ejercicio", "exercise_type": "Tipo de ejercicio", "duration_min": "Duración (min)", "intensity": "Intensidad", "body_weight": "Peso corporal (kg)", "estimated_burn": "Calorías quemadas estimadas", "exercise_note": "Nota de ejercicio", "walking": "Caminar", "running": "Correr", "strength": "Fuerza / pesas", "cycling": "Bicicleta", "mobility": "Movilidad / estiramiento", "low": "Baja", "medium": "Media", "high": "Alta", "exercise_good": "El nivel de actividad de hoy es bueno. Si lo conectas a una rutina semanal, se vuelve un hábito fuerte.", "exercise_low": "El ejercicio es bajo. Incluso caminar 15-20 minutos mantiene el sistema activo.",
        },
    }
    return data.get(lang, data["English"])

def render_login_profile():
    st.subheader(t["profile_title"])

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input(t["full_name"], st.session_state.profile["name"])
        student_id = st.text_input(t["student_id"], st.session_state.profile["student_id"])

    with col2:
        faculty = st.selectbox(
            t["faculty"],
            [t["faculty_software"], "Computer Science", "AI Engineering", "Business", "Design"],
            index=0,
        )
        semester = st.slider(t["semester"], 1, 8, st.session_state.profile["semester"])

    with col3:
        age = st.number_input(t["age"], min_value=10, max_value=80, value=st.session_state.profile["age"])
        goal = st.selectbox(
            t["main_goal"],
            [t["goal_improve_productivity"], t["goal_gain_weight"], t["goal_lose_fat"], t["goal_reduce_stress"], t["goal_improve_sleep"], t["goal_build_discipline"]],
            index=0,
        )

    col4, col5 = st.columns(2)

    with col4:
        height_cm = st.number_input(t["height"], min_value=120, max_value=230, value=st.session_state.profile["height_cm"])

    with col5:
        weight_kg = st.number_input(t["weight"], min_value=35, max_value=180, value=st.session_state.profile["weight_kg"])

    if st.button(t["save_profile"]):
        st.session_state.profile = {
            "name": name,
            "student_id": student_id,
            "faculty": faculty,
            "semester": semester,
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "goal": goal,
        }
        save_profile_to_db(st.session_state.profile)
        st.success(t["profile_saved"])

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    bmi_status, bmi_advice = get_bmi_status(bmi)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", bmi)
    m2.metric(t["bmi_status"], bmi_status)
    m3.metric(t["semester"], semester)
    m4.metric(t["main_goal"], goal)

    st.info(bmi_advice)



def _current_streak(records, column, threshold):
    if records.empty or column not in records.columns:
        return 0
    df = records.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    streak = 0
    expected = date.today()
    values = {row["date"]: row[column] for _, row in df.iterrows()}
    while expected in values and values[expected] >= threshold:
        streak += 1
        expected = expected - timedelta(days=1)
    return streak


def render_home_booster():
    records = calculate_scores(st.session_state.records)
    profile = st.session_state.profile
    name = profile.get("name", "Student") or "Student"

    avg_productivity = records["productivity_score"].mean()
    avg_wellness = records["wellness_score"].mean()
    avg_sleep = records["sleep_hours"].mean()
    avg_stress = records["stress_level"].mean()

    study_streak = _current_streak(records, "study_hours", 1)
    water_streak = _current_streak(records, "water_liters", 1.8)
    exercise_streak = _current_streak(records, "exercise_minutes", 20)

    quote = get_motivation_quote(avg_productivity, avg_stress, avg_sleep)
    energy_label = t["energy_good"] if avg_sleep >= 6.5 and avg_stress <= 6 else t["energy_watch"]
    risk_label = t["risk_high"] if avg_stress >= 7 or avg_sleep < 5.5 else t["risk_medium"] if avg_stress >= 5.5 or avg_sleep < 6.5 else t["risk_low"]

    st.markdown(
        f"""
        <div class="welcome-card">
            <div>
                <h3>{t["welcome_title"].format(name=name)}</h3>
                <p>{t["welcome_text"]}</p>
                <p class="welcome-quote">“{quote}”</p>
                <div class="welcome-detail-grid">
                    <div class="welcome-mini-card"><span>{t["welcome_focus_label"]}</span><b>{t["welcome_focus_value"]}</b></div>
                    <div class="welcome-mini-card"><span>{t["quick_energy"]}</span><b>{energy_label}</b></div>
                    <div class="welcome-mini-card"><span>{t["quick_risk"]}</span><b>{risk_label}</b></div>
                </div>
                <div class="welcome-next-step">
                    <b>{t["welcome_next_step"]}</b>
                    <span>{t["welcome_next_step_value"]}</span><br>
                    <small>{t["welcome_profile_hint"]}</small>
                </div>
            </div>
            <div class="ai-avatar-card">
                <div class="ai-avatar">🤖</div>
                <b>{t["ai_mission_title"]}</b>
                <ul>
                    <li>{t["mission_water"]}</li>
                    <li>{t["mission_study"]}</li>
                    <li>{t["mission_walk"]}</li>
                    <li>{t["mission_sleep"]}</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t["productivity_kpi"], f"{avg_productivity:.0f}/100")
    k2.metric(t["wellness_kpi"], f"{avg_wellness:.0f}/100")
    k3.metric(t["sleep_kpi"], f"{avg_sleep:.1f}h")
    k4.metric(t["stress_kpi"], f"{avg_stress:.1f}/10")

    s1, s2, s3 = st.columns(3)
    s1.metric(f"🔥 {t['study_streak']}", f"{study_streak} {t['day_unit']}")
    s2.metric(f"💧 {t['water_streak']}", f"{water_streak} {t['day_unit']}")
    s3.metric(f"🏃 {t['exercise_streak']}", f"{exercise_streak} {t['day_unit']}")


def render_student_bottom_summary():
    records = calculate_scores(st.session_state.records)
    status_note = create_status_note(records, st.session_state.profile)

    avg_productivity = records["productivity_score"].mean()
    avg_stress = records["stress_level"].mean()
    avg_sleep = records["sleep_hours"].mean()

    emoji, status_label = get_status_emoji(avg_productivity, avg_stress, avg_sleep)
    quote = get_motivation_quote(avg_productivity, avg_stress, avg_sleep)

    st.divider()
    st.markdown(
        f"""
        <div class="status-card">
        <h3>{emoji} {t["student_status"]}: {status_label}</h3>
        <p>{status_note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="quote-card">
        <h3>{t["daily_motivation"]}</h3>
        <p><i>"{quote}"</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings():
    st.subheader(t["settings_title"])
    st.write(t["settings_text"])

    bg_options = [
        t["background_soft"],
        t["background_dark"],
        t["background_light"],
        t["background_green"],
        t["background_orange"],
    ]

    current_bg = st.session_state.background_mode
    if current_bg not in bg_options:
        current_bg = bg_options[0]

    chosen_bg = st.selectbox(t["background"], bg_options, index=bg_options.index(current_bg))
    st.session_state.background_mode = chosen_bg

    st.markdown(
        f"""
        <div class="info-card">
        <h3>Dashboard Preview</h3>
        <p><b>Theme:</b> {st.session_state.theme_name}</p>
        <p><b>Background:</b> {st.session_state.background_mode}</p>
        <p><b>Language:</b> {st.session_state.language}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning("After changing background or language, Streamlit may rerun automatically. If the visual does not update instantly, refresh once.")


def render_database_history():
    st.subheader(f"🗄️ {t['db']}")
    db_records = load_records_from_db()

    if db_records.empty:
        st.info(t["no_records"])
    else:
        st.success(t["records_loaded"].format(count=len(db_records)))
        st.dataframe(db_records, use_container_width=True)
        db_records["record_date"] = pd.to_datetime(db_records["record_date"])
        chart_data = db_records.set_index("record_date")[
            ["sleep_hours", "study_hours", "stress_level", "wellness_score", "productivity_score"]
        ]
        st.line_chart(chart_data)

        status_note = create_status_note(db_records, st.session_state.profile)
        st.markdown(
            f"""
            <div class="card-green">
            <h3>{t["current_student_status_short"]}</h3>
            <p>{status_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )



# === V17 DEEP PRODUCT ANALYSIS HELPERS ===
def advanced_texts():
    lang = st.session_state.language
    base = {
        "advanced_analysis": "🚀 Gelişmiş Performans Analizi",
        "nutrition_dashboard": "Beslenme Performans Paneli",
        "exercise_dashboard": "Egzersiz Performans Paneli",
        "coach_title": "Stratejik AI Koç Paneli",
        "report_title": "Detaylı Haftalık Rapor",
        "calorie_balance": "Kalori Dengesi",
        "macro_split": "Makro Dağılımı",
        "exercise_efficiency": "Egzersiz Verimliliği",
        "burned_calories": "Yakılan Kalori",
        "net_calories": "Net Kalori",
        "protein_target": "Protein Hedefi",
        "analysis_comment": "Analiz Yorumu",
        "chart_commentary": "Grafik Yorumu",
        "risk_interpretation": "Risk Yorumu",
        "coach_strategy": "Bugünün Stratejik Planı",
        "priority_plan": "Öncelik Planı",
        "nutrition_strategy": "Beslenme Stratejisi",
        "exercise_strategy": "Egzersiz Stratejisi",
        "study_strategy": "Çalışma Stratejisi",
        "recovery_strategy": "Toparlanma Stratejisi",
        "executive_summary": "Yönetici Özeti",
        "weekly_decision": "Haftalık Karar",
        "next_actions": "Sonraki Aksiyonlar",
        "not_enough_calories": "Kalori hedefinin altındasın. Gün sonunda protein + kompleks karbonhidrat içeren dengeli bir ek öğün mantıklı olur.",
        "calories_high": "Kalori hedefini aşıyorsun. Eğer hedef kilo almak değilse porsiyon ve yağ kaynaklarını kontrol et.",
        "calories_good": "Kalori dengen hedefe yakın. Bu, sürdürülebilir performans için iyi bir sinyal.",
        "protein_low_detail": "Protein düşük görünüyor. Kas gelişimi, toparlanma ve tokluk için protein hedefini artır.",
        "protein_good_detail": "Protein hedefin güçlü. Egzersiz ve odak performansını destekler.",
        "exercise_low_detail": "Egzersiz süresi düşük. Bugün en az 20 dakikalık yürüyüş veya mobilite ekle.",
        "exercise_good_detail": "Egzersiz süren iyi. Bunu haftalık seriye bağlarsan alışkanlık oluşur.",
        "sleep_low_detail": "Uyku ortalaman düşük. Verimlilik grafiğindeki düşüşlerin ana sebebi bu olabilir.",
        "sleep_good_detail": "Uyku ortalaman kabul edilebilir. Bunu sabit saatlerle koru.",
        "stress_high_detail": "Stres trendi yüksek. Planı küçült, toparlanma aralığı ekle, derin çalışma bloklarını azalt.",
        "stress_ok_detail": "Stres seviyesi yönetilebilir. Yine de yoğun günlerde kısa mola sistemi kullan.",
        "graph_sleep": "Uyku grafiği odak ve verimlilik için temel sinyaldir. 6 saatin altındaki günler riskli kabul edilir.",
        "graph_study": "Çalışma grafiğinde süre kadar istikrar önemlidir. Ani iniş çıkışlar planlama sorunu gösterebilir.",
        "graph_wellness": "Sağlık skoru uyku, su, stres, beslenme ve egzersizin birleşik fotoğrafıdır.",
        "report_intro": "Bu rapor öğrencinin akademik, fiziksel ve davranışsal performansını birlikte değerlendirir.",
        "mission": "Bugünün net görevi: küçük ama ölçülebilir ilerleme üret.",
    }
    translations = {
        "English": {
            "advanced_analysis": "🚀 Advanced Performance Analysis", "nutrition_dashboard": "Nutrition Performance Panel", "exercise_dashboard": "Exercise Performance Panel", "coach_title": "Strategic AI Coach Panel", "report_title": "Detailed Weekly Report", "calorie_balance": "Calorie Balance", "macro_split": "Macro Split", "exercise_efficiency": "Exercise Efficiency", "burned_calories": "Burned Calories", "net_calories": "Net Calories", "protein_target": "Protein Target", "analysis_comment": "Analysis Comment", "chart_commentary": "Chart Commentary", "risk_interpretation": "Risk Interpretation", "coach_strategy": "Today’s Strategic Plan", "priority_plan": "Priority Plan", "nutrition_strategy": "Nutrition Strategy", "exercise_strategy": "Exercise Strategy", "study_strategy": "Study Strategy", "recovery_strategy": "Recovery Strategy", "executive_summary": "Executive Summary", "weekly_decision": "Weekly Decision", "next_actions": "Next Actions", "not_enough_calories": "You are below your calorie target. A balanced extra meal with protein and complex carbs makes sense.", "calories_high": "You are above your calorie target. If weight gain is not the goal, control portions and fat sources.", "calories_good": "Your calorie balance is close to target. This is a good signal for sustainable performance.", "protein_low_detail": "Protein looks low. Increase protein for muscle repair, recovery and satiety.", "protein_good_detail": "Your protein target is strong. It supports exercise and focus performance.", "exercise_low_detail": "Exercise duration is low. Add at least 20 minutes of walking or mobility today.", "exercise_good_detail": "Exercise duration is good. Connect it to a weekly streak to build the habit.", "sleep_low_detail": "Average sleep is low. It may be the main reason behind productivity drops.", "sleep_good_detail": "Average sleep is acceptable. Protect it with consistent sleep times.", "stress_high_detail": "Stress trend is high. Reduce plan size, add recovery blocks, and lower deep-work intensity.", "stress_ok_detail": "Stress is manageable. Still use short breaks on intense days.", "graph_sleep": "The sleep chart is the core signal for focus and productivity. Days below 6 hours are risky.", "graph_study": "The study chart values consistency as much as duration. Sharp swings may indicate planning problems.", "graph_wellness": "The wellness score combines sleep, water, stress, nutrition and exercise.", "report_intro": "This report evaluates academic, physical and behavioral performance together.", "mission": "Today’s clear mission: produce small but measurable progress."
        },
        "Deutsch": {
            "advanced_analysis": "🚀 Erweiterte Leistungsanalyse", "nutrition_dashboard": "Ernährungs-Performance", "exercise_dashboard": "Trainings-Performance", "coach_title": "Strategisches KI-Coach-Panel", "report_title": "Detaillierter Wochenbericht", "calorie_balance": "Kalorienbilanz", "macro_split": "Makroverteilung", "exercise_efficiency": "Trainingseffizienz", "burned_calories": "Verbrannte Kalorien", "net_calories": "Netto-Kalorien", "protein_target": "Proteinziel", "analysis_comment": "Analysekommentar", "chart_commentary": "Diagrammkommentar", "risk_interpretation": "Risikobewertung", "coach_strategy": "Strategischer Plan für heute", "priority_plan": "Prioritätenplan", "nutrition_strategy": "Ernährungsstrategie", "exercise_strategy": "Trainingsstrategie", "study_strategy": "Lernstrategie", "recovery_strategy": "Erholungsstrategie", "executive_summary": "Kurzfazit", "weekly_decision": "Wöchentliche Entscheidung", "next_actions": "Nächste Aktionen", "not_enough_calories": "Du liegst unter deinem Kalorienziel. Eine zusätzliche Mahlzeit mit Protein und komplexen Kohlenhydraten wäre sinnvoll.", "calories_high": "Du liegst über deinem Kalorienziel. Wenn Gewichtszunahme nicht das Ziel ist, kontrolliere Portionen und Fettquellen.", "calories_good": "Deine Kalorienbilanz liegt nah am Ziel. Das ist ein gutes Signal für nachhaltige Leistung.", "protein_low_detail": "Protein wirkt niedrig. Erhöhe Protein für Regeneration, Muskelaufbau und Sättigung.", "protein_good_detail": "Dein Proteinziel ist stark. Es unterstützt Training und Fokus.", "exercise_low_detail": "Die Trainingsdauer ist niedrig. Ergänze heute mindestens 20 Minuten Gehen oder Mobilität.", "exercise_good_detail": "Die Trainingsdauer ist gut. Verbinde sie mit einer Wochenserie.", "sleep_low_detail": "Der Schlafdurchschnitt ist niedrig. Das kann der Hauptgrund für Produktivitätseinbrüche sein.", "sleep_good_detail": "Der Schlafdurchschnitt ist akzeptabel. Schütze ihn mit festen Schlafzeiten.", "stress_high_detail": "Der Stress-Trend ist hoch. Verkleinere den Plan, füge Erholung ein und reduziere Deep-Work-Intensität.", "stress_ok_detail": "Stress ist handhabbar. Nutze trotzdem kurze Pausen an intensiven Tagen.", "graph_sleep": "Das Schlafdiagramm ist ein Kernsignal für Fokus und Produktivität. Tage unter 6 Stunden sind riskant.", "graph_study": "Beim Lerndiagramm zählt Stabilität genauso wie Dauer. Starke Schwankungen zeigen Planungsprobleme.", "graph_wellness": "Der Wellness-Score kombiniert Schlaf, Wasser, Stress, Ernährung und Training.", "report_intro": "Dieser Bericht bewertet akademische, körperliche und verhaltensbezogene Leistung gemeinsam.", "mission": "Klare Mission für heute: kleine, messbare Fortschritte erzeugen."
        },
        "Español": {
            "advanced_analysis": "🚀 Análisis avanzado de rendimiento", "nutrition_dashboard": "Panel de nutrición", "exercise_dashboard": "Panel de ejercicio", "coach_title": "Panel estratégico del Coach IA", "report_title": "Informe semanal detallado", "calorie_balance": "Balance calórico", "macro_split": "Distribución de macros", "exercise_efficiency": "Eficiencia del ejercicio", "burned_calories": "Calorías quemadas", "net_calories": "Calorías netas", "protein_target": "Objetivo de proteína", "analysis_comment": "Comentario de análisis", "chart_commentary": "Comentario del gráfico", "risk_interpretation": "Interpretación de riesgo", "coach_strategy": "Plan estratégico de hoy", "priority_plan": "Plan de prioridades", "nutrition_strategy": "Estrategia de nutrición", "exercise_strategy": "Estrategia de ejercicio", "study_strategy": "Estrategia de estudio", "recovery_strategy": "Estrategia de recuperación", "executive_summary": "Resumen ejecutivo", "weekly_decision": "Decisión semanal", "next_actions": "Próximas acciones", "not_enough_calories": "Estás por debajo del objetivo calórico. Conviene una comida extra equilibrada con proteína y carbohidratos complejos.", "calories_high": "Estás por encima del objetivo calórico. Si no buscas subir de peso, controla porciones y grasas.", "calories_good": "Tu balance calórico está cerca del objetivo. Es una buena señal para rendimiento sostenible.", "protein_low_detail": "La proteína parece baja. Auméntala para recuperación, músculo y saciedad.", "protein_good_detail": "Tu objetivo de proteína es fuerte. Apoya el ejercicio y el enfoque.", "exercise_low_detail": "La duración del ejercicio es baja. Añade al menos 20 minutos de caminata o movilidad hoy.", "exercise_good_detail": "La duración del ejercicio es buena. Conéctala con una racha semanal.", "sleep_low_detail": "El promedio de sueño es bajo. Puede ser la causa principal de caídas de productividad.", "sleep_good_detail": "El promedio de sueño es aceptable. Protégelo con horarios estables.", "stress_high_detail": "La tendencia de estrés es alta. Reduce el plan, añade recuperación y baja la intensidad.", "stress_ok_detail": "El estrés es manejable. Aun así usa pausas cortas en días intensos.", "graph_sleep": "El gráfico de sueño es señal central para enfoque y productividad. Días bajo 6 horas son riesgosos.", "graph_study": "En el gráfico de estudio importa la constancia tanto como la duración. Saltos fuertes indican problemas de planificación.", "graph_wellness": "La puntuación de bienestar combina sueño, agua, estrés, nutrición y ejercicio.", "report_intro": "Este informe evalúa rendimiento académico, físico y conductual en conjunto.", "mission": "Misión clara de hoy: producir progreso pequeño pero medible."
        },
        "Русский": {
            "advanced_analysis": "🚀 Расширенный анализ продуктивности", "nutrition_dashboard": "Панель питания", "exercise_dashboard": "Панель тренировок", "coach_title": "Стратегическая панель ИИ-коуча", "report_title": "Подробный недельный отчет", "calorie_balance": "Баланс калорий", "macro_split": "Распределение макро", "exercise_efficiency": "Эффективность тренировки", "burned_calories": "Сожженные калории", "net_calories": "Чистые калории", "protein_target": "Цель по белку", "analysis_comment": "Комментарий анализа", "chart_commentary": "Комментарий к графику", "risk_interpretation": "Оценка риска", "coach_strategy": "Стратегический план на сегодня", "priority_plan": "План приоритетов", "nutrition_strategy": "Стратегия питания", "exercise_strategy": "Стратегия тренировки", "study_strategy": "Стратегия учебы", "recovery_strategy": "Стратегия восстановления", "executive_summary": "Краткий вывод", "weekly_decision": "Решение недели", "next_actions": "Следующие действия", "not_enough_calories": "Ты ниже цели по калориям. Добавь сбалансированный прием пищи с белком и сложными углеводами.", "calories_high": "Ты выше цели по калориям. Если набор веса не цель, контролируй порции и жиры.", "calories_good": "Баланс калорий близок к цели. Это хороший сигнал для устойчивой продуктивности.", "protein_low_detail": "Белка мало. Увеличь белок для восстановления, мышц и сытости.", "protein_good_detail": "Цель по белку хорошая. Это поддерживает тренировки и концентрацию.", "exercise_low_detail": "Тренировки мало. Добавь сегодня минимум 20 минут ходьбы или мобильности.", "exercise_good_detail": "Длительность тренировки хорошая. Свяжи это с недельной серией.", "sleep_low_detail": "Средний сон низкий. Это может быть основной причиной падения продуктивности.", "sleep_good_detail": "Средний сон приемлемый. Сохраняй стабильное время сна.", "stress_high_detail": "Тренд стресса высокий. Уменьши план, добавь восстановление и снизь интенсивность.", "stress_ok_detail": "Стресс управляемый. Но в тяжелые дни используй короткие паузы.", "graph_sleep": "График сна — ключевой сигнал фокуса и продуктивности. Дни ниже 6 часов рискованны.", "graph_study": "В графике учебы важна стабильность, а не только длительность. Резкие скачки показывают проблему планирования.", "graph_wellness": "Wellness Score объединяет сон, воду, стресс, питание и тренировки.", "report_intro": "Этот отчет оценивает учебную, физическую и поведенческую продуктивность вместе.", "mission": "Задача дня: создать небольшой, но измеримый прогресс."
        }
    }
    if lang in translations:
        base.update(translations[lang])
    return base


def current_user_inputs():
    return {
        "calorie_goal": st.session_state.get("nutrition_calorie_goal", 2600),
        "calories_taken": st.session_state.get("nutrition_calories_taken", 2100),
        "protein_g": st.session_state.get("nutrition_protein_g", 110),
        "carb_g": st.session_state.get("nutrition_carb_g", 260),
        "fat_g": st.session_state.get("nutrition_fat_g", 70),
        "meal_count": st.session_state.get("nutrition_meal_count", 4),
        "exercise_type": st.session_state.get("exercise_type_input", ""),
        "exercise_duration": st.session_state.get("exercise_duration_input", 30),
        "exercise_intensity": st.session_state.get("exercise_intensity_input", ""),
        "exercise_burned": st.session_state.get("exercise_burned_input", 0),
    }


def render_ai_insight_card(title, body, cls="info-card"):
    st.markdown(f"""
    <div class="{cls}">
        <h3>{title}</h3>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def generate_deep_insights(records):
    a = advanced_texts()
    u = current_user_inputs()
    avg_sleep = records["sleep_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_productivity = records["productivity_score"].mean()
    protein_need = max(70, int(st.session_state.profile.get("weight_kg", 70) * 1.4))
    calorie_gap = u["calorie_goal"] - u["calories_taken"]
    macro_cal = int(u["protein_g"] * 4 + u["carb_g"] * 4 + u["fat_g"] * 9)
    net_cal = u["calories_taken"] - u["exercise_burned"]

    nutrition_msg = a["calories_good"]
    if calorie_gap > 350:
        nutrition_msg = a["not_enough_calories"]
    elif calorie_gap < -350:
        nutrition_msg = a["calories_high"]
    protein_msg = a["protein_good_detail"] if u["protein_g"] >= protein_need else a["protein_low_detail"]
    exercise_msg = a["exercise_good_detail"] if u["exercise_duration"] >= 20 else a["exercise_low_detail"]
    sleep_msg = a["sleep_good_detail"] if avg_sleep >= 6 else a["sleep_low_detail"]
    stress_msg = a["stress_ok_detail"] if avg_stress < 7 else a["stress_high_detail"]

    risk = "Low" if avg_stress < 5 and avg_sleep >= 6.5 and avg_productivity >= 50 else "Medium" if avg_stress < 7 else "High"
    return {
        "calorie_gap": calorie_gap,
        "macro_cal": macro_cal,
        "net_cal": net_cal,
        "protein_need": protein_need,
        "nutrition_msg": nutrition_msg,
        "protein_msg": protein_msg,
        "exercise_msg": exercise_msg,
        "sleep_msg": sleep_msg,
        "stress_msg": stress_msg,
        "risk": risk,
    }

def render_dashboard_tabs():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            t["daily_tracking"],
            t["nutrition"],
            t["exercise"],
            t["analytics"],
            t["coach"],
            t["report"],
            t["smart"],
        ]
    )

    with tab1:
        st.subheader(t["daily_title"])

        st.markdown(
            f"""
            <div class="info-card" style="padding:26px; margin-bottom:22px;">
                <h3 style="margin-top:0;">📝 {t['daily_title']}</h3>
                <p style="opacity:.88; margin-bottom:0;">
                    {t.get('daily_form_intro', 'Enter your daily data first. The AI analysis will be generated only after you press the button.')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("daily_tracking_form", clear_on_submit=False):
            st.markdown(f"### {t.get('daily_input_panel', 'Daily Input Panel')}")
            col1, col2 = st.columns(2)

            with col1:
                entry_date = st.date_input(t["date_label"], value=date.today(), key="daily_entry_date_form")
                mood_mode = st.radio(
                    t.get("daily_mode", "Daily Mode"),
                    [
                        t.get("mode_tired", "😴 Tired"),
                        t.get("mode_normal", "🙂 Normal"),
                        t.get("mode_motivated", "🔥 Motivated"),
                        t.get("mode_deep_work", "🚀 Deep Work"),
                    ],
                    horizontal=True,
                    key=f"daily_mode_{st.session_state.language}",
                )
                mood_text = st.text_area(
                    t["mood_question"],
                    value=t["default_mood"],
                    key=f"daily_mood_text_form_{st.session_state.language}",
                )
                sleep_hours = st.slider(t["sleep"], 0, 12, 7, key="daily_sleep_hours_form")
                study_hours = st.slider(t["study"], 0, 10, 4, key="daily_study_hours_form")
                task_completion = st.slider(t["task"], 0, 100, 65, key="daily_task_completion_form")

            with col2:
                focus_level = st.slider(t["focus"], 1, 10, 7, key="daily_focus_level_form")
                stress_level = st.slider(t["stress"], 1, 10, 4, key="daily_stress_level_form")
                exercise_minutes = st.slider(t["exercise_min"], 0, 120, 25, key="daily_exercise_minutes_form")
                water_liters = st.slider(t["water"], 0.0, 4.0, 2.0, key="daily_water_liters_form")
                nutrition_quality = st.slider(t["nutrition_quality"], 1, 10, 7, key="daily_nutrition_quality_form")

            st.markdown("---")
            submitted = st.form_submit_button(t["save_day"], use_container_width=True)

        if submitted:
            new_row = pd.DataFrame(
                [
                    {
                        "date": entry_date,
                        "mood_text": mood_text,
                        "sleep_hours": sleep_hours,
                        "study_hours": study_hours,
                        "focus_level": focus_level,
                        "stress_level": stress_level,
                        "exercise_minutes": exercise_minutes,
                        "task_completion": task_completion,
                        "water_liters": water_liters,
                        "nutrition_quality": nutrition_quality,
                        "steps": 7500,
                    }
                ]
            )

            st.session_state.records = (
                pd.concat([st.session_state.records, new_row], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
            )
            save_daily_record_to_db(new_row)

            sentiment = analyzer.polarity_scores(mood_text)["compound"]
            input_data = pd.DataFrame(
                [
                    {
                        "sleep_hours": sleep_hours,
                        "study_hours": study_hours,
                        "sentiment_score": sentiment,
                        "focus_level": focus_level,
                        "stress_level": stress_level,
                        "exercise_minutes": exercise_minutes,
                        "task_completion": task_completion,
                        "water_liters": water_liters,
                        "nutrition_quality": nutrition_quality,
                    }
                ]
            )

            prediction = model.predict(input_data)[0]
            productivity = labels[prediction]
            temp = calculate_scores(new_row)
            productivity_score = temp["productivity_score"].iloc[0]
            wellness_score = temp["wellness_score"].iloc[0]
            risk_score = temp["risk_score"].iloc[0]
            mood = t["positive"] if sentiment >= 0.05 else t["negative"] if sentiment <= -0.05 else t["neutral"]
            risk_level = t["high_risk"] if risk_score >= 70 else t["medium_risk"] if risk_score >= 45 else t["low_risk"]
            emoji, status_label = get_status_emoji(productivity_score, stress_level, sleep_hours)

            if productivity == "Low":
                advice = t["ai_low_advice"]
                plan = t["plan_low"]
                coach_style = "warning"
            elif productivity == "Medium":
                advice = t["ai_medium_advice"]
                plan = t["plan_medium"]
                coach_style = "info"
            else:
                advice = t["ai_high_advice"]
                plan = t["plan_high"]
                coach_style = "success"

            st.session_state.last_daily_analysis = {
                "mood": mood,
                "status": f"{emoji} {status_label}",
                "productivity_score": productivity_score,
                "wellness_score": wellness_score,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "advice": advice,
                "plan": plan,
                "coach_style": coach_style,
                "chart": pd.DataFrame(
                    {
                        t["chart_value"]: [
                            sleep_hours,
                            study_hours,
                            focus_level,
                            stress_level,
                            exercise_minutes / 10,
                            task_completion / 10,
                            water_liters,
                            nutrition_quality,
                            sentiment * 10,
                        ]
                    },
                    index=[
                        t["chart_sleep"], t["chart_study"], t["chart_focus"], t["chart_stress"], t["chart_exercise"],
                        t["chart_tasks"], t["chart_water"], t["chart_nutrition"], t["chart_sentiment"],
                    ],
                ),
            }

        if "last_daily_analysis" in st.session_state:
            result = st.session_state.last_daily_analysis
            st.markdown(f"### {t['ai_analysis']}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(t["mood_metric"], result["mood"])
            c2.metric(t["status_metric"], result["status"])
            c3.metric(t["productivity_score"], f"{result['productivity_score']}/100")
            c4.metric(t["wellness_score"], f"{result['wellness_score']}/100")
            c5.metric(t["risk_level"], result["risk_level"])

            if result["coach_style"] == "warning":
                st.warning(result["advice"])
            elif result["coach_style"] == "success":
                st.success(result["advice"])
            else:
                st.info(result["advice"])

            render_ai_insight_card(
                t.get("suggested_daily_plan", "Suggested Daily Plan"),
                result["plan"],
                "card-purple",
            )
            st.bar_chart(result["chart"])
        else:
            st.info(t.get("daily_waiting_message", "Fill the form and press the analysis button to generate today's AI report."))

    with tab2:
        x = section_extra_texts()
        st.subheader(t["nutrition"])
        profile = st.session_state.profile
        height_cm = profile["height_cm"]
        weight_kg = profile["weight_kg"]
        goal = profile["goal"]
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        bmi_status, bmi_advice = get_bmi_status(bmi)

        col1, col2, col3 = st.columns(3)
        col1.metric("BMI", bmi)
        col2.metric(t["bmi_status"], bmi_status)
        col3.metric(t["goal"], goal)
        st.info(bmi_advice)

        st.markdown(f"### {x['nutrition_input_title']}")
        n1, n2, n3 = st.columns(3)
        with n1:
            calorie_goal = st.number_input(x["calorie_goal"], min_value=1000, max_value=6000, value=2600, step=50, key="nutrition_calorie_goal")
            calories_taken = st.number_input(x["calories_taken"], min_value=0, max_value=7000, value=2100, step=50, key="nutrition_calories_taken")
        with n2:
            protein_g = st.number_input(x["protein_g"], min_value=0, max_value=350, value=110, step=5, key="nutrition_protein_g")
            carb_g = st.number_input(x["carb_g"], min_value=0, max_value=700, value=260, step=10, key="nutrition_carb_g")
        with n3:
            fat_g = st.number_input(x["fat_g"], min_value=0, max_value=250, value=70, step=5, key="nutrition_fat_g")
            meal_count = st.slider(x["meal_count"], 1, 8, 4, key="nutrition_meal_count")

        macro_calories = int((protein_g * 4) + (carb_g * 4) + (fat_g * 9))
        remaining = calorie_goal - calories_taken
        c1, c2, c3 = st.columns(3)
        c1.metric(x["macro_calories"], macro_calories)
        c2.metric(x["remaining_calories"], remaining)
        c3.metric(x["meal_count"], meal_count)

        if protein_g < max(70, int(weight_kg * 1.2)):
            st.warning(x["protein_low"])
        elif calories_taken < calorie_goal - 350:
            st.info(x["calorie_low"])
        elif calories_taken > calorie_goal + 350:
            st.warning(x["calorie_high"])
        else:
            st.success(x["nutrition_good"])

        st.markdown(
            f"""
            <div class="card-green">
            <h3>{x['nutrition_note']}</h3>
            <p>{x['nutrition_detail_note']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        food_data = pd.DataFrame(
            {
                t["food_category"]: [t["protein"], t["carbohydrate"], t["healthy_fat"], t["fruit"], t["vegetable"], t["hydration"]],
                t["food_examples"]: [
                    t["protein_examples"], t["carb_examples"], t["fat_examples"],
                    t["fruit_examples"], t["veg_examples"], t["hydration_examples"],
                ],
                t["food_purpose"]: [
                    t["protein_purpose"], t["carb_purpose"], t["fat_purpose"],
                    t["fruit_purpose"], t["veg_purpose"], t["hydration_purpose"],
                ],
            }
        )
        st.dataframe(food_data, use_container_width=True)

    with tab3:
        x = section_extra_texts()
        st.subheader(t["exercise"])
        fitness_goal = st.selectbox(
            t["exercise_goal"],
            [t["general_health"], t["weight_gain_muscle"], t["fat_loss"], t["stress_reduction"], t["posture_mobility"]],
            key=f"exercise_goal_{st.session_state.language}",
        )
        plans = {
            t["general_health"]: t["plan_general"],
            t["weight_gain_muscle"]: t["plan_muscle"],
            t["fat_loss"]: t["plan_fat"],
            t["stress_reduction"]: t["plan_stress"],
            t["posture_mobility"]: t["plan_posture"],
        }
        st.markdown(
            f"""
            <div class="card-green">
            <h3>{fitness_goal}</h3>
            <p>{t["exercise_plan_text"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in plans[fitness_goal]:
            st.write("•", item)

        st.markdown(f"### {x['exercise_input_title']}")
        e1, e2, e3 = st.columns(3)
        with e1:
            exercise_type = st.selectbox(x["exercise_type"], [x["walking"], x["running"], x["strength"], x["cycling"], x["mobility"]], key="exercise_type_input")
            duration = st.slider(x["duration_min"], 0, 180, 30, key="exercise_duration_input")
        with e2:
            intensity = st.selectbox(x["intensity"], [x["low"], x["medium"], x["high"]], index=1, key="exercise_intensity_input")
            body_weight = st.number_input(x["body_weight"], min_value=35, max_value=180, value=int(st.session_state.profile.get("weight_kg", 70)))
        with e3:
            base_met = {x["walking"]: 3.5, x["running"]: 8.5, x["strength"]: 5.0, x["cycling"]: 6.8, x["mobility"]: 2.5}.get(exercise_type, 4.0)
            multiplier = {x["low"]: 0.8, x["medium"]: 1.0, x["high"]: 1.25}.get(intensity, 1.0)
            burned = int(base_met * multiplier * 3.5 * body_weight / 200 * duration)
            st.session_state.exercise_burned_input = burned
            st.metric(x["estimated_burn"], burned)

        if duration >= 20:
            st.success(x["exercise_good"])
        else:
            st.warning(x["exercise_low"])

        exercise_log = pd.DataFrame({
            x["exercise_type"]: [exercise_type],
            x["duration_min"]: [duration],
            x["intensity"]: [intensity],
            x["estimated_burn"]: [burned],
        })
        st.dataframe(exercise_log, use_container_width=True)

    with tab4:
        a = advanced_texts()
        records = calculate_scores(st.session_state.records)
        records["date"] = pd.to_datetime(records["date"])
        dashboard = records.sort_values("date").set_index("date")
        insights = generate_deep_insights(records)
        u = current_user_inputs()

        st.subheader(a["advanced_analysis"])
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(t["avg_productivity"], f"{round(records['productivity_score'].mean(), 1)}/100")
        k2.metric(t["avg_wellness"], f"{round(records['wellness_score'].mean(), 1)}/100")
        k3.metric(a["net_calories"], insights["net_cal"])
        k4.metric(a["burned_calories"], u["exercise_burned"])
        k5.metric(a["protein_target"], f"{u['protein_g']}/{insights['protein_need']}g")

        ncol, ecol = st.columns(2)
        with ncol:
            render_ai_insight_card(a["nutrition_dashboard"], f"{insights['nutrition_msg']} {insights['protein_msg']}", "card-green")
            macro_df = pd.DataFrame(
                {a["macro_split"]: [u["protein_g"], u["carb_g"], u["fat_g"]]},
                index=[t.get("protein", "Protein"), t.get("carbohydrate", "Carbohydrate"), t.get("healthy_fat", "Fat")],
            )
            st.bar_chart(macro_df)
            st.caption(f"{a['calorie_balance']}: {insights['calorie_gap']} | {a['macro_split']}: {insights['macro_cal']} kcal")
        with ecol:
            render_ai_insight_card(a["exercise_dashboard"], insights["exercise_msg"], "card-orange")
            exercise_df = pd.DataFrame(
                {a["exercise_efficiency"]: [u["exercise_duration"], u["exercise_burned"], records["exercise_minutes"].mean()]},
                index=[t.get("exercise_min", "Exercise Minutes"), a["burned_calories"], t.get("average_exercise", "Average Exercise")],
            )
            st.bar_chart(exercise_df)
            st.caption(f"{a['exercise_efficiency']}: {u['exercise_duration']} min / {u['exercise_burned']} kcal")

        st.markdown(f"### {a['chart_commentary']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_ai_insight_card(t["sleep_trend"], a["graph_sleep"] + " " + insights["sleep_msg"], "info-card")
        with c2:
            render_ai_insight_card(t["study_trend"], a["graph_study"], "info-card")
        with c3:
            render_ai_insight_card(t["wellness_trend"], a["graph_wellness"], "info-card")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t["sleep_trend"])
            st.line_chart(dashboard[["sleep_hours", "wellness_score"]])
            st.subheader(t["stress_trend"])
            st.line_chart(dashboard[["stress_level", "risk_score"]])
        with col2:
            st.subheader(t["productivity_trend"])
            st.line_chart(dashboard[["study_hours", "productivity_score", "task_completion"]])
            st.subheader(t["nutrition_trend"])
            st.line_chart(dashboard[["nutrition_quality", "water_liters", "exercise_minutes"]])

        st.markdown(f"### {a['risk_interpretation']}")
        if insights["risk"] == "High":
            st.error(insights["stress_msg"])
        elif insights["risk"] == "Medium":
            st.warning(insights["sleep_msg"] + " " + insights["stress_msg"])
        else:
            st.success(insights["sleep_msg"] + " " + insights["stress_msg"])

        with st.expander(t["complete_tracking_data"]):
            st.dataframe(records, use_container_width=True)

    with tab5:
        a = advanced_texts()
        records = calculate_scores(st.session_state.records)
        insights = generate_deep_insights(records)
        u = current_user_inputs()
        st.subheader(a["coach_title"])

        avg_productivity = records["productivity_score"].mean()
        avg_stress = records["stress_level"].mean()
        avg_sleep = records["sleep_hours"].mean()

        top1, top2, top3 = st.columns(3)
        top1.metric(t["avg_productivity"], f"{avg_productivity:.1f}/100")
        top2.metric(t["avg_sleep"], f"{avg_sleep:.1f}h")
        top3.metric(t["avg_stress"], f"{avg_stress:.1f}/10")

        render_ai_insight_card(a["coach_strategy"], a["mission"], "card-purple")

        p1, p2 = st.columns(2)
        with p1:
            render_ai_insight_card(a["nutrition_strategy"], f"{insights['nutrition_msg']} {insights['protein_msg']} Net: {insights['net_cal']} kcal.", "card-green")
            render_ai_insight_card(a["study_strategy"], get_motivation_quote(avg_productivity, avg_stress, avg_sleep), "card-blue")
        with p2:
            render_ai_insight_card(a["exercise_strategy"], f"{insights['exercise_msg']} {a['burned_calories']}: {u['exercise_burned']} kcal.", "card-orange")
            render_ai_insight_card(a["recovery_strategy"], f"{insights['sleep_msg']} {insights['stress_msg']}", "info-card")

        st.markdown(f"### {a['priority_plan']}")
        plan_items = [
            insights["nutrition_msg"],
            insights["exercise_msg"],
            insights["sleep_msg"],
            insights["stress_msg"],
        ]
        for i, item in enumerate(plan_items, start=1):
            st.checkbox(f"{i}. {item}", key=f"coach_plan_{i}_{st.session_state.language}")

        st.markdown("### AI Signals")
        for level, msg in get_ai_recommendations(records):
            if level == "success":
                st.success(msg)
            elif level == "error":
                st.error(msg)
            elif level == "info":
                st.info(msg)
            else:
                st.warning(msg)

        with st.expander(t["model_feature_importance"]):
            importance_data = pd.DataFrame({t["importance"]: model.feature_importances_}, index=features).sort_values(t["importance"], ascending=False)
            st.bar_chart(importance_data)
            st.metric(t["training_accuracy"], f"{round(accuracy * 100, 2)}%")

        render_student_bottom_summary()

    with tab6:
        a = advanced_texts()
        records = calculate_scores(st.session_state.records)
        insights = generate_deep_insights(records)
        u = current_user_inputs()
        st.subheader(a["report_title"])

        st.markdown(
            f"""
            <div class="card-blue">
            <h3>{a['executive_summary']}</h3>
            <p>{a['report_intro']} {create_status_note(records, st.session_state.profile)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        weekly_summary = pd.DataFrame(
            {
                t["metric"]: [
                    t["average_sleep"], t["average_study"], t["average_focus"], t["average_stress"],
                    t["average_exercise"], t["average_nutrition"], t["average_water"],
                    t["average_task"], t["average_productivity"], t["average_wellness"],
                    a["calorie_balance"], a["net_calories"], a["burned_calories"], a["protein_target"],
                ],
                t["chart_value"]: [
                    round(records["sleep_hours"].mean(), 2),
                    round(records["study_hours"].mean(), 2),
                    round(records["focus_level"].mean(), 2),
                    round(records["stress_level"].mean(), 2),
                    round(records["exercise_minutes"].mean(), 2),
                    round(records["nutrition_quality"].mean(), 2),
                    round(records["water_liters"].mean(), 2),
                    round(records["task_completion"].mean(), 2),
                    round(records["productivity_score"].mean(), 2),
                    round(records["wellness_score"].mean(), 2),
                    insights["calorie_gap"], insights["net_cal"], u["exercise_burned"], f"{u['protein_g']}/{insights['protein_need']}g",
                ],
            }
        )
        st.dataframe(weekly_summary, use_container_width=True)

        r1, r2 = st.columns(2)
        with r1:
            render_ai_insight_card(a["weekly_decision"], f"{insights['nutrition_msg']} {insights['exercise_msg']}", "card-orange")
        with r2:
            render_ai_insight_card(a["next_actions"], f"1) {insights['protein_msg']} 2) {insights['sleep_msg']} 3) {insights['stress_msg']}", "card-green")

        recommendations = get_ai_recommendations(records)
        status_note = create_status_note(records, st.session_state.profile)
        csv = weekly_summary.to_csv(index=False).encode("utf-8")
        st.download_button(t["download_csv"], csv, "weekly_student_report.csv", "text/csv")

        if PDF_AVAILABLE:
            pdf_file = create_pdf_report(st.session_state.profile, weekly_summary, recommendations, status_note)
            st.download_button(t["download_pdf"], pdf_file, "weekly_student_report.pdf", "application/pdf")
        else:
            st.warning(t["pdf_not_active"])

    with tab7:
        st.subheader(t["smart"])
        st.write(t["smart_intro"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="card-blue">
                <h3>{t["smartwatch_sleep_tracking"]}</h3>
                <p>{t["smart_sleep_text"]}</p>
                <ul>
                    <li>{t["deep_sleep_duration"]}</li>
                    <li>{t["rem_sleep"]}</li>
                    <li>{t["sleep_quality_score"]}</li>
                    <li>{t["recovery_score"]}</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            sleep_data = pd.DataFrame(
                {
                    t["sleep_quality"]: [62, 68, 55, 74, 70, 82, 78],
                    t["deep_sleep"]: [1.2, 1.5, 1.0, 1.8, 1.6, 2.1, 1.9],
                    t["recovery"]: [50, 58, 45, 70, 68, 85, 80],
                },
                index=[t["mon"], t["tue"], t["wed"], t["thu"], t["fri"], t["sat"], t["sun"]],
            )
            st.line_chart(sleep_data)

        with col2:
            st.markdown(
                f"""
                <div class="card-green">
                <h3>{t["exercise_tracking"]}</h3>
                <p>{t["exercise_tracking_text"]}</p>
                <ul>
                    <li>{t["step_count"]}</li>
                    <li>{t["training_duration"]}</li>
                    <li>{t["activity_intensity"]}</li>
                    <li>{t["energy_score"]}</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            exercise_data = pd.DataFrame(
                {
                    t["exercise_minutes_label"]: [0, 15, 10, 25, 20, 40, 35],
                    t["energy_score"]: [45, 58, 55, 70, 68, 85, 80],
                    t["steps_1000"]: [2, 4, 3, 6, 7, 10, 9],
                },
                index=[t["mon"], t["tue"], t["wed"], t["thu"], t["fri"], t["sat"], t["sun"]],
            )
            st.bar_chart(exercise_data)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                f"""
                <div class="card-purple">
                <h3>{t["weekly_reports"]}</h3>
                <p>{t["weekly_reports_text"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="card-orange">
                <h3>{t["long_term_behavior_analysis"]}</h3>
                <p>{t["long_term_behavior_text"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )




# === V14: Clean one-row tabs + remove extra separator ===
st.markdown("""
<style>
/* Keep all section icons in one clean row */
div[data-testid="stTabs"] {
    margin-top: 14px !important;
    margin-bottom: 12px !important;
}

div[data-testid="stTabs"] div[role="tablist"] {
    width: 100% !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
    padding: 14px 6px 14px 6px !important;
    margin: 18px 0 12px 0 !important;
    overflow: hidden !important;
    border-top: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.24) !important;
}

div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {
    flex: 1 1 0 !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: 210px !important;
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    padding: 0 10px !important;
    border-radius: 16px !important;
    box-sizing: border-box !important;
}

div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] div,
div[data-testid="stTabs"] [role="tab"] div,
div[data-testid="stTabs"] button[role="tab"] span,
div[data-testid="stTabs"] [role="tab"] span {
    font-size: 14px !important;
    font-weight: 620 !important;
    letter-spacing: -0.15px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Delete the second unnecessary line below the tab bar */
div[data-testid="stTabs"] hr,
div[data-testid="stTabs"] div[data-testid="stMarkdownContainer"] hr,
div[data-testid="stTabs"] div[data-testid="stVerticalBlock"] > div:has(hr) {
    display: none !important;
}

/* Pull the first tab content slightly closer to the tab bar */
.welcome-card {
    margin-top: 6px !important;
}
.welcome-detail-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 18px;
}
.welcome-mini-card {
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(15,23,42,0.46);
    border: 1px solid rgba(148,163,184,0.18);
}
.welcome-mini-card span {
    display: block;
    font-size: 12px;
    color: #cbd5e1;
    margin-bottom: 5px;
}
.welcome-mini-card b {
    font-size: 15px;
    font-weight: 720;
}
.welcome-next-step {
    margin-top: 14px;
    padding: 13px 15px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(251,146,60,0.18), rgba(37,99,235,0.12));
    border: 1px solid rgba(251,146,60,0.18);
}
.welcome-next-step b {
    display: block;
    margin-bottom: 4px;
}
@media (max-width: 900px) {
    .welcome-detail-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# Page routing
# Header + concept card appear only on the main dashboard page.
# Login/Profile, Settings, and Database pages stay clean and separate.
if st.session_state.sidebar_page == "home":
    render_header()
    render_dashboard_tabs()
elif st.session_state.sidebar_page == "login":
    st.markdown('<div class="login-profile-wrap">', unsafe_allow_html=True)
    render_login_profile()
    st.markdown('</div>', unsafe_allow_html=True)
elif st.session_state.sidebar_page == "settings":
    render_settings()
elif st.session_state.sidebar_page == "database":
    render_database_history()
