
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
        "daily_tracking": "📅 Günlük Takip",
        "nutrition": "🥗 Beslenme ve Sağlık",
        "exercise": "🎯 Tatbikat Planı",
        "analytics": "📊 Analitikler",
        "coach": "🤖 Yapay Zeka Koçu",
        "report": "📄 Haftalık Rapor",
        "smart": "⌚ Akıllı Entegr.",
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
        "daily_tracking": "📅 Daily Tracking",
        "nutrition": "🥗 Nutrition & Health",
        "exercise": "🎯 Exercise Plan",
        "analytics": "📊 Analytics",
        "coach": "🤖 AI Coach",
        "report": "📄 Weekly Report",
        "smart": "⌚ Smart Integr.",
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
        "daily_tracking": "📅 Tagestracking",
        "nutrition": "🥗 Ernährung & Gesundheit",
        "exercise": "🎯 Trainingsplan",
        "analytics": "📊 Analysen",
        "coach": "🤖 KI-Coach",
        "report": "📄 Wochenbericht",
        "smart": "⌚ Smart-Integr.",
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
        "daily_tracking": "📅 Ежедневный трекинг",
        "nutrition": "🥗 Питание и здоровье",
        "exercise": "🎯 План тренировок",
        "analytics": "📊 Аналитика",
        "coach": "🤖 ИИ-коуч",
        "report": "📄 Еженедельный отчет",
        "smart": "⌚ Smart-интегр.",
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
        "daily_tracking": "📅 Seguimiento diario",
        "nutrition": "🥗 Nutrición y salud",
        "exercise": "🎯 Plan de ejercicio",
        "analytics": "📊 Analíticas",
        "coach": "🤖 Coach IA",
        "report": "📄 Informe semanal",
        "smart": "⌚ Integr. inteligentes",
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


if "language" not in st.session_state:
    st.session_state.language = "Türkçe"
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Sunset Orange"
if "background_mode" not in st.session_state:
    st.session_state.background_mode = "Yumuşak Çoklu Renk"
if "sidebar_page" not in st.session_state:
    st.session_state.sidebar_page = "login"


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

    page_labels = {
        "login": t["login"],
        "settings": t["settings"],
        "database": t["db"],
    }

    selected_label = st.radio(
        "Navigation",
        [page_labels["login"], page_labels["settings"], page_labels["database"]],
        index=["login", "settings", "database"].index(st.session_state.sidebar_page),
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
        st.rerun()

    t = get_text()

    st.markdown(
        f"""
        <div class="language-card">
            <b>{t["active_language"]}:</b><br>{st.session_state.language}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

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
    padding-top: 3rem;
    max-width: 1500px;
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
    font-weight: 900 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 13px !important;
    line-height: 1 !important;
    margin: -4px 0 0 0 !important;
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



/* === FINAL EXACT LAYOUT PATCH === */

/* Keep tab menu clean: one line only, no duplicate top/middle divider */
div[data-testid="stTabs"] div[role="tablist"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 14px !important;

    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;

    padding: 16px 0 18px 0 !important;
    margin: 0 !important;

    border-top: none !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.22) !important;
    scroll-behavior: smooth !important;
}}

/* Hide ugly scrollbar on tab row */
div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar {{
    height: 0px !important;
}}

/* All top buttons same height and professional width */
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {{
    height: 56px !important;
    min-height: 56px !important;
    max-height: 56px !important;

    width: 190px !important;
    min-width: 190px !important;
    max-width: 190px !important;

    padding: 0 16px !important;
    margin: 0 !important;

    border-radius: 17px !important;
    box-sizing: border-box !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    overflow: hidden !important;
    line-height: 1 !important;
}}

/* Tab text and emoji: centered, slightly up, no slipping */
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] [role="tab"] p {{
    margin: -2px 0 0 0 !important;
    padding: 0 !important;

    line-height: 1 !important;
    height: 18px !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    font-size: 13px !important;
    font-weight: 900 !important;

    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}

/* Prevent internal Streamlit wrappers from moving the emoji down */
div[data-testid="stTabs"] button[role="tab"] div,
div[data-testid="stTabs"] [role="tab"] div,
div[data-testid="stTabs"] button[role="tab"] span,
div[data-testid="stTabs"] [role="tab"] span {{
    margin: 0 !important;
    padding: 0 !important;

    line-height: 1 !important;
    height: auto !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

/* Remove BaseWeb tab underline/highlight so our buttons stay clean */
div[data-baseweb="tab-highlight"],
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{
    background: transparent !important;
    height: 0 !important;
}}

/* Sidebar: menu buttons same full width as language select and active language card */
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
    gap: 16px !important;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;

    height: 66px !important;
    min-height: 66px !important;

    display: flex !important;
    align-items: center !important;

    padding: 0 22px !important;
    margin: 0 !important;

    border-radius: 18px !important;
    box-sizing: border-box !important;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    margin: 0 !important;
    padding: 0 !important;

    line-height: 1 !important;
    font-size: 16px !important;
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

/* Make the select and sidebar cards visually aligned with radio buttons */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    min-height: 58px !important;
    border-radius: 16px !important;
}}

section[data-testid="stSidebar"] .language-card {{
    border-radius: 18px !important;
}}

</style>
"""
    css = css.format(background_style=background_css(bg_mode), **theme_dict)
    st.markdown(css, unsafe_allow_html=True)


apply_css(theme, st.session_state.background_mode)


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
        return "Underweight", "Increase healthy calories, protein intake, and strength training."
    if bmi < 25:
        return "Normal", "Maintain balanced nutrition and consistent exercise."
    if bmi < 30:
        return "Overweight", "Focus on calorie control, walking, and regular training."
    return "High Range", "A structured plan and professional guidance are recommended."


def get_status_emoji(avg_productivity, avg_stress, avg_sleep):
    if avg_productivity >= 70 and avg_stress <= 5 and avg_sleep >= 7:
        return "🚀", "Excellent"
    if avg_productivity >= 55 and avg_stress <= 6:
        return "🙂", "Stable"
    if avg_stress >= 7 or avg_sleep < 6:
        return "⚠️", "Needs Attention"
    return "🌱", "Improving"


def get_motivation_quote(avg_productivity, avg_stress, avg_sleep):
    if avg_productivity >= 70:
        return "Discipline turns good days into progress and bad days into lessons."
    if avg_stress >= 7:
        return "Slow down, reset, and continue. Sustainable progress beats burnout."
    if avg_sleep < 6:
        return "Better sleep is not wasted time; it is fuel for better performance."
    return "Small consistent steps create strong long-term results."


def create_status_note(records, profile):
    if records.empty:
        return "No tracking data is available yet. Please save daily records first."

    avg_sleep = records["sleep_hours"].mean()
    avg_study = records["study_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_productivity = records["productivity_score"].mean()
    avg_wellness = records["wellness_score"].mean()
    avg_exercise = records["exercise_minutes"].mean()
    avg_water = records["water_liters"].mean()
    student_name = profile.get("name", "Student") or "Student"

    productivity_text = "your productivity level is strong" if avg_productivity >= 70 else "your productivity level is moderate" if avg_productivity >= 50 else "your productivity level needs improvement"
    stress_text = "your stress level is high, so recovery time should be increased" if avg_stress >= 7 else "your stress level is manageable but should still be watched" if avg_stress >= 5 else "your stress level is currently under control"
    sleep_text = "your sleep duration is low and may reduce focus" if avg_sleep < 6 else "your sleep duration looks acceptable"
    exercise_text = "your exercise activity is low, so light walking or mobility work is recommended" if avg_exercise < 20 else "your exercise activity supports your wellness"
    water_text = "your hydration is low, so drink more water during the day" if avg_water < 1.8 else "your hydration looks acceptable"

    return (
        f"{student_name}, based on your saved tracking data, {productivity_text}. "
        f"Your average sleep is {avg_sleep:.1f} hours, your average study time is {avg_study:.1f} hours, "
        f"and your average wellness score is {avg_wellness:.1f}/100. "
        f"Currently, {sleep_text}; {stress_text}; {exercise_text}; and {water_text}. "
        f"My recommendation is to keep a stable daily routine, avoid overload, and track your progress consistently."
    )


def get_ai_recommendations(records):
    avg_sleep = records["sleep_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_productivity = records["productivity_score"].mean()
    avg_nutrition = records["nutrition_quality"].mean()
    avg_exercise = records["exercise_minutes"].mean()
    avg_water = records["water_liters"].mean()

    messages = []
    messages.append(("warning", "Sleep is low. Try to increase sleep by at least 1 hour.") if avg_sleep < 6 else ("success", "Sleep duration is acceptable."))
    messages.append(("error", "Stress trend is high. Add recovery time and reduce workload intensity.") if avg_stress > 6 else ("success", "Stress level is manageable."))
    messages.append(("warning", "Nutrition quality is low. Add protein, fruit, vegetables and stable meals.") if avg_nutrition < 6 else ("success", "Nutrition quality is acceptable."))
    messages.append(("warning", "Exercise activity is low. Add walking or mobility sessions.") if avg_exercise < 20 else ("success", "Exercise supports wellness and focus."))
    messages.append(("warning", "Water intake is low. Increase hydration for better focus.") if avg_water < 1.8 else ("success", "Hydration level is acceptable."))

    if avg_productivity >= 70:
        messages.append(("success", "Productivity trend is strong. Deep work sessions can be increased."))
    elif avg_productivity >= 50:
        messages.append(("info", "Productivity trend is moderate. Use a more structured weekly plan."))
    else:
        messages.append(("warning", "Productivity needs improvement. Start with smaller tasks and shorter study blocks."))

    return messages


def create_pdf_report(profile, weekly_summary, recommendations, status_note):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "AI Student Weekly Performance Report")
    y -= 35
    pdf.setFont("Helvetica", 11)

    for line in [
        f"Student: {profile.get('name', 'Student')}",
        f"Student ID: {profile.get('student_id', '0000')}",
        f"Faculty: {profile.get('faculty', 'Software Engineering')}",
        f"Semester: {profile.get('semester', 2)}",
        f"Goal: {profile.get('goal', 'Improve Productivity')}",
    ]:
        pdf.drawString(50, y, line)
        y -= 18

    y -= 20
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Student Status Summary")
    y -= 22
    pdf.setFont("Helvetica", 10)

    for i in range(0, len(status_note), 90):
        pdf.drawString(60, y, status_note[i:i + 90])
        y -= 16

    y -= 15
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Weekly Metrics")
    y -= 25
    pdf.setFont("Helvetica", 11)

    for _, row in weekly_summary.iterrows():
        pdf.drawString(60, y, f"{row['Metric']}: {row['Value']}")
        y -= 18

    y -= 20
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "AI Coach Recommendations")
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


def render_login_profile():
    st.subheader(t["profile_title"])

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input(t["full_name"], st.session_state.profile["name"])
        student_id = st.text_input(t["student_id"], st.session_state.profile["student_id"])

    with col2:
        faculty = st.selectbox(
            t["faculty"],
            ["Yazılım Mühendisliği", "Computer Science", "AI Engineering", "Business", "Design"],
            index=0,
        )
        semester = st.slider(t["semester"], 1, 8, st.session_state.profile["semester"])

    with col3:
        age = st.number_input(t["age"], min_value=10, max_value=80, value=st.session_state.profile["age"])
        goal = st.selectbox(
            t["main_goal"],
            ["Verimliliği Artır", "Kilo Al", "Yağ Kaybet", "Stresi Azalt", "Uykuyu İyileştir", "Disiplin Kur"],
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
    m2.metric("BMI Status", bmi_status)
    m3.metric(t["semester"], semester)
    m4.metric(t["main_goal"], goal)

    st.info(bmi_advice)



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
        st.success(f"{len(db_records)} records loaded from SQLite database.")
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
            <h3>Current Student Status</h3>
            <p>{status_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


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

        col1, col2 = st.columns(2)

        with col1:
            entry_date = st.date_input("Date", value=date.today(), key="daily_entry_date")
            mood_text = st.text_area(t["mood_question"], value="I feel focused and ready to study.", key="daily_mood_text")
            sleep_hours = st.slider(t["sleep"], 0, 12, 7, key="daily_sleep_hours")
            study_hours = st.slider(t["study"], 0, 10, 4, key="daily_study_hours")
            task_completion = st.slider(t["task"], 0, 100, 65, key="daily_task_completion")

        with col2:
            focus_level = st.slider(t["focus"], 1, 10, 7, key="daily_focus_level")
            stress_level = st.slider(t["stress"], 1, 10, 4, key="daily_stress_level")
            exercise_minutes = st.slider(t["exercise_min"], 0, 120, 25, key="daily_exercise_minutes")
            water_liters = st.slider(t["water"], 0.0, 4.0, 2.0, key="daily_water_liters")
            nutrition_quality = st.slider(t["nutrition_quality"], 1, 10, 7, key="daily_nutrition_quality")

        if st.button(t["save_day"], key="save_daily_record_button"):
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
            mood = "Positive" if sentiment >= 0.05 else "Negative" if sentiment <= -0.05 else "Neutral"
            risk_level = "High Risk" if risk_score >= 70 else "Medium Risk" if risk_score >= 45 else "Low Risk"
            emoji, status_label = get_status_emoji(productivity_score, stress_level, sleep_hours)

            st.divider()
            st.subheader("AI Analysis")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Mood", mood)
            c2.metric("Status", f"{emoji} {status_label}")
            c3.metric("Productivity Score", f"{productivity_score}/100")
            c4.metric("Wellness Score", f"{wellness_score}/100")
            c5.metric("Risk Level", risk_level)

            if productivity == "Low":
                st.warning("AI Coach: Reduce workload, drink water, eat a healthy meal, and complete one small task.")
                plan = "25 min light study -> 5 min break -> 20 min review"
            elif productivity == "Medium":
                st.info("AI Coach: Continue with moderate tasks, avoid multitasking, and keep a stable rhythm.")
                plan = "45 min study -> 10 min break -> 45 min practice"
            else:
                st.success("AI Coach: Strong productivity state. Start with difficult tasks and use deep work.")
                plan = "60 min deep work -> 10 min break -> 60 min focused study"

            st.write(f"Suggested Daily Plan: {plan}")

            today_chart = pd.DataFrame(
                {
                    "Value": [
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
                    "Sleep", "Study", "Focus", "Stress", "Exercise / 10",
                    "Tasks / 10", "Water", "Nutrition", "Sentiment x10",
                ],
            )
            st.bar_chart(today_chart)

    with tab2:
        st.subheader(t["nutrition"])
        profile = st.session_state.profile
        height_cm = profile["height_cm"]
        weight_kg = profile["weight_kg"]
        goal = profile["goal"]
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        bmi_status, bmi_advice = get_bmi_status(bmi)

        col1, col2, col3 = st.columns(3)
        col1.metric("BMI", bmi)
        col2.metric("Status", bmi_status)
        col3.metric("Goal", goal)
        st.info(bmi_advice)

        food_data = pd.DataFrame(
            {
                "Category": ["Protein", "Carbohydrate", "Healthy Fat", "Fruit", "Vegetable", "Hydration"],
                "Examples": [
                    "Eggs, chicken, fish, yogurt, lentils",
                    "Rice, oats, potatoes, whole grain bread",
                    "Olive oil, avocado, nuts, peanut butter",
                    "Banana, apple, berries, orange",
                    "Broccoli, spinach, salad, carrots",
                    "Water, mineral water, unsweetened tea",
                ],
                "Purpose": [
                    "Muscle repair and satiety",
                    "Energy for studying and training",
                    "Hormonal health and long-term energy",
                    "Vitamins and quick energy",
                    "Micronutrients and digestion",
                    "Focus, recovery, and mood stability",
                ],
            }
        )
        st.dataframe(food_data, use_container_width=True)

    with tab3:
        st.subheader(t["exercise"])
        fitness_goal = st.selectbox(
            "Exercise Goal",
            ["General Health", "Weight Gain / Muscle", "Fat Loss", "Stress Reduction", "Posture & Mobility"],
        )
        plans = {
            "General Health": [
                "Monday: 30 min walking + 10 min stretching",
                "Wednesday: Full body bodyweight training",
                "Friday: 30 min cycling or walking",
                "Sunday: Light mobility and recovery",
            ],
            "Weight Gain / Muscle": [
                "Monday: Push training",
                "Tuesday: Pull training",
                "Thursday: Legs",
                "Saturday: Full body strength training",
            ],
            "Fat Loss": [
                "Monday: 40 min brisk walking",
                "Wednesday: Full body circuit",
                "Friday: Interval cardio",
                "Sunday: Long walk",
            ],
            "Stress Reduction": [
                "Monday: 20 min walk + breathing",
                "Wednesday: Mobility session",
                "Friday: Light cardio",
                "Sunday: Stretching and recovery",
            ],
            "Posture & Mobility": [
                "Daily: 5 min neck mobility",
                "Daily: 5 min shoulder mobility",
                "3x/week: Core stability",
                "3x/week: Back strengthening",
            ],
        }
        st.markdown(
            f"""
            <div class="card-green">
            <h3>{fitness_goal}</h3>
            <p>This exercise plan supports physical health, focus, recovery and student productivity.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in plans[fitness_goal]:
            st.write("•", item)

    with tab4:
        records = calculate_scores(st.session_state.records)
        records["date"] = pd.to_datetime(records["date"])
        dashboard = records.sort_values("date").set_index("date")

        st.subheader(t["analytics"])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Avg Sleep", round(records["sleep_hours"].mean(), 2))
        m2.metric("Avg Study", round(records["study_hours"].mean(), 2))
        m3.metric("Avg Stress", round(records["stress_level"].mean(), 2))
        m4.metric("Avg Nutrition", round(records["nutrition_quality"].mean(), 2))
        m5.metric("Avg Productivity", round(records["productivity_score"].mean(), 2))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sleep Trend")
            st.line_chart(dashboard["sleep_hours"])
            st.subheader("Stress Trend")
            st.line_chart(dashboard["stress_level"])
            st.subheader("Nutrition Quality Trend")
            st.line_chart(dashboard["nutrition_quality"])
        with col2:
            st.subheader("Study Trend")
            st.line_chart(dashboard["study_hours"])
            st.subheader("Productivity Trend")
            st.line_chart(dashboard["productivity_score"])
            st.subheader("Wellness Trend")
            st.line_chart(dashboard["wellness_score"])

        st.subheader("Complete Tracking Data")
        st.dataframe(records, use_container_width=True)

    with tab5:
        records = calculate_scores(st.session_state.records)
        st.subheader(t["coach"])

        avg_productivity = records["productivity_score"].mean()
        avg_stress = records["stress_level"].mean()
        avg_sleep = records["sleep_hours"].mean()
        emoji, status_label = get_status_emoji(avg_productivity, avg_stress, avg_sleep)
        quote = get_motivation_quote(avg_productivity, avg_stress, avg_sleep)
        status_note = create_status_note(records, st.session_state.profile)

        st.markdown(
            f"""
            <div class="status-card">
            <h3>{emoji} Current Student Status: {status_label}</h3>
            <p>{status_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="quote-card">
            <h3>✨ Daily Motivation Quote</h3>
            <p><i>"{quote}"</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for level, msg in get_ai_recommendations(records):
            if level == "success":
                st.success(msg)
            elif level == "error":
                st.error(msg)
            elif level == "info":
                st.info(msg)
            else:
                st.warning(msg)

        st.subheader("Model Feature Importance")
        importance_data = pd.DataFrame({"Importance": model.feature_importances_}, index=features).sort_values("Importance", ascending=False)
        st.bar_chart(importance_data)
        st.metric("Training Accuracy", f"{round(accuracy * 100, 2)}%")

    with tab6:
        records = calculate_scores(st.session_state.records)
        st.subheader(t["report"])

        weekly_summary = pd.DataFrame(
            {
                "Metric": [
                    "Average Sleep", "Average Study", "Average Focus", "Average Stress",
                    "Average Exercise", "Average Nutrition", "Average Water Intake",
                    "Average Task Completion", "Average Productivity", "Average Wellness",
                ],
                "Value": [
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
                ],
            }
        )

        st.dataframe(weekly_summary, use_container_width=True)
        recommendations = get_ai_recommendations(records)
        status_note = create_status_note(records, st.session_state.profile)

        st.markdown(
            f"""
            <div class="card-blue">
            <h3>Student Weekly Status Note</h3>
            <p>{status_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        csv = weekly_summary.to_csv(index=False).encode("utf-8")
        st.download_button("Download Weekly Report as CSV", csv, "weekly_student_report.csv", "text/csv")

        if PDF_AVAILABLE:
            pdf_file = create_pdf_report(st.session_state.profile, weekly_summary, recommendations, status_note)
            st.download_button("Download Weekly Report as PDF", pdf_file, "weekly_student_report.pdf", "application/pdf")
        else:
            st.warning("PDF export is not active. Add reportlab to requirements.txt and install it.")

    with tab7:
        st.subheader(t["smart"])
        st.write("This section shows how the system can evolve from manual input into automated student tracking.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="card-blue">
                <h3>Smartwatch Sleep Tracking</h3>
                <p>Future versions can collect real sleep cycle data from Apple Watch, Fitbit, Garmin, or sleep tracking apps.</p>
                <ul>
                    <li>Deep sleep duration</li>
                    <li>REM sleep</li>
                    <li>Sleep quality score</li>
                    <li>Recovery score</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            sleep_data = pd.DataFrame(
                {
                    "Sleep Quality": [62, 68, 55, 74, 70, 82, 78],
                    "Deep Sleep": [1.2, 1.5, 1.0, 1.8, 1.6, 2.1, 1.9],
                    "Recovery": [50, 58, 45, 70, 68, 85, 80],
                },
                index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            )
            st.line_chart(sleep_data)

        with col2:
            st.markdown(
                """
                <div class="card-green">
                <h3>Exercise Tracking</h3>
                <p>The system can connect exercise and activity data with productivity analysis.</p>
                <ul>
                    <li>Step count</li>
                    <li>Training duration</li>
                    <li>Activity intensity</li>
                    <li>Energy score</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            exercise_data = pd.DataFrame(
                {
                    "Exercise Minutes": [0, 15, 10, 25, 20, 40, 35],
                    "Energy Score": [45, 58, 55, 70, 68, 85, 80],
                    "Steps / 1000": [2, 4, 3, 6, 7, 10, 9],
                },
                index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            )
            st.bar_chart(exercise_data)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                """
                <div class="card-purple">
                <h3>Weekly Reports</h3>
                <p>The system can automatically generate weekly academic, wellness, nutrition and exercise reports.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                """
                <div class="card-orange">
                <h3>Long-Term Behavior Analysis</h3>
                <p>The system can detect patterns such as sleep-productivity relationship, stress impact, and study consistency.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


render_header()

st.divider()
render_dashboard_tabs()

if st.session_state.sidebar_page == "login":
    st.divider()
    render_login_profile()
elif st.session_state.sidebar_page == "settings":
    st.divider()
    render_settings()
elif st.session_state.sidebar_page == "database":
    st.divider()
    render_database_history()

render_student_bottom_summary()
