import io
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
    layout="wide"
)


THEMES = {
    "Neon Blue": {"primary": "#38bdf8", "secondary": "#2563eb", "accent": "#22c55e", "warning": "#f97316", "danger": "#ef4444"},
    "Cyber Purple": {"primary": "#a855f7", "secondary": "#7c3aed", "accent": "#ec4899", "warning": "#f59e0b", "danger": "#ef4444"},
    "Emerald Health": {"primary": "#10b981", "secondary": "#059669", "accent": "#84cc16", "warning": "#f97316", "danger": "#dc2626"},
    "Sunset Orange": {"primary": "#fb923c", "secondary": "#ea580c", "accent": "#facc15", "warning": "#f59e0b", "danger": "#ef4444"},
}

with st.sidebar:
    selected_theme = st.selectbox("🎨 Choose Dashboard Theme", list(THEMES.keys()))

theme = THEMES[selected_theme]

st.markdown(f"""
<style>
.main-title {{
    font-size: 48px;
    font-weight: 900;
}}
.sub-title {{
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 24px;
}}
.hero-card {{
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, {theme["secondary"]}, {theme["primary"]});
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}}
.info-card {{
    padding: 20px;
    border-radius: 18px;
    background: #111827;
    border: 1px solid #374151;
    color: white;
    margin-bottom: 15px;
}}
.card-blue {{
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1e3a8a, {theme["primary"]});
    color: white;
    margin-bottom: 15px;
}}
.card-green {{
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #064e3b, {theme["accent"]});
    color: white;
    margin-bottom: 15px;
}}
.card-purple {{
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #581c87, #a855f7);
    color: white;
    margin-bottom: 15px;
}}
.card-orange {{
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #9a3412, {theme["warning"]});
    color: white;
    margin-bottom: 15px;
}}
.card-red {{
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #7f1d1d, {theme["danger"]});
    color: white;
    margin-bottom: 15px;
}}
div.stButton > button {{
    background: linear-gradient(135deg, {theme["secondary"]}, {theme["primary"]});
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.4rem;
    font-weight: 800;
}}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🎓 AI Student Performance & Wellness Tracker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A complete AI dashboard for student productivity, sleep, wellness, nutrition, exercise, weekly reports, AI coaching and future smart integrations.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-card">
<h3>Project Concept</h3>
<p>
This platform tracks student performance over time. It does not only analyze one daily input.
It combines academic performance, sleep habits, stress level, nutrition, exercise, and AI-based recommendations.
</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.divider()
    st.header("📌 Core Modules")
    st.write("✅ Login / Student Profile")
    st.write("✅ Daily Tracking")
    st.write("✅ Nutrition & Health")
    st.write("✅ Exercise Program")
    st.write("✅ Analytics Dashboard")
    st.write("✅ AI Coach")
    st.write("✅ Weekly PDF Report")
    st.write("✅ Smartwatch Integration")
    st.write("✅ Long-Term Behavior Analysis")

    st.divider()
    st.header("🛠 Technologies")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- VADER NLP")
    st.write("- Scikit-learn")
    st.write("- Random Forest")
    st.write("- Pandas")
    st.write("- ReportLab PDF")


analyzer = SentimentIntensityAnalyzer()

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "Student",
        "student_id": "0000",
        "faculty": "Software Engineering",
        "semester": 2,
        "age": 22,
        "height_cm": 175,
        "weight_kg": 70,
        "goal": "Improve Productivity",
    }

if "records" not in st.session_state:
    today = date.today()
    st.session_state.records = pd.DataFrame({
        "date": [today - timedelta(days=i) for i in range(13, -1, -1)],
        "mood_text": [
            "I feel tired", "I am okay", "I feel stressed", "I feel focused",
            "I am motivated", "I feel good", "I feel tired but ready",
            "I am productive", "I feel calm", "I feel focused",
            "I feel low energy", "I feel disciplined", "I feel strong", "I feel balanced",
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
    })


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


def get_bmi_status(bmi):
    if bmi < 18.5:
        return "Underweight", "Increase healthy calories, protein intake, and strength training."
    if bmi < 25:
        return "Normal", "Maintain balanced nutrition and consistent exercise."
    if bmi < 30:
        return "Overweight", "Focus on calorie control, walking, and regular training."
    return "High Range", "A structured plan and professional guidance are recommended."


def get_ai_recommendations(records):
    avg_sleep = records["sleep_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_productivity = records["productivity_score"].mean()
    avg_nutrition = records["nutrition_quality"].mean()
    avg_exercise = records["exercise_minutes"].mean()
    avg_water = records["water_liters"].mean()

    messages = []

    if avg_sleep < 6:
        messages.append(("warning", "Sleep is low. Try to increase sleep by at least 1 hour."))
    else:
        messages.append(("success", "Sleep duration is acceptable."))

    if avg_stress > 6:
        messages.append(("error", "Stress trend is high. Add recovery time and reduce workload intensity."))
    else:
        messages.append(("success", "Stress level is manageable."))

    if avg_nutrition < 6:
        messages.append(("warning", "Nutrition quality is low. Add protein, fruit, vegetables and stable meals."))
    else:
        messages.append(("success", "Nutrition quality is acceptable."))

    if avg_exercise < 20:
        messages.append(("warning", "Exercise activity is low. Add walking or mobility sessions."))
    else:
        messages.append(("success", "Exercise supports wellness and focus."))

    if avg_water < 1.8:
        messages.append(("warning", "Water intake is low. Increase hydration for better focus."))
    else:
        messages.append(("success", "Hydration level is acceptable."))

    if avg_productivity >= 70:
        messages.append(("success", "Productivity trend is strong. Deep work sessions can be increased."))
    elif avg_productivity >= 50:
        messages.append(("info", "Productivity trend is moderate. Use a more structured weekly plan."))
    else:
        messages.append(("warning", "Productivity needs improvement. Start with smaller tasks and shorter study blocks."))

    return messages


def create_pdf_report(profile, summary, recommendations):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "AI Student Weekly Performance Report")

    y -= 35
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Student: {profile['name']}")
    y -= 18
    pdf.drawString(50, y, f"Student ID: {profile['student_id']}")
    y -= 18
    pdf.drawString(50, y, f"Faculty: {profile['faculty']}")
    y -= 18
    pdf.drawString(50, y, f"Semester: {profile['semester']}")
    y -= 18
    pdf.drawString(50, y, f"Goal: {profile['goal']}")

    y -= 35
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Weekly Metrics")

    pdf.setFont("Helvetica", 11)
    y -= 25
    for _, row in summary.iterrows():
        pdf.drawString(60, y, f"{row['Metric']}: {row['Value']}")
        y -= 18

    y -= 15
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "AI Coach Recommendations")

    pdf.setFont("Helvetica", 10)
    y -= 25
    for _, msg in recommendations:
        text = f"- {msg}"
        pdf.drawString(60, y, text[:95])
        y -= 18

        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)

    pdf.save()
    buffer.seek(0)
    return buffer


training_data = pd.DataFrame({
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
})

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


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "👤 Login / Profile",
    "🧾 Daily Tracker",
    "🥗 Nutrition & Health",
    "🏋️ Exercise Plan",
    "📊 Analytics",
    "🤖 AI Coach",
    "📄 Weekly Report",
    "⌚ Smart Integrations",
])


with tab1:
    st.subheader("👤 Student Login & Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Full Name", st.session_state.profile["name"])
        student_id = st.text_input("Student ID", st.session_state.profile["student_id"])

    with col2:
        faculty = st.selectbox(
            "Faculty",
            ["Software Engineering", "Computer Science", "Business", "Design", "AI Engineering"],
            index=0,
        )
        semester = st.slider("Semester", 1, 8, st.session_state.profile["semester"])

    with col3:
        age = st.number_input("Age", min_value=10, max_value=80, value=st.session_state.profile["age"])
        goal = st.selectbox(
            "Main Goal",
            ["Improve Productivity", "Gain Weight", "Lose Fat", "Reduce Stress", "Improve Sleep", "Build Discipline"],
            index=0,
        )

    col4, col5 = st.columns(2)

    with col4:
        height_cm = st.number_input("Height (cm)", min_value=120, max_value=230, value=st.session_state.profile["height_cm"])

    with col5:
        weight_kg = st.number_input("Weight (kg)", min_value=35, max_value=180, value=st.session_state.profile["weight_kg"])

    if st.button("Save Student Profile"):
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
        st.success("Student profile saved successfully.")

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    bmi_status, bmi_advice = get_bmi_status(bmi)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", bmi)
    m2.metric("BMI Status", bmi_status)
    m3.metric("Semester", semester)
    m4.metric("Goal", goal)

    st.info(bmi_advice)

    st.markdown("""
    <div class="info-card">
    <h3>Why Login / Profile Matters</h3>
    <p>
    The system is personalized for each student. It stores profile data and uses this information
    for nutrition, exercise, productivity and wellness recommendations.
    </p>
    </div>
    """, unsafe_allow_html=True)


with tab2:
    st.subheader("🧾 Daily Student Tracking")

    col1, col2 = st.columns(2)

    with col1:
        entry_date = st.date_input("Date", value=date.today())
        mood_text = st.text_area("How do you feel today?", value="I feel focused and ready to study.")
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
        study_hours = st.slider("Study Hours", 0, 10, 4)
        task_completion = st.slider("Task Completion (%)", 0, 100, 65)

    with col2:
        focus_level = st.slider("Focus Level", 1, 10, 7)
        stress_level = st.slider("Stress Level", 1, 10, 4)
        exercise_minutes = st.slider("Exercise Minutes", 0, 120, 25)
        water_liters = st.slider("Water Intake (Liters)", 0.0, 4.0, 2.0)
        nutrition_quality = st.slider("Nutrition Quality", 1, 10, 7)

    if st.button("Save Today and Generate AI Analysis"):
        new_row = pd.DataFrame([{
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
        }])

        st.session_state.records = (
            pd.concat([st.session_state.records, new_row], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
        )

        sentiment = analyzer.polarity_scores(mood_text)["compound"]

        input_data = pd.DataFrame([{
            "sleep_hours": sleep_hours,
            "study_hours": study_hours,
            "sentiment_score": sentiment,
            "focus_level": focus_level,
            "stress_level": stress_level,
            "exercise_minutes": exercise_minutes,
            "task_completion": task_completion,
            "water_liters": water_liters,
            "nutrition_quality": nutrition_quality,
        }])

        prediction = model.predict(input_data)[0]
        productivity = labels[prediction]

        temp = calculate_scores(new_row)
        productivity_score = temp["productivity_score"].iloc[0]
        wellness_score = temp["wellness_score"].iloc[0]
        risk_score = temp["risk_score"].iloc[0]

        mood = "Positive" if sentiment >= 0.05 else "Negative" if sentiment <= -0.05 else "Neutral"
        risk_level = "High Risk" if risk_score >= 70 else "Medium Risk" if risk_score >= 45 else "Low Risk"

        st.divider()
        st.subheader("Today’s AI Analysis")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mood", mood)
        c2.metric("Productivity", productivity)
        c3.metric("Productivity Score", f"{productivity_score}/100")
        c4.metric("Wellness Score", f"{wellness_score}/100")
        c5.metric("Risk Level", risk_level)

        if productivity == "Low":
            st.warning("AI Coach: Reduce workload, drink water, eat a healthy meal, and complete one small task.")
            plan = "25 min light study → 5 min break → 20 min review"
        elif productivity == "Medium":
            st.info("AI Coach: Continue with moderate tasks, avoid multitasking, and keep a stable rhythm.")
            plan = "45 min study → 10 min break → 45 min practice"
        else:
            st.success("AI Coach: Strong productivity state. Start with difficult tasks and use deep work.")
            plan = "60 min deep work → 10 min break → 60 min focused study"

        st.write(f"**Suggested Daily Plan:** {plan}")

        today_chart = pd.DataFrame({
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
        }, index=[
            "Sleep",
            "Study",
            "Focus",
            "Stress",
            "Exercise / 10",
            "Tasks / 10",
            "Water",
            "Nutrition",
            "Sentiment x10",
        ])

        st.subheader("Today’s Tracking Chart")
        st.bar_chart(today_chart)


with tab3:
    st.subheader("🥗 Nutrition & Health Planner")

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

    food_data = pd.DataFrame({
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
    })

    st.dataframe(food_data, use_container_width=True)

    if goal == "Gain Weight":
        st.success("Nutrition Focus: Increase calories with protein, oats, rice, potatoes, olive oil and healthy snacks.")
    elif goal == "Lose Fat":
        st.info("Nutrition Focus: Increase protein, vegetables, water intake and reduce processed foods.")
    elif goal == "Improve Sleep":
        st.info("Nutrition Focus: Avoid caffeine late in the day and eat lighter meals before sleep.")
    else:
        st.success("Nutrition Focus: Keep balanced meals and stable hydration for consistent energy.")


with tab4:
    st.subheader("🏋️ Personalized Exercise Program")

    fitness_goal = st.selectbox(
        "Choose Exercise Goal",
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

    st.markdown(f"""
    <div class="card-green">
    <h3>{fitness_goal}</h3>
    <p>This exercise plan supports physical health, focus, recovery and student productivity.</p>
    </div>
    """, unsafe_allow_html=True)

    for item in plans[fitness_goal]:
        st.write("✅", item)


with tab5:
    records = calculate_scores(st.session_state.records)
    records["date"] = pd.to_datetime(records["date"])
    dashboard = records.sort_values("date").set_index("date")

    st.subheader("📊 Analytics Dashboard")

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


with tab6:
    records = calculate_scores(st.session_state.records)

    st.subheader("🤖 Personal AI Coach")

    recommendations = get_ai_recommendations(records)

    for level, msg in recommendations:
        if level == "success":
            st.success(msg)
        elif level == "error":
            st.error(msg)
        elif level == "info":
            st.info(msg)
        else:
            st.warning(msg)

    st.divider()

    st.subheader("AI Coach Daily Guidance")

    latest = records.sort_values("date").iloc[-1]

    st.markdown(f"""
    <div class="info-card">
    <h3>Today’s Personalized Guidance</h3>
    <p><b>Sleep:</b> {latest["sleep_hours"]} hours</p>
    <p><b>Study:</b> {latest["study_hours"]} hours</p>
    <p><b>Stress:</b> {latest["stress_level"]}/10</p>
    <p><b>Wellness Score:</b> {latest["wellness_score"]}/100</p>
    <p><b>Productivity Score:</b> {latest["productivity_score"]}/100</p>
    <p>
    The AI Coach acts like a digital mentor. It observes patterns, warns the student when risk is high,
    and suggests practical actions for study, recovery, hydration, exercise and nutrition.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Model Feature Importance")
    importance_data = pd.DataFrame(
        {"Importance": model.feature_importances_},
        index=features,
    ).sort_values("Importance", ascending=False)

    st.bar_chart(importance_data)

    st.subheader("Model Information")
    st.write("Model: **Random Forest Classifier**")
    st.metric("Training Accuracy", f"{round(accuracy * 100, 2)}%")


with tab7:
    records = calculate_scores(st.session_state.records)

    st.subheader("📄 Automatic Weekly Report")

    weekly_summary = pd.DataFrame({
        "Metric": [
            "Average Sleep",
            "Average Study",
            "Average Focus",
            "Average Stress",
            "Average Exercise",
            "Average Nutrition",
            "Average Water Intake",
            "Average Task Completion",
            "Average Productivity",
            "Average Wellness",
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
    })

    st.dataframe(weekly_summary, use_container_width=True)

    recommendations = get_ai_recommendations(records)

    st.subheader("Weekly AI Recommendation")
    for level, msg in recommendations:
        if level == "success":
            st.success(msg)
        elif level == "error":
            st.error(msg)
        elif level == "info":
            st.info(msg)
        else:
            st.warning(msg)

    csv = weekly_summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Weekly Report as CSV",
        data=csv,
        file_name="weekly_student_report.csv",
        mime="text/csv",
    )

    if PDF_AVAILABLE:
        pdf_file = create_pdf_report(st.session_state.profile, weekly_summary, recommendations)
        st.download_button(
            label="Download Weekly Report as PDF",
            data=pdf_file,
            file_name="weekly_student_report.pdf",
            mime="application/pdf",
        )
    else:
        st.warning("PDF export is not active. Add reportlab to requirements.txt and install it.")


with tab8:
    st.subheader("⌚ Smart Integrations & Future Tracking")

    st.write("This section shows how the system can evolve from manual input into automated student tracking.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card-blue">
        <h3>⌚ Smartwatch Sleep Tracking</h3>
        <p>Future versions can collect real sleep cycle data from Apple Watch, Fitbit, Garmin, or sleep tracking apps.</p>
        <ul>
            <li>Deep sleep duration</li>
            <li>REM sleep</li>
            <li>Sleep quality score</li>
            <li>Recovery score</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        sleep_data = pd.DataFrame({
            "Sleep Quality": [62, 68, 55, 74, 70, 82, 78],
            "Deep Sleep": [1.2, 1.5, 1.0, 1.8, 1.6, 2.1, 1.9],
            "Recovery": [50, 58, 45, 70, 68, 85, 80],
        }, index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        st.line_chart(sleep_data)

    with col2:
        st.markdown("""
        <div class="card-green">
        <h3>🏃 Exercise Tracking</h3>
        <p>The system can connect exercise and activity data with productivity analysis.</p>
        <ul>
            <li>Step count</li>
            <li>Training duration</li>
            <li>Activity intensity</li>
            <li>Energy score</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        exercise_data = pd.DataFrame({
            "Exercise Minutes": [0, 15, 10, 25, 20, 40, 35],
            "Energy Score": [45, 58, 55, 70, 68, 85, 80],
            "Steps / 1000": [2, 4, 3, 6, 7, 10, 9],
        }, index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        st.bar_chart(exercise_data)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="card-purple">
        <h3>📅 Weekly Reports</h3>
        <p>The system can automatically generate weekly academic, wellness, nutrition and exercise reports.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card-orange">
        <h3>📈 Long-Term Behavior Analysis</h3>
        <p>The system can detect patterns such as sleep-productivity relationship, stress impact, and study consistency.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Final Vision")
    st.success(
        "The final version can become a full AI student coaching platform combining academic tracking, health habits, nutrition, exercise, smartwatch data, PDF reports and personalized AI recommendations."
    )
