import streamlit as st
import pandas as pd
from datetime import date, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="AI Student Performance Tracker",
    page_icon="🎓",
    layout="wide"
)

THEMES = {
    "Neon Blue": {
        "primary": "#38bdf8",
        "secondary": "#2563eb",
        "accent": "#22c55e",
        "warning": "#f97316",
        "danger": "#ef4444",
    },
    "Cyber Purple": {
        "primary": "#a855f7",
        "secondary": "#7c3aed",
        "accent": "#ec4899",
        "warning": "#f59e0b",
        "danger": "#ef4444",
    },
    "Emerald Health": {
        "primary": "#10b981",
        "secondary": "#059669",
        "accent": "#84cc16",
        "warning": "#f97316",
        "danger": "#dc2626",
    },
    "Sunset Orange": {
        "primary": "#fb923c",
        "secondary": "#ea580c",
        "accent": "#facc15",
        "warning": "#f59e0b",
        "danger": "#ef4444",
    }
}

with st.sidebar:
    selected_theme = st.selectbox(
        "🎨 Choose Dashboard Theme",
        list(THEMES.keys())
    )

theme = THEMES[selected_theme]

st.markdown(f"""
<style>
.main-title {{
    font-size: 46px;
    font-weight: 900;
    margin-bottom: 5px;
}}
.sub-title {{
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 25px;
}}
.big-card {{
    padding: 24px;
    border-radius: 20px;
    background: linear-gradient(135deg, {theme["secondary"]}, {theme["primary"]});
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}}
.card-green {{
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #065f46, {theme["accent"]});
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
    padding: 0.7rem 1.4rem;
    font-weight: 700;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🎓 AI Student Performance & Wellness Tracker</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">A complete student tracking platform for productivity, sleep, wellness, nutrition, exercise, AI insights and future smart integrations.</div>',
    unsafe_allow_html=True
)

st.markdown(f"""
<div class="big-card">
<h3>Project Concept</h3>
<p>This system tracks student performance over time. It does not only analyze one daily input. It combines academic performance, sleep habits, stress level, nutrition, exercise, and AI-based recommendations.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.divider()
    st.header("📌 Core Modules")
    st.write("✅ Daily Tracking")
    st.write("✅ Wellness & Nutrition")
    st.write("✅ Exercise Program")
    st.write("✅ Analytics Dashboard")
    st.write("✅ AI Insights")
    st.write("✅ Weekly Report")
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

analyzer = SentimentIntensityAnalyzer()

if "records" not in st.session_state:
    today = date.today()
    st.session_state.records = pd.DataFrame({
        "date": [today - timedelta(days=i) for i in range(9, -1, -1)],
        "mood_text": [
            "I feel tired", "I am okay", "I feel stressed", "I feel focused",
            "I am motivated", "I feel good", "I feel tired but ready",
            "I am productive", "I feel calm", "I feel focused"
        ],
        "sleep_hours": [5, 6, 4, 7, 8, 7, 6, 8, 7, 8],
        "study_hours": [2, 3, 1, 4, 5, 4, 3, 6, 4, 5],
        "focus_level": [4, 5, 3, 7, 8, 7, 6, 9, 7, 8],
        "stress_level": [8, 7, 9, 5, 3, 4, 6, 2, 4, 3],
        "exercise_minutes": [0, 10, 0, 20, 30, 25, 15, 40, 25, 35],
        "task_completion": [30, 45, 25, 70, 85, 75, 60, 92, 78, 88],
        "water_liters": [1.0, 1.4, 0.8, 1.8, 2.0, 1.9, 1.5, 2.3, 2.0, 2.2],
        "calories_quality": [4, 5, 3, 7, 8, 7, 6, 9, 8, 8]
    })


def calculate_scores(df):
    df = df.copy()
    df["sentiment_score"] = df["mood_text"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    df["wellness_score"] = (
        (df["sleep_hours"] / 12 * 20)
        + (df["focus_level"] / 10 * 20)
        + ((10 - df["stress_level"]) / 10 * 20)
        + (df["exercise_minutes"].clip(0, 60) / 60 * 20)
        + (df["water_liters"].clip(0, 2.5) / 2.5 * 10)
        + (df["calories_quality"] / 10 * 10)
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


training_data = pd.DataFrame({
    "sleep_hours": [4, 5, 6, 7, 8, 3, 9, 2, 6, 7, 5, 8, 6, 4, 9, 7, 8, 5, 6, 3],
    "study_hours": [2, 3, 4, 5, 6, 1, 7, 1, 3, 6, 2, 5, 4, 2, 8, 5, 7, 3, 4, 1],
    "sentiment_score": [-0.6, -0.3, 0.0, 0.4, 0.7, -0.8, 0.8, -0.9, 0.2, 0.5, -0.1, 0.6, 0.1, -0.5, 0.9, 0.4, 0.7, -0.2, 0.1, -0.7],
    "focus_level": [3, 4, 5, 7, 8, 2, 9, 1, 5, 8, 4, 7, 6, 3, 9, 7, 9, 5, 6, 2],
    "stress_level": [8, 7, 5, 4, 2, 9, 1, 10, 5, 3, 6, 2, 4, 8, 1, 4, 2, 7, 5, 9],
    "exercise_minutes": [0, 10, 20, 25, 30, 0, 40, 0, 15, 35, 10, 30, 20, 5, 45, 25, 45, 10, 20, 0],
    "task_completion": [30, 40, 55, 70, 85, 20, 95, 15, 60, 80, 45, 75, 65, 35, 98, 72, 94, 45, 60, 20],
    "water_liters": [1.0, 1.2, 1.5, 1.8, 2.1, 0.8, 2.4, 0.6, 1.7, 2.0, 1.3, 2.1, 1.8, 1.0, 2.5, 1.9, 2.4, 1.2, 1.6, 0.7],
    "calories_quality": [4, 5, 6, 7, 8, 3, 9, 2, 6, 8, 5, 8, 7, 4, 9, 7, 9, 5, 6, 3],
    "productivity": [0, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 0, 2, 1, 2, 1, 1, 0]
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
    "calories_quality"
]

labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

X = training_data[features]
y = training_data["productivity"]

model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X, y)
accuracy = accuracy_score(y, model.predict(X))

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧾 Daily Tracker",
    "🥗 Nutrition & Health",
    "🏋️ Exercise Plan",
    "📊 Analytics",
    "🧠 AI Insights",
    "📅 Weekly Report",
    "⌚ Smart Integrations"
])

with tab1:
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
        calories_quality = st.slider("Nutrition Quality", 1, 10, 7)

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
            "calories_quality": calories_quality
        }])

        st.session_state.records = pd.concat(
            [st.session_state.records, new_row],
            ignore_index=True
        ).drop_duplicates(subset=["date"], keep="last")

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
            "calories_quality": calories_quality
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
            st.warning("AI Recommendation: Reduce workload, take a recovery break, drink water, and complete one small task.")
            plan = "25 min light study → 5 min break → 20 min review"
        elif productivity == "Medium":
            st.info("AI Recommendation: Continue with moderate tasks, avoid multitasking, and keep a stable study rhythm.")
            plan = "45 min study → 10 min break → 45 min practice"
        else:
            st.success("AI Recommendation: Strong productivity state. Start with difficult tasks and use deep work.")
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
                calories_quality,
                sentiment * 10
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
            "Sentiment x10"
        ])

        st.subheader("Today’s Tracking Chart")
        st.bar_chart(today_chart)

with tab2:
    st.subheader("🥗 Nutrition & Health Planner")

    col1, col2, col3 = st.columns(3)

    with col1:
        height_cm = st.number_input("Height (cm)", min_value=120, max_value=230, value=175)
    with col2:
        weight_kg = st.number_input("Weight (kg)", min_value=35, max_value=180, value=70)
    with col3:
        goal = st.selectbox("Health Goal", ["Maintain", "Gain Weight", "Lose Fat", "Improve Energy"])

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        bmi_status = "Underweight"
        advice = "Increase healthy calories, protein intake, and strength training."
    elif bmi < 25:
        bmi_status = "Normal"
        advice = "Maintain balanced nutrition and consistent exercise."
    elif bmi < 30:
        bmi_status = "Overweight"
        advice = "Focus on calorie control, walking, and regular training."
    else:
        bmi_status = "Obese Range"
        advice = "A structured health plan and professional guidance are recommended."

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("BMI", bmi)
    m2.metric("Status", bmi_status)
    m3.metric("Goal", goal)

    st.info(advice)

    st.subheader("Healthy Food Suggestions")

    food_data = pd.DataFrame({
        "Category": ["Protein", "Carbohydrate", "Healthy Fat", "Fruit", "Vegetable", "Hydration"],
        "Examples": [
            "Eggs, chicken, fish, yogurt, lentils",
            "Rice, oats, potatoes, whole grain bread",
            "Olive oil, avocado, nuts, peanut butter",
            "Banana, apple, berries, orange",
            "Broccoli, spinach, salad, carrots",
            "Water, mineral water, unsweetened tea"
        ],
        "Purpose": [
            "Muscle repair and satiety",
            "Energy for studying and training",
            "Hormonal health and long-term energy",
            "Vitamins and quick energy",
            "Micronutrients and digestion",
            "Focus, recovery, and mood stability"
        ]
    })

    st.dataframe(food_data, use_container_width=True)

with tab3:
    st.subheader("🏋️ Personalized Exercise Program")

    fitness_goal = st.selectbox(
        "Choose Exercise Goal",
        ["General Health", "Weight Gain / Muscle", "Fat Loss", "Stress Reduction", "Posture & Mobility"]
    )

    if fitness_goal == "General Health":
        plan = [
            "Monday: 30 min walking + 10 min stretching",
            "Wednesday: Full body bodyweight training",
            "Friday: 30 min cycling or walking",
            "Sunday: Light mobility and recovery"
        ]
    elif fitness_goal == "Weight Gain / Muscle":
        plan = [
            "Monday: Push training",
            "Tuesday: Pull training",
            "Thursday: Legs",
            "Saturday: Full body strength training"
        ]
    elif fitness_goal == "Fat Loss":
        plan = [
            "Monday: 40 min brisk walking",
            "Wednesday: Full body circuit",
            "Friday: Interval cardio",
            "Sunday: Long walk"
        ]
    elif fitness_goal == "Stress Reduction":
        plan = [
            "Monday: 20 min walk + breathing",
            "Wednesday: Mobility session",
            "Friday: Light cardio",
            "Sunday: Stretching and recovery"
        ]
    else:
        plan = [
            "Daily: 5 min neck mobility",
            "Daily: 5 min shoulder mobility",
            "3x/week: Core stability",
            "3x/week: Back strengthening"
        ]

    st.markdown(f"""
    <div class="card-green">
    <h3>{fitness_goal}</h3>
    <p>This plan supports physical health, focus, recovery, and student productivity.</p>
    </div>
    """, unsafe_allow_html=True)

    for item in plan:
        st.write("✅", item)

with tab4:
    records = calculate_scores(st.session_state.records)
    records["date"] = pd.to_datetime(records["date"])
    dashboard = records.sort_values("date").set_index("date")

    st.subheader("📊 Analytics Dashboard")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg Sleep", round(records["sleep_hours"].mean(), 2))
    m2.metric("Avg Study", round(records["study_hours"].mean(), 2))
    m3.metric("Avg Stress", round(records["stress_level"].mean(), 2))
    m4.metric("Avg Nutrition", round(records["calories_quality"].mean(), 2))
    m5.metric("Avg Productivity", round(records["productivity_score"].mean(), 2))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sleep Trend")
        st.line_chart(dashboard["sleep_hours"])

        st.subheader("Stress Trend")
        st.line_chart(dashboard["stress_level"])

        st.subheader("Nutrition Quality Trend")
        st.line_chart(dashboard["calories_quality"])

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

    st.subheader("🧠 AI-Based Insights")

    avg_sleep = records["sleep_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_study = records["study_hours"].mean()
    avg_productivity = records["productivity_score"].mean()
    avg_nutrition = records["calories_quality"].mean()
    avg_exercise = records["exercise_minutes"].mean()

    if avg_sleep < 6:
        st.warning("Sleep duration is low. This can reduce focus and productivity.")
    else:
        st.success("Sleep duration is acceptable.")

    if avg_stress > 6:
        st.warning("Stress level is high. Recovery time and workload reduction are recommended.")
    else:
        st.success("Stress level is manageable.")

    if avg_study < 3:
        st.warning("Study consistency is low. A structured weekly plan is recommended.")
    else:
        st.success("Study consistency is acceptable.")

    if avg_nutrition < 6:
        st.warning("Nutrition quality is low. Better food choices can improve energy.")
    else:
        st.success("Nutrition quality is acceptable.")

    if avg_exercise < 20:
        st.warning("Exercise activity is low. Light walking or mobility work is recommended.")
    else:
        st.success("Exercise activity supports wellness.")

    if avg_productivity >= 70:
        st.success("Overall productivity trend is strong.")
    elif avg_productivity >= 50:
        st.info("Overall productivity trend is moderate.")
    else:
        st.warning("Overall productivity trend needs improvement.")

    st.divider()

    st.subheader("Model Feature Importance")
    importance_data = pd.DataFrame({
        "Importance": model.feature_importances_
    }, index=features).sort_values("Importance", ascending=False)

    st.bar_chart(importance_data)

    st.subheader("Model Information")
    st.write("Model: **Random Forest Classifier**")
    st.metric("Training Accuracy", f"{round(accuracy * 100, 2)}%")
    st.caption("The current model uses a demo dataset. Future versions should use real long-term user tracking data.")

with tab6:
    records = calculate_scores(st.session_state.records)

    st.subheader("📅 Automatic Weekly Report")

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
            "Average Wellness"
        ],
        "Value": [
            round(records["sleep_hours"].mean(), 2),
            round(records["study_hours"].mean(), 2),
            round(records["focus_level"].mean(), 2),
            round(records["stress_level"].mean(), 2),
            round(records["exercise_minutes"].mean(), 2),
            round(records["calories_quality"].mean(), 2),
            round(records["water_liters"].mean(), 2),
            round(records["task_completion"].mean(), 2),
            round(records["productivity_score"].mean(), 2),
            round(records["wellness_score"].mean(), 2)
        ]
    })

    st.dataframe(weekly_summary, use_container_width=True)

    st.subheader("Weekly AI Recommendation")

    if records["productivity_score"].mean() < 50:
        st.warning("Reduce workload, improve sleep consistency, increase hydration, and use shorter study blocks.")
    elif records["productivity_score"].mean() < 70:
        st.info("Productivity is moderate. Improve consistency with a fixed weekly routine.")
    else:
        st.success("Strong performance. Deep work sessions and advanced tasks can be increased.")

    csv = weekly_summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Weekly Report as CSV",
        data=csv,
        file_name="weekly_student_report.csv",
        mime="text/csv"
    )

with tab7:
    st.subheader("⌚ Smart Integrations & Future Tracking")

    st.write(
        "This section shows how the system can evolve from manual input into automated student tracking."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
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
            "Recovery": [50, 58, 45, 70, 68, 85, 80]
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
            "Energy Score": [45, 58, 55, 70, 68, 85, 80]
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
        "The final version can become a full AI student coaching platform combining academic tracking, health habits, nutrition, exercise, smartwatch data, and personalized AI recommendations."
    )
