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

st.markdown("""
<style>
.main-title {
    font-size: 46px;
    font-weight: 900;
}
.sub-title {
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 25px;
}
.card {
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 18px;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}
.blue-card {background: linear-gradient(135deg, #1e3a8a, #2563eb);}
.green-card {background: linear-gradient(135deg, #065f46, #10b981);}
.purple-card {background: linear-gradient(135deg, #581c87, #a855f7);}
.orange-card {background: linear-gradient(135deg, #9a3412, #f97316);}
.red-card {background: linear-gradient(135deg, #7f1d1d, #ef4444);}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🎓 AI Student Performance & Wellness Tracker</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Tracking student behavior, analyzing productivity trends, and generating AI-based study recommendations.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("📌 Project Concept")
    st.write("This is not only a daily calculator.")
    st.write("It is a tracking and analytics dashboard for student performance.")

    st.divider()

    st.header("Core Modules")
    st.write("✅ Daily Tracker")
    st.write("✅ Analytics Dashboard")
    st.write("✅ AI Insights")
    st.write("✅ Weekly Reports")
    st.write("✅ Smart Integrations")
    st.write("✅ Long-Term Analysis")

    st.divider()

    st.header("Technologies")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- VADER NLP")
    st.write("- Scikit-learn")
    st.write("- Pandas")
    st.write("- Random Forest")

analyzer = SentimentIntensityAnalyzer()

if "records" not in st.session_state:
    today = date.today()
    st.session_state.records = pd.DataFrame({
        "date": [today - timedelta(days=i) for i in range(6, -1, -1)],
        "mood_text": [
            "I feel tired",
            "I am okay",
            "I feel stressed",
            "I feel focused",
            "I am a bit tired",
            "I feel motivated",
            "I feel good"
        ],
        "sleep_hours": [5, 6, 4, 7, 6, 8, 7],
        "study_hours": [2, 3, 1, 4, 3, 5, 4],
        "focus_level": [4, 5, 3, 7, 6, 8, 7],
        "stress_level": [8, 7, 9, 5, 6, 3, 4],
        "exercise_minutes": [0, 10, 0, 20, 15, 30, 25],
        "task_completion": [30, 45, 25, 70, 60, 85, 75],
    })


def add_scores(df):
    df = df.copy()

    df["sentiment_score"] = df["mood_text"].apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )

    df["wellness_score"] = (
        (df["sleep_hours"] / 12 * 25)
        + (df["focus_level"] / 10 * 25)
        + ((10 - df["stress_level"]) / 10 * 25)
        + (df["exercise_minutes"].clip(0, 60) / 60 * 25)
    ).round(2)

    df["productivity_score"] = (
        (df["study_hours"] / 10 * 25)
        + (df["task_completion"] / 100 * 35)
        + (df["focus_level"] / 10 * 25)
        + (df["sentiment_score"].clip(lower=0) * 15)
    ).round(2)

    df["risk_score"] = (
        (df["stress_level"] * 8)
        + ((12 - df["sleep_hours"]) * 4)
        + ((10 - df["focus_level"]) * 4)
    ).round(2)

    return df


training_data = pd.DataFrame({
    "sleep_hours": [4, 5, 6, 7, 8, 3, 9, 2, 6, 7, 5, 8, 6, 4, 9, 6, 7, 5, 8, 3],
    "study_hours": [2, 3, 4, 5, 6, 1, 7, 1, 3, 6, 2, 5, 4, 2, 8, 4, 6, 3, 7, 1],
    "sentiment_score": [-0.6, -0.3, 0.0, 0.4, 0.7, -0.8, 0.8, -0.9, 0.2, 0.5, -0.1, 0.6, 0.1, -0.5, 0.9, 0.2, 0.6, -0.2, 0.8, -0.7],
    "focus_level": [3, 4, 5, 7, 8, 2, 9, 1, 5, 8, 4, 7, 6, 3, 9, 6, 8, 5, 9, 2],
    "stress_level": [8, 7, 5, 4, 2, 9, 1, 10, 5, 3, 6, 2, 4, 8, 1, 5, 3, 7, 2, 9],
    "exercise_minutes": [0, 10, 20, 25, 30, 0, 40, 0, 15, 35, 10, 30, 20, 5, 45, 15, 35, 5, 50, 0],
    "task_completion": [30, 40, 55, 70, 85, 20, 95, 15, 60, 80, 45, 75, 65, 35, 98, 62, 84, 45, 92, 22],
    "productivity": [0, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 0, 2, 1, 2, 1, 2, 0]
})

features = [
    "sleep_hours",
    "study_hours",
    "sentiment_score",
    "focus_level",
    "stress_level",
    "exercise_minutes",
    "task_completion"
]

labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

X = training_data[features]
y = training_data["productivity"]

model = RandomForestClassifier(random_state=42, n_estimators=150)
model.fit(X, y)
accuracy = accuracy_score(y, model.predict(X))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧾 Daily Tracker",
    "📊 Analytics Dashboard",
    "🧠 AI Insights",
    "📅 Weekly Report",
    "⌚ Smart Integrations",
    "🚀 Roadmap"
])

with tab1:
    st.subheader("🧾 Daily Student Tracker")

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

    if st.button("Save Today and Analyze"):
        new_row = pd.DataFrame([{
            "date": entry_date,
            "mood_text": mood_text,
            "sleep_hours": sleep_hours,
            "study_hours": study_hours,
            "focus_level": focus_level,
            "stress_level": stress_level,
            "exercise_minutes": exercise_minutes,
            "task_completion": task_completion
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
            "task_completion": task_completion
        }])

        prediction = model.predict(input_data)[0]
        productivity = labels[prediction]

        temp = add_scores(new_row)
        wellness_score = temp["wellness_score"].iloc[0]
        productivity_score = temp["productivity_score"].iloc[0]

        if sentiment >= 0.05:
            mood = "Positive"
        elif sentiment <= -0.05:
            mood = "Negative"
        else:
            mood = "Neutral"

        if stress_level >= 8 or sleep_hours <= 4:
            risk_level = "High Risk"
        elif stress_level >= 6 or sleep_hours <= 6:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        st.divider()
        st.subheader("Today’s AI Analysis")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mood", mood)
        c2.metric("Productivity", productivity)
        c3.metric("Productivity Score", f"{productivity_score}/100")
        c4.metric("Wellness Score", f"{wellness_score}/100")
        c5.metric("Risk Level", risk_level)

        if productivity == "Low":
            st.warning("Recommendation: Reduce workload, take a recovery break, and focus on one small task.")
            plan = "25 min light study → 5 min break → 20 min review"
        elif productivity == "Medium":
            st.info("Recommendation: Continue with moderate tasks and avoid multitasking.")
            plan = "45 min study → 10 min break → 45 min practice"
        else:
            st.success("Recommendation: Strong productivity state. Start with difficult tasks first.")
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
                sentiment * 10
            ]
        }, index=[
            "Sleep Hours",
            "Study Hours",
            "Focus Level",
            "Stress Level",
            "Exercise / 10",
            "Task Completion / 10",
            "Sentiment x10"
        ])

        st.subheader("Today’s Tracking Chart")
        st.bar_chart(today_chart)

with tab2:
    records = add_scores(st.session_state.records)
    records["date"] = pd.to_datetime(records["date"])
    dashboard = records.sort_values("date").set_index("date")

    st.subheader("📊 Student Analytics Dashboard")
    st.write("This dashboard tracks student performance over time, not only one input.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg Sleep", round(records["sleep_hours"].mean(), 2))
    m2.metric("Avg Study", round(records["study_hours"].mean(), 2))
    m3.metric("Avg Stress", round(records["stress_level"].mean(), 2))
    m4.metric("Avg Focus", round(records["focus_level"].mean(), 2))
    m5.metric("Avg Productivity", round(records["productivity_score"].mean(), 2))

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sleep Trend")
        st.line_chart(dashboard["sleep_hours"])

        st.subheader("Stress Trend")
        st.line_chart(dashboard["stress_level"])

    with col_b:
        st.subheader("Study Trend")
        st.line_chart(dashboard["study_hours"])

        st.subheader("Productivity Trend")
        st.line_chart(dashboard["productivity_score"])

    st.subheader("Complete Tracking Data")
    st.dataframe(records, use_container_width=True)

with tab3:
    records = add_scores(st.session_state.records)

    st.subheader("🧠 AI-Based Insights")

    avg_sleep = records["sleep_hours"].mean()
    avg_study = records["study_hours"].mean()
    avg_stress = records["stress_level"].mean()
    avg_focus = records["focus_level"].mean()
    avg_productivity = records["productivity_score"].mean()

    if avg_sleep < 6:
        st.warning("Sleep duration is below the recommended level. This may reduce productivity.")
    else:
        st.success("Sleep duration is at a healthy level.")

    if avg_stress > 6:
        st.warning("Stress trend is high. Recovery time and workload reduction are recommended.")
    else:
        st.success("Stress level is manageable.")

    if avg_study < 3:
        st.warning("Study consistency is low. A structured weekly plan is recommended.")
    else:
        st.success("Study consistency is acceptable.")

    if avg_focus < 5:
        st.warning("Focus level is low. Shorter study sessions may help.")
    else:
        st.success("Focus level is acceptable.")

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
    st.caption(
        "The current model uses a demo dataset. In a real system, this model would be trained with long-term tracking data."
    )

with tab4:
    records = add_scores(st.session_state.records)

    st.subheader("📅 Automatic Weekly Report")

    weekly_summary = pd.DataFrame({
        "Metric": [
            "Average Sleep Hours",
            "Average Study Hours",
            "Average Focus Level",
            "Average Stress Level",
            "Average Exercise Minutes",
            "Average Task Completion",
            "Average Productivity Score",
            "Average Wellness Score"
        ],
        "Value": [
            round(records["sleep_hours"].mean(), 2),
            round(records["study_hours"].mean(), 2),
            round(records["focus_level"].mean(), 2),
            round(records["stress_level"].mean(), 2),
            round(records["exercise_minutes"].mean(), 2),
            round(records["task_completion"].mean(), 2),
            round(records["productivity_score"].mean(), 2),
            round(records["wellness_score"].mean(), 2)
        ]
    })

    st.dataframe(weekly_summary, use_container_width=True)

    st.subheader("Weekly Recommendation")

    if records["productivity_score"].mean() < 50:
        st.warning("The student should reduce workload, improve sleep consistency, and use shorter focused sessions.")
    elif records["productivity_score"].mean() < 70:
        st.info("The student has moderate productivity. A more consistent weekly plan is recommended.")
    else:
        st.success("The student shows strong productivity. Deep work sessions can be increased.")

    csv = weekly_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Weekly Report as CSV",
        data=csv,
        file_name="weekly_student_report.csv",
        mime="text/csv"
    )

with tab5:
    st.subheader("⌚ Smartwatch & Tracking Integrations")

    st.write(
        "This section demonstrates how future versions can connect manual tracking with automated tracking systems."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card blue-card">
            <h3>⌚ Smartwatch Sleep Tracking</h3>
            <p>Future versions can collect sleep cycle data from Apple Watch, Fitbit, Garmin, or sleep apps.</p>
            <ul>
                <li>Deep sleep duration</li>
                <li>Sleep quality score</li>
                <li>Wake-up consistency</li>
                <li>Recovery level</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        sleep_cycle_data = pd.DataFrame({
            "Sleep Quality": [62, 68, 55, 74, 70, 82, 78],
            "Deep Sleep": [1.2, 1.5, 1.0, 1.8, 1.6, 2.1, 1.9]
        }, index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        st.line_chart(sleep_cycle_data)

    with col2:
        st.markdown("""
        <div class="card green-card">
            <h3>🏃 Exercise Tracking</h3>
            <p>The system can connect exercise and activity data with productivity analysis.</p>
            <ul>
                <li>Exercise minutes</li>
                <li>Step count</li>
                <li>Activity intensity</li>
                <li>Energy level</li>
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
        <div class="card purple-card">
            <h3>📅 Weekly Reports</h3>
            <p>The app can automatically summarize weekly academic and wellness data.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card orange-card">
            <h3>📈 Long-Term Behavior Analysis</h3>
            <p>The system can detect long-term behavioral patterns and productivity risks.</p>
        </div>
        """, unsafe_allow_html=True)

with tab6:
    st.subheader("🚀 Development Roadmap")

    st.markdown("""
    ### Current MVP
    - Manual daily student tracking
    - NLP-based mood analysis
    - ML-based productivity prediction
    - Trend visualization
    - Weekly report generation

    ### Next Version
    - Smartwatch sleep cycle integration
    - Automatic activity tracking
    - User accounts
    - Cloud database
    - Real student dataset
    - Personalized AI coaching

    ### Final Vision
    A long-term student performance platform that combines academic tracking,
    wellness analytics, behavioral insights, and AI-based study planning.
    """)

    st.success(
        "The project now directly addresses tracking, analysis, prediction, recommendation, and future integration."
    )
