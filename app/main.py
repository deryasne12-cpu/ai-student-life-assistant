import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="AI Student Performance Tracker",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Student Performance & Wellness Tracker")

st.info(
    "This MVP tracks daily student performance data, analyzes behavior trends, "
    "predicts productivity with Machine Learning, and generates personalized AI insights."
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("📌 Project Concept")
    st.write("This project is not only a daily calculator.")
    st.write("It is designed as a student tracking and analytics system.")

    st.divider()

    st.header("Future Integrations")
    st.write("- Smartwatch sleep tracking")
    st.write("- Exercise tracking")
    st.write("- Weekly reports")
    st.write("- Long-term behavior analysis")

    st.divider()

    st.header("Technologies")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- VADER NLP")
    st.write("- Scikit-learn")
    st.write("- Pandas")
    st.write("- Random Forest")

# --------------------------------------------------
# Demo history data
# --------------------------------------------------
history_data = pd.DataFrame({
    "date": pd.date_range(end=date.today(), periods=7),
    "sleep_hours": [5, 6, 4, 7, 6, 8, 7],
    "study_hours": [2, 3, 1, 4, 3, 5, 4],
    "focus_level": [4, 5, 3, 7, 6, 8, 7],
    "stress_level": [8, 7, 9, 5, 6, 3, 4],
    "exercise_minutes": [0, 10, 0, 20, 15, 30, 25],
    "task_completion": [30, 45, 25, 70, 60, 85, 75],
    "productivity_score": [35, 48, 28, 72, 63, 88, 78]
})

# --------------------------------------------------
# Training dataset
# --------------------------------------------------
training_data = pd.DataFrame({
    "sleep_hours": [4, 5, 6, 7, 8, 3, 9, 2, 6, 7, 5, 8, 6, 4, 9],
    "study_hours": [2, 3, 4, 5, 6, 1, 7, 1, 3, 6, 2, 5, 4, 2, 8],
    "sentiment_score": [-0.6, -0.3, 0.0, 0.4, 0.7, -0.8, 0.8, -0.9, 0.2, 0.5, -0.1, 0.6, 0.1, -0.5, 0.9],
    "focus_level": [3, 4, 5, 7, 8, 2, 9, 1, 5, 8, 4, 7, 6, 3, 9],
    "stress_level": [8, 7, 5, 4, 2, 9, 1, 10, 5, 3, 6, 2, 4, 8, 1],
    "exercise_minutes": [0, 10, 20, 25, 30, 0, 40, 0, 15, 35, 10, 30, 20, 5, 45],
    "task_completion": [30, 40, 55, 70, 85, 20, 95, 15, 60, 80, 45, 75, 65, 35, 98],
    "productivity": [0, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 0, 2]
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

X = training_data[features]
y = training_data["productivity"]

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)
accuracy = accuracy_score(y, model.predict(X))

labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

analyzer = SentimentIntensityAnalyzer()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧾 Daily Tracker",
    "📊 Analytics Dashboard",
    "🧠 AI Insights",
    "🚀 Future Roadmap"
])

# --------------------------------------------------
# Tab 1: Daily Tracker
# --------------------------------------------------
with tab1:
    st.subheader("Daily Student Tracking")

    col1, col2 = st.columns(2)

    with col1:
        mood_text = st.text_area("How do you feel today?")
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
        study_hours = st.slider("Study Hours", 0, 10, 4)
        task_completion = st.slider("Task Completion (%)", 0, 100, 60)

    with col2:
        focus_level = st.slider("Focus Level", 1, 10, 6)
        stress_level = st.slider("Stress Level", 1, 10, 5)
        exercise_minutes = st.slider("Exercise Minutes", 0, 120, 20)

    if st.button("Analyze and Track Today"):

        if mood_text.strip() == "":
            st.warning("Please enter your mood text first.")
        else:
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

            wellness_score = round(
                ((sleep_hours / 12) * 25) +
                ((focus_level / 10) * 25) +
                (((10 - stress_level) / 10) * 25) +
                (min(exercise_minutes / 60, 1) * 25),
                2
            )

            productivity_score = round(
                ((study_hours / 10) * 25) +
                ((task_completion / 100) * 35) +
                ((focus_level / 10) * 25) +
                (max(sentiment, 0) * 15),
                2
            )

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

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mood", mood)
            c2.metric("Productivity", productivity)
            c3.metric("Wellness Score", f"{wellness_score}/100")
            c4.metric("Risk Level", risk_level)

            if prediction == 0:
                recommendation = "Reduce workload, take breaks, and focus on one small task first."
                plan = "25 min light study → 5 min break → 20 min review"
            elif prediction == 1:
                recommendation = "Continue with moderate tasks and avoid multitasking."
                plan = "45 min study → 10 min break → 45 min practice"
            else:
                recommendation = "You are in a strong productivity state. Start with the most difficult task."
                plan = "60 min deep work → 10 min break → 60 min focused study"

            st.success(recommendation)
            st.write(f"**Suggested Daily Plan:** {plan}")

            today_row = pd.DataFrame({
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
            st.bar_chart(today_row)

# --------------------------------------------------
# Tab 2: Analytics Dashboard
# --------------------------------------------------
with tab2:
    st.subheader("7-Day Student Analytics Dashboard")

    st.write(
        "This section demonstrates how the system can track student performance over time."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Sleep", round(history_data["sleep_hours"].mean(), 2))
    c2.metric("Avg Study", round(history_data["study_hours"].mean(), 2))
    c3.metric("Avg Stress", round(history_data["stress_level"].mean(), 2))
    c4.metric("Avg Productivity", round(history_data["productivity_score"].mean(), 2))

    st.subheader("Sleep Trend")
    st.line_chart(history_data.set_index("date")["sleep_hours"])

    st.subheader("Study Hours Trend")
    st.line_chart(history_data.set_index("date")["study_hours"])

    st.subheader("Stress Trend")
    st.line_chart(history_data.set_index("date")["stress_level"])

    st.subheader("Productivity Trend")
    st.line_chart(history_data.set_index("date")["productivity_score"])

    st.subheader("Weekly Data Table")
    st.dataframe(history_data)

# --------------------------------------------------
# Tab 3: AI Insights
# --------------------------------------------------
with tab3:
    st.subheader("AI-Based Student Insights")

    avg_sleep = history_data["sleep_hours"].mean()
    avg_study = history_data["study_hours"].mean()
    avg_stress = history_data["stress_level"].mean()
    avg_productivity = history_data["productivity_score"].mean()

    if avg_sleep < 6:
        st.warning("Sleep duration is below the recommended level. This may reduce productivity.")
    else:
        st.success("Sleep duration is at a healthy level.")

    if avg_stress > 6:
        st.warning("Stress level is high. Recovery time and workload reduction are recommended.")
    else:
        st.success("Stress level is manageable.")

    if avg_study < 3:
        st.warning("Study consistency is low. A structured weekly plan is recommended.")
    else:
        st.success("Study consistency is acceptable.")

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
        "This is an MVP prototype. The current model uses a small demo dataset. "
        "Future versions should use real user tracking data for more reliable results."
    )

# --------------------------------------------------
# Tab 4: Future Roadmap
# --------------------------------------------------
with tab4:
    st.subheader("Future Development Roadmap")

    st.write("This section directly addresses the future development direction of the project.")

    st.markdown("""
    ### Planned Improvements

    - Smartwatch integration for sleep cycle tracking
    - Automatic exercise and activity tracking
    - Weekly and monthly productivity reports
    - Historical behavior analysis
    - Personalized AI coaching
    - Cloud deployment
    - User login system
    - Larger real-world dataset
    - Advanced ML models such as Gradient Boosting or Neural Networks
    """)

    st.subheader("Why This Matters")

    st.write(
        "The current version demonstrates the core MVP. "
        "The next version will move from manual input to automated tracking. "
        "This makes the system more realistic, scalable, and useful for long-term student support."
    )
