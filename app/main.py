import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

st.set_page_config(
    page_title="AI Student Performance Tracker",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Student Performance & Wellness Tracker")

st.info(
    "This MVP tracks daily student inputs, analyzes mood with NLP, predicts productivity "
    "with Machine Learning, and generates personalized study recommendations."
)

with st.sidebar:
    st.header("📌 Project Scope")
    st.write("This version uses manual daily input.")
    st.write("Future versions can integrate:")
    st.write("- Smartwatch sleep tracking")
    st.write("- Exercise tracking")
    st.write("- Study history")
    st.write("- Long-term behavior analytics")

    st.divider()

    st.header("🛠 Technologies")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- VADER NLP")
    st.write("- Scikit-learn")
    st.write("- Pandas")
    st.write("- Random Forest")

data = pd.DataFrame({
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

X = data[features]
y = data["productivity"]

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

accuracy = accuracy_score(y, model.predict(X))

labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

analyzer = SentimentIntensityAnalyzer()

st.subheader("🧾 Daily Student Tracker")

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

if st.button("Analyze Student Performance"):

    if mood_text.strip() == "":
        st.warning("Please enter a mood text first.")
    else:
        score = analyzer.polarity_scores(mood_text)
        sentiment = score["compound"]

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
            (
                (sleep_hours / 12) * 25 +
                (focus_level / 10) * 25 +
                ((10 - stress_level) / 10) * 25 +
                min(exercise_minutes / 60, 1) * 25
            ),
            2
        )

        productivity_score = round(
            (
                (study_hours / 10) * 25 +
                (task_completion / 100) * 35 +
                (focus_level / 10) * 25 +
                max(sentiment, 0) * 15
            ),
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

        if prediction == 0:
            recommendation = (
                "Reduce workload, take a short break, and focus on one small task first."
            )
            plan = "25 min light study → 5 min break → 20 min review"
        elif prediction == 1:
            recommendation = (
                "Continue with moderate tasks and avoid multitasking."
            )
            plan = "45 min study → 10 min break → 45 min practice"
        else:
            recommendation = (
                "You are in a strong productivity state. Start with the most difficult task."
            )
            plan = "60 min deep work → 10 min break → 60 min focused study"

        st.divider()
        st.subheader("📊 AI Analysis Dashboard")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mood", mood)
        m2.metric("Productivity", productivity)
        m3.metric("Wellness Score", f"{wellness_score}/100")
        m4.metric("Risk Level", risk_level)

        st.success(recommendation)
        st.write(f"**Suggested Daily Plan:** {plan}")

        st.divider()
        st.subheader("📈 Tracking Visualization")

        tracking_data = pd.DataFrame({
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

        st.bar_chart(tracking_data)

        score_data = pd.DataFrame({
            "Score": [productivity_score, wellness_score]
        }, index=["Productivity Score", "Wellness Score"])

        st.subheader("Performance Scores")
        st.bar_chart(score_data)

        importance_data = pd.DataFrame({
            "Importance": model.feature_importances_
        }, index=features).sort_values("Importance", ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(importance_data)

        st.divider()
        st.subheader("🧠 Model Information")

        st.write("Model: **Random Forest Classifier**")
        st.metric("Training Accuracy", f"{round(accuracy * 100, 2)}%")
        st.caption(
            "This is an MVP prototype. The current version uses manual user input. "
            "Future versions can use real tracking data from smartwatches, sleep trackers, "
            "exercise apps, and long-term behavior history."
        )

        st.subheader("Training Dataset Preview")
        preview_data = data.copy()
        preview_data["productivity"] = preview_data["productivity"].map(labels)
        st.dataframe(preview_data)

        st.divider()
        st.subheader("🚀 Future Development Roadmap")

        st.write("- Smartwatch integration for sleep cycle tracking")
        st.write("- Automatic exercise and activity tracking")
        st.write("- Weekly productivity reports")
        st.write("- Long-term student behavior analysis")
        st.write("- Personalized AI coaching system")
        st.write("- Cloud deployment and user accounts")
