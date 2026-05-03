import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
import numpy as np

st.set_page_config(page_title="AI Student Life Assistant", page_icon="🎓")

st.title("🎓 AI Student Life Assistant")

st.write(
    "This application analyzes a student's mood, sleep time, and study time "
    "to predict productivity and generate personalized daily suggestions."
)

# -------------------------
# Simple ML Model
# -------------------------
X = np.array([
    [4, -0.7, 1],
    [5, -0.4, 2],
    [6, 0.0, 2],
    [7, 0.3, 3],
    [8, 0.6, 4],
    [3, -0.8, 1],
    [9, 0.7, 5],
    [6, 0.2, 4],
    [4, 0.1, 1],
    [7, -0.2, 3]
])

y = [
    "Low",
    "Low",
    "Medium",
    "High",
    "High",
    "Low",
    "High",
    "Medium",
    "Low",
    "Medium"
]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# -------------------------
# User Inputs
# -------------------------
mood_text = st.text_area("How do you feel today?")

sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
study_hours = st.slider("Study Hours", 0, 10, 2)

analyzer = SentimentIntensityAnalyzer()

if st.button("Analyze"):

    if mood_text.strip() == "":
        st.warning("Please enter your mood text first.")
    else:
        score = analyzer.polarity_scores(mood_text)
        sentiment = score["compound"]

        if sentiment >= 0.05:
            mood = "Positive"
        elif sentiment <= -0.05:
            mood = "Negative"
        else:
            mood = "Neutral"

        productivity = model.predict([[sleep_hours, sentiment, study_hours]])[0]

        if productivity == "Low":
            suggestion = (
                "Take a short break, reduce workload, and focus on one small task. "
                "Use a 25-minute study session followed by a 5-minute break."
            )
            daily_plan = "25 min light study → 5 min break → 30 min review → rest"
        elif productivity == "Medium":
            suggestion = (
                "Do moderate tasks and avoid multitasking. "
                "Focus on completing one important assignment first."
            )
            daily_plan = "45 min study → 10 min break → 45 min practice"
        else:
            suggestion = (
                "Start with your most difficult task and use a deep work session. "
                "You are in a good condition for focused study."
            )
            daily_plan = "90 min deep work → 15 min break → 60 min study"

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mood", mood)
        col2.metric("Sentiment Score", round(sentiment, 2))
        col3.metric("Productivity", productivity)

        st.success(suggestion)

        st.subheader("Daily Study Plan")
        st.write(daily_plan)
