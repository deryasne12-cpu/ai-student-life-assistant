import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="AI Student Life Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Life Assistant")

st.write(
    "This AI-powered application analyzes your mood, sleep, and study habits "
    "to predict your productivity level and give personalized suggestions."
)

# --------------------------
# INPUTS
# --------------------------
mood_text = st.text_area("How do you feel today?")
sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
study_hours = st.slider("Study Hours", 0, 10, 2)

# --------------------------
# SENTIMENT ANALYSIS
# --------------------------
analyzer = SentimentIntensityAnalyzer()

# --------------------------
# SIMPLE DATASET (TRAINING)
# --------------------------
# Features: [sleep_hours, sentiment_score, study_hours]
X = np.array([
    [4, -0.6, 2],
    [5, -0.3, 3],
    [6, 0.0, 4],
    [7, 0.4, 5],
    [8, 0.7, 6],
    [3, -0.8, 1],
    [9, 0.8, 7],
    [2, -0.9, 1],
    [6, 0.2, 3],
    [7, 0.5, 6]
])

# Labels: 0 = Low, 1 = Medium, 2 = High
y = np.array([0, 0, 1, 1, 2, 0, 2, 0, 1, 2])

# --------------------------
# TRAIN MODEL
# --------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# --------------------------
# BUTTON ACTION
# --------------------------
if st.button("Analyze"):

    if mood_text.strip() == "":
        st.warning("Please enter your mood text first.")
    else:
        score = analyzer.polarity_scores(mood_text)
        sentiment = score["compound"]

        # Prediction
        prediction = model.predict([[sleep_hours, sentiment, study_hours]])[0]

        if prediction == 0:
            productivity = "Low"
            suggestion = "Take a break, reduce workload, and focus on one small task."
            plan = "25 min light study → 5 min break → rest"
        elif prediction == 1:
            productivity = "Medium"
            suggestion = "Maintain steady work and avoid distractions."
            plan = "45 min study → 10 min break → repeat"
        else:
            productivity = "High"
            suggestion = "Great energy! Start with difficult tasks and use deep work."
            plan = "60 min deep work → 10 min break → repeat"

        # Mood Label
        if sentiment >= 0.05:
            mood = "Positive"
        elif sentiment <= -0.05:
            mood = "Negative"
        else:
            mood = "Neutral"

        # --------------------------
        # OUTPUT
        # --------------------------
        st.subheader("Results")

        st.write(f"**Mood:** {mood}")
        st.write(f"**Sentiment Score:** {round(sentiment, 4)}")
        st.write(f"**Productivity Level:** {productivity}")
        st.write(f"**Suggestion:** {suggestion}")
        st.write(f"**Daily Plan:** {plan}")
