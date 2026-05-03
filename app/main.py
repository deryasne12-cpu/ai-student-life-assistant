import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="AI Student Life Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Life Assistant")

st.write(
    "This AI-powered application analyzes mood, sleep, and study habits "
    "to predict productivity and generate personalized study suggestions."
)

# --------------------------
# Training Dataset
# --------------------------
data = pd.DataFrame({
    "sleep_hours": [4, 5, 6, 7, 8, 3, 9, 2, 6, 7, 5, 8],
    "sentiment_score": [-0.6, -0.3, 0.0, 0.4, 0.7, -0.8, 0.8, -0.9, 0.2, 0.5, -0.1, 0.6],
    "study_hours": [2, 3, 4, 5, 6, 1, 7, 1, 3, 6, 2, 5],
    "productivity": [0, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 2]
})

X = data[["sleep_hours", "sentiment_score", "study_hours"]]
y = data["productivity"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# --------------------------
# User Inputs
# --------------------------
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

        prediction = model.predict([[sleep_hours, sentiment, study_hours]])[0]
        productivity = labels[prediction]

        if sentiment >= 0.05:
            mood = "Positive"
        elif sentiment <= -0.05:
            mood = "Negative"
        else:
            mood = "Neutral"

        if prediction == 0:
            suggestion = "Take a break, reduce workload, and focus on one small task."
            plan = "25 min light study → 5 min break → rest"
        elif prediction == 1:
            suggestion = "Maintain steady work and avoid distractions."
            plan = "45 min study → 10 min break → repeat"
        else:
            suggestion = "Great energy! Start with difficult tasks and use deep work."
            plan = "60 min deep work → 10 min break → repeat"

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mood", mood)
        col2.metric("Sentiment", round(sentiment, 2))
        col3.metric("Productivity", productivity)

        st.success(suggestion)

        st.subheader("Daily Study Plan")
        st.write(plan)

        st.subheader("Model Information")
        st.write(f"Training Accuracy: **{round(accuracy * 100, 2)}%**")

        feature_importance = pd.DataFrame({
            "Feature": ["Sleep Hours", "Sentiment Score", "Study Hours"],
            "Importance": model.feature_importances_
        })

        st.subheader("Feature Importance")
        st.bar_chart(feature_importance.set_index("Feature"))

        st.subheader("Training Dataset Preview")
        preview_data = data.copy()
        preview_data["productivity"] = preview_data["productivity"].map(labels)
        st.dataframe(preview_data)
