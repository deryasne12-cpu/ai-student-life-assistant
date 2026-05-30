import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

st.set_page_config(
    page_title="AI Student Life Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Life Assistant")

st.info(
    "Analyze mood, sleep habits, and study hours using NLP and Machine Learning "
    "to predict productivity and generate personalized recommendations."
)

with st.sidebar:
    st.header("Project Information")
    st.write("**AI Student Life Assistant**")
    st.write("Technologies:")
    st.write("- NLP: VADER Sentiment Analysis")
    st.write("- ML: Decision Tree Classifier")
    st.write("- UI: Streamlit")
    st.write("- Data: Pandas")
    st.write("Version: 1.0")

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

accuracy = accuracy_score(y, model.predict(X))

labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

st.subheader("Student Input")

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

        input_data = pd.DataFrame([{
            "sleep_hours": sleep_hours,
            "sentiment_score": sentiment,
            "study_hours": study_hours
        }])

        prediction = model.predict(input_data)[0]
        productivity = labels[prediction]

        if sentiment >= 0.05:
            mood = "Positive"
        elif sentiment <= -0.05:
            mood = "Negative"
        else:
            mood = "Neutral"

        if prediction == 0:
            suggestion = "Take a short break, reduce workload, and focus on one small task."
            plan = "25 min light study → 5 min break → rest"
        elif prediction == 1:
            suggestion = "Work at a moderate pace and avoid distractions."
            plan = "45 min study → 10 min break → repeat"
        else:
            suggestion = "Great energy level. Start with difficult tasks and use deep work."
            plan = "60 min deep work → 10 min break → repeat"

        st.divider()
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mood", mood)
        col2.metric("Sentiment Score", round(sentiment, 2))
        col3.metric("Productivity", productivity)

        st.success(suggestion)
        st.write(f"**Daily Plan:** {plan}")

        st.divider()
        st.subheader("Charts")

        st.write("Sleep, Study, and Sentiment Overview")
        overview_chart = pd.DataFrame(
            {"Value": [sleep_hours, study_hours, sentiment]},
            index=["Sleep Hours", "Study Hours", "Sentiment Score"]
        )
        st.bar_chart(overview_chart)

        st.write("Productivity Level Chart")
        productivity_chart = pd.DataFrame(
            {"Level": [prediction + 1]},
            index=[productivity]
        )
        st.bar_chart(productivity_chart)

        st.write("Feature Importance")
        importance_chart = pd.DataFrame(
            {"Importance": model.feature_importances_},
            index=["Sleep Hours", "Sentiment Score", "Study Hours"]
        )
        st.bar_chart(importance_chart)

        st.divider()
        st.subheader("Model Information")

        st.write("Model: **Decision Tree Classifier**")
        st.metric("Model Accuracy", f"{round(accuracy * 100, 2)}%")
        st.caption(
            "The model uses a small sample dataset for demonstration purposes. "
            "The accuracy value is calculated on the training dataset."
        )

        st.subheader("Training Dataset")
        st.caption("Sample training dataset used for demonstration purposes.")

        preview_data = data.copy()
        preview_data["productivity"] = preview_data["productivity"].map(labels)
        st.dataframe(preview_data)
