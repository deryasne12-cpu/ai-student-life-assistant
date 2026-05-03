import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="AI Student Life Assistant", page_icon="🎓")

st.title("🎓 AI Student Life Assistant")

st.write(
    "This application analyzes a student's mood, sleep time, and study time "
    "to generate a simple productivity prediction and personalized suggestion."
)

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

        if sleep_hours < 5 or sentiment < -0.2:
            productivity = "Low"
        elif sleep_hours >= 7 and study_hours >= 2 and sentiment > 0:
            productivity = "High"
        else:
            productivity = "Medium"

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
        st.write("Mood:", mood)
        st.write("Sentiment Score:", sentiment)
        st.write("Productivity Level:", productivity)
        st.write("Suggestion:", suggestion)
        st.write("Daily Plan:", daily_plan)
