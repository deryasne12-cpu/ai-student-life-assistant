import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Student Life Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Yapay Zeka Destekli Öğrenci Yaşam Asistanı")

st.write(
    "Bu uygulama, öğrencinin ruh halini, uyku süresini ve çalışma süresini analiz ederek "
    "verimlilik tahmini ve kişiselleştirilmiş çalışma önerileri üretir."
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

training_predictions = model.predict(X)
accuracy = accuracy_score(y, training_predictions)

labels = {
    0: "Düşük",
    1: "Orta",
    2: "Yüksek"
}

# --------------------------
# User Inputs
# --------------------------
mood_text = st.text_area("Bugün nasıl hissediyorsunuz?")
sleep_hours = st.slider("Uyku Saatleri", 0, 12, 6)
study_hours = st.slider("Çalışma Saatleri", 0, 10, 2)

analyzer = SentimentIntensityAnalyzer()

if st.button("Analiz et"):

    if mood_text.strip() == "":
        st.warning("Lütfen önce ruh halinizi yazın.")
    else:
        score = analyzer.polarity_scores(mood_text)
        sentiment = score["compound"]

        prediction = model.predict([[sleep_hours, sentiment, study_hours]])[0]
        productivity = labels[prediction]

        if sentiment >= 0.05:
            mood = "Olumlu"
        elif sentiment <= -0.05:
            mood = "Olumsuz"
        else:
            mood = "Nötr"

        if prediction == 0:
            suggestion = "Kısa bir mola verin, iş yükünüzü azaltın ve küçük bir göreve odaklanın."
            plan = "25 dakika hafif çalışma → 5 dakika ara → dinlenme"
        elif prediction == 1:
            suggestion = "Orta düzeyde işler yapın ve aynı anda birden fazla iş yapmaktan kaçının."
            plan = "45 dakika çalışma → 10 dakika ara → tekrar"
        else:
            suggestion = "Enerjiniz iyi görünüyor. Zor görevlerle başlayın ve derin çalışma yapın."
            plan = "60 dakika derin çalışma → 10 dakika ara → tekrar"

        st.subheader("Sonuçlar")

        col1, col2, col3 = st.columns(3)
        col1.metric("Ruh Hali", mood)
        col2.metric("Duygu Puanı", round(sentiment, 2))
        col3.metric("Verimlilik", productivity)

        st.success(suggestion)
        st.write(f"**Günlük Plan:** {plan}")

        st.subheader("📊 Analiz Grafikleri")

        # 1. Sentiment Score Chart
        fig1, ax1 = plt.subplots()
        ax1.bar(["Duygu Puanı"], [sentiment])
        ax1.set_ylim(-1, 1)
        ax1.set_title("Duygu Puanı Analizi")
        ax1.set_ylabel("Skor")
        st.pyplot(fig1)

        # 2. Sleep vs Study Chart
        fig2, ax2 = plt.subplots()
        ax2.bar(["Uyku Saatleri", "Çalışma Saatleri"], [sleep_hours, study_hours])
        ax2.set_title("Uyku ve Çalışma Karşılaştırması")
        ax2.set_ylabel("Saat")
        st.pyplot(fig2)

        # 3. Productivity Chart
        productivity_map = {
            "Düşük": 1,
            "Orta": 2,
            "Yüksek": 3
        }

        fig3, ax3 = plt.subplots()
        ax3.barh(["Verimlilik"], [productivity_map[productivity]])
        ax3.set_xlim(0, 3)
        ax3.set_title("Verimlilik Seviyesi")
        ax3.set_xlabel("Seviye")
        st.pyplot(fig3)

        st.subheader("🤖 Model Bilgisi")
        st.write(f"Model: **Decision Tree Classifier**")
        st.write(f"Eğitim doğruluğu: **{round(accuracy * 100, 2)}%**")

        feature_importance = pd.DataFrame({
            "Özellik": ["Uyku Saatleri", "Duygu Puanı", "Çalışma Saatleri"],
            "Önem": model.feature_importances_
        })

        st.subheader("📌 Özellik Önem Grafiği")
        st.bar_chart(feature_importance.set_index("Özellik"))

        st.subheader("📁 Eğitim Veri Seti")
        preview_data = data.copy()
        preview_data["productivity"] = preview_data["productivity"].map(labels)
        st.dataframe(preview_data)
