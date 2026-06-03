# 🎓 AI Student Life Assistant

An AI-powered system that analyzes a student's mood, sleep habits, and study time to predict productivity and generate personalized study recommendations.

---

## 🚀 Live Demo
[https://YOUR-LINK.streamlit.app](https://ai-student-life-assistant-jdoywjyewvfejkqgmrcxgc.streamlit.app/)

## 🚀 Overview

This project integrates **Natural Language Processing (NLP)** and **Machine Learning (ML)** to help students better manage their productivity and daily routines.

The system takes user input (mood text, sleep hours, study hours), processes it through sentiment analysis, and predicts productivity levels using a trained ML model.

---

## 🎯 Objectives

- Improve student productivity awareness  
- Provide AI-driven recommendations  
- Combine NLP + ML in a real-world scenario  
- Build an interactive web application  

---

## 🧠 System Architecture

User Input (Mood + Sleep + Study)  
→ NLP (VADER Sentiment Analysis)  
→ Feature Vector [sleep, sentiment, study]  
→ Decision Tree Model  
→ Productivity Prediction  
→ Suggestions & Daily Plan  

---

## ⚙️ Technologies Used

- **Python**
- **Streamlit** (User Interface)
- **VADER Sentiment Analysis (NLP)**
- **Scikit-learn (Machine Learning)**
- **NumPy**
- **Git & GitHub**

---

## 🤖 Machine Learning Model

The project uses a **Decision Tree Classifier**.

### 📌 Input Features:
- Sleep hours  
- Sentiment score (calculated from text)  
- Study hours  

### 📌 Output Classes:
- Low Productivity  
- Medium Productivity  
- High Productivity  

### 📌 Why Decision Tree?
- Easy to interpret  
- Suitable for small datasets  
- Fast and efficient for real-time prediction  

---

## 💻 Installation & Run

Open in browser:

http://localhost:8501

To open localhost, please use a terminal connection. I'm sharing the code below:

cd ai-student-life-assistant
python -m streamlit run app/main.py  

🧪 Example Scenarios
🔴 Scenario 1 – Low Productivity

Input:

Mood: "I feel tired and stressed"
Sleep: 4
Study: 1

Output:

Productivity: Low
Suggestion: Take a break and do light study

🟡 Scenario 2 – Medium Productivity

Input:

Mood: "I feel okay but not very focused"
Sleep: 6
Study: 3

Output:

Productivity: Medium
Suggestion: Maintain steady work
🟢 Scenario 3 – High Productivity

Input:

Mood: "I feel motivated and focused"
Sleep: 8
Study: 4

Output:

Productivity: High
Suggestion: Start deep work

📁 Project Structure

ai-student-life-assistant/
│
├── app/
│   └── main.py
│
├── requirements.txt
├── README.md

🎯 Key Features
Mood detection using NLP
Sentiment scoring
ML-based productivity prediction
Personalized recommendations
Daily study plan generation
Interactive UI with Streamlit
🔍 Future Improvements
Use real dataset instead of synthetic data
Add more ML models (Random Forest, SVM)
Track user history
Add charts and analytics
Deploy to cloud
💸 Cost Analysis

This project uses only open-source tools.

Estimated cost: 0€

👥 Team Members
Derya Süne – Project Manager
Zamonbek Saydullaev – AI/NLP Developer
Lazizbek Khudashkurov – Machine Learning Engineer
Abdullah Büyükdeniz – AI/NLP Developer
Ömer Kesen – Backend & Frontend Developer


```bash
pip install -r requirements.txt
python -m streamlit run app/main.py
