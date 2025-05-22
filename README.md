# 🌾 Weather-Based Crop Recommendation System

A smart decision support system that recommends the best crops to cultivate based on future weather predictions and soil type. Designed to assist farmers and agricultural planners in maximizing yield and minimizing crop failure risks.

---

## 🔍 Project Overview

This project leverages **XGBoost Regression** to forecast weather parameters (temperature, humidity, and rainfall) for the next 300 days based on historical Hyderabad weather data. These predictions are combined with **soil type input** and **growth duration of crops** to recommend the most suitable crops using a **weighted scoring system**.

---

## 📌 Features

- 📈 Forecasts temperature, humidity, and rainfall using XGBoost
- 🌱 Accepts soil type input from user
- ⏳ Considers crop growth duration to ensure seasonal suitability
- 📊 Recommends top 3 best-fit crops with percentage scores
- 🧠 Streamlit-powered interactive UI for easy access and analysis

---

## 🧠 How It Works

### 1. **Weather Prediction (Jupyter Notebook)**
- Model: XGBoost Regression
- Input: Past weather data
- Output: 300-day forecast for temperature, humidity, and rainfall

### 2. **Crop Recommendation Engine (VS Code / Streamlit App)**
- Takes predicted weather, user-selected soil type, and crop growth duration
- Compares with a crop suitability database
- Calculates weighted scores and ranks top 3 crops

---

## 🛠️ Tech Stack

- 💻 Python
- 📦 XGBoost
- 📊 Pandas, NumPy, Scikit-learn
- 🌐 Streamlit
- 📓 Jupyter Notebook
- 🧪 Matplotlib / Seaborn (for visualization)

---

## 🚀 Run Locally

### 📁 Clone the Repository

```bash
git clone https://github.com/raghav-1504/Crop-Recommendation-System.git
cd Crop-Recommendation-System
🧪 Install Requirements
💡 Run the Streamlit App
📁 Project Structure
bash
Copy
Edit
weather-crop-recommendation/
│
├── CRS1.ipynb          # Jupyter Notebook for XGBoost weather prediction
├── app.py                     # Streamlit app for crop recommendation
├── data/                           # Dataset folder (soil types, crop info, etc.)
└── README.md                       # You're here!
🙋‍♂️ Author
Raghav
B.Tech in AI & Data Science
KL University, Hyderabad
📧 raghavchekuri@gmail.com
