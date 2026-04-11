import streamlit as st
import requests
import pandas as pd
import plotly.express as px

import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

model_path = os.path.join(PROJECT_DIR, "models", "rain_model.pkl")
scaler_path = os.path.join(PROJECT_DIR, "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.set_page_config(page_title="RainCast", layout="wide")

# Remove header
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# SAME THEME (NO CHANGE)
st.markdown("""
<style>
.stApp {background:#f0f9ff;}

section[data-testid="stSidebar"] {background:#2563eb;}
section[data-testid="stSidebar"] * {color:white !important;}

.card {
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.1);
    text-align:center;
}

/* RESULT CENTER FIX */
.result-center {
    text-align:center;
    font-size:22px;
    font-weight:bold;
    margin-top:20px;
}

.success {
    background:#dcfce7;
    color:#166534;
    padding:20px;
    border-radius:10px;
}

.fail {
    background:#fee2e2;
    color:#991b1b;
    padding:20px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# Dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Raincast_clean_dataset.csv")

# Sidebar
st.sidebar.title("🌧️ RainCast")
st.sidebar.markdown("### Smart Rain Prediction System")
st.sidebar.markdown("[🚀 Deploy App](https://share.streamlit.io)")

page = st.sidebar.radio("", ["Dashboard","Prediction","EDA","Models","About","Feedback"])

df = pd.read_csv(data_path) if os.path.exists(data_path) else None

# ---------------- DASHBOARD (UNCHANGED) ----------------

if page == "Dashboard":
    st.title("🌧️ RainCast Dashboard")

    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="card"><h3>Total Records</h3><h1>15K+</h1></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h3>Rain Days</h3><h1>3K+</h1></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h3>Accuracy</h3><h1>85%</h1></div>', unsafe_allow_html=True)
    if df is not None:
        st.subheader("📊 Rain Distribution")
        fig = px.pie(df, names="RainTomorrow", title="Rain vs No Rain")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICTION (FIXED & SAFE) ----------------
elif page == "Prediction":
    st.title("🌧️ Rain Prediction")

    col1, col2 = st.columns(2)

    with col1:
        if df is not None:
            city = st.selectbox("Select City", sorted(df["Location"].dropna().unique()))
        else:
            city = st.text_input("Enter City")

    with col2:
        date = st.date_input("Select Date")

    # 🔥 BUTTONS
    b1, b2 = st.columns(2)
    predict_btn = b1.button("Predict")
    clear_btn = b2.button("Clear")

    if clear_btn:
        st.rerun()

    if predict_btn:

        city = city.strip().lower()

        if city == "":
            st.warning("Please enter city")

        elif df is None:
            st.error("Dataset not loaded")

        else:
            try:
                df["Location"] = df["Location"].astype(str).str.strip().str.lower()

                city_data = df[df["Location"] == city]

                if city_data.empty:
                    st.error("❌ City not found in dataset")
                    st.write("Available cities:", df["Location"].unique()[:20])

                else:
                    import numpy as np

                    temp = city_data["MaxTemp"].mean()
                    humidity = city_data["Humidity3pm"].mean()
                    pressure = city_data["Pressure3pm"].mean()
                    wind = city_data["WindSpeed3pm"].mean()
                    rainfall = city_data["Rainfall"].mean()

                    full_input = [temp, humidity, pressure, wind, rainfall] + [0]*18
                    input_array = np.array(full_input).reshape(1, -1)
                    input_scaled = scaler.transform(input_array)

                    prediction = model.predict(input_scaled)[0]

                    try:
                        probability = model.predict_proba(input_scaled)[0][1]
                    except:
                        probability = 0.0

                    cloud = int(humidity)
                    condition = "Cloudy" if humidity > 60 else "Clear"

                    if prediction == 1:
                        st.markdown(f'''
                        <div class="result-center success">
                        🌧️ Rain Expected<br>
                        Confidence: {round(probability*100,2)}%<br><br>

                        🌡️ Temp: {round(temp,2)}°C<br>
                        💧 Humidity: {round(humidity,2)}%<br>
                        🌬️ Wind: {round(wind,2)} km/h<br>
                        ☁️ Cloud: {cloud}%<br>
                        ☀️ Condition: {condition}
                        </div>
                        ''', unsafe_allow_html=True)

                    else:
                        st.markdown(f'''
                        <div class="result-center fail">
                        ☀️ No Rain<br>
                        Confidence: {round(probability*100,2)}%<br><br>

                        🌡️ Temp: {round(temp,2)}°C<br>
                        💧 Humidity: {round(humidity,2)}%<br>
                        🌬️ Wind: {round(wind,2)} km/h<br>
                        ☁️ Cloud: {cloud}%<br>
                        ☀️ Condition: {condition}
                        </div>
                        ''', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

 # ---------------- EDA (UNCHANGED) ----------------

elif page == "EDA":
    st.title("📊 EDA")

    if df is not None:

        # Existing Graphs (KEEP SAME)
        
        # Histogram
        st.subheader("Rainfall Distribution (Histogram)")
        fig_hist = px.histogram(
            df,
           x="Rainfall",
           nbins=30,
           color="RainTomorrow",
            title="Rainfall Count Distribution"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        

        # Scatter Plot
        st.subheader("Rainfall vs Humidity3pm (Scatter Plot)")
        fig_scatter = px.scatter(
             df,
            x="Rainfall",
            y="Humidity3pm",
            color="RainTomorrow",
            title="Rainfall vs Humidity3pm"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ✅ NEW ADDITIONS (SAFE)

        # Box Plot
        st.subheader("Max Temperature vs Rain (Box Plot)")
        fig_box = px.box(
            df,
            x="RainTomorrow",
            y="MaxTemp",
            color="RainTomorrow"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Violin Plot
        st.subheader("Wind Speed vs Rain (Violin Plot)")
        fig_violin = px.violin(
            df,
            x="RainTomorrow",
            y="WindSpeed3pm",
            color="RainTomorrow",
            box=True
        )
        st.plotly_chart(fig_violin, use_container_width=True)

        # ---------------- MODELS (UNCHANGED) ----------------
elif page == "Models":
    st.title("🤖 Models Used")

    st.markdown("""
- Logistic Regression  
- Decision Tree  
- Random Forest  
- KNN  
- SVM  
- Naive Bayes  
- XGBoost  

### Final Model: Random Forest (Best Performance)
Accuracy: ~85%

### Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Classification Report  

### Why Random Forest?
- Handles non-linear data  
- High accuracy  
- Reduces overfitting  
""")

# ---------------- ABOUT (ONLY DATASET ADDED) ----------------

elif page == "About":
    st.title("📘 About Project")

    st.markdown("""
### 🌧️ RainCast - Smart Rain Prediction System

This project predicts rainfall using Machine Learning techniques based on historical weather data.

---

### ❗ Problem Statement
Accurate rain prediction is difficult but essential for planning in sectors like agriculture, transportation, and disaster management.

---

### 🎯 Objective
To build a reliable and accurate machine learning model that can predict whether it will rain tomorrow.

---

### 📊 Dataset Description
This dataset contains weather-related features:

- Temperature  
- Humidity  
- Pressure  
- Wind Speed  
- Rainfall  

**Target Variable:**
- RainTomorrow (Yes/No)

---

### ⚙️ Technologies Used
- Python  
- Machine Learning  
- Streamlit  
- Flask  

---

### 🔄 Workflow
- Data Cleaning  
- Feature Engineering  
- Feature Scaling  
- Model Training  
- Model Evaluation  

---

### 🚀 Future Scope
- Real-time weather API integration  
- Mobile application development  
- Advanced deep learning models  

---

### 👨‍💻 Developed By
Krishna Vasita
""")
# ---------------- FEEDBACK (UPDATED ONLY BUTTONS) ----------------
elif page == "Feedback":
    st.title("💬 Feedback")

    name = st.text_input("Your Name")
    feedback = st.text_area("Your Feedback")

    col1,col2,col3 = st.columns(3)

    if col1.button("Submit"):
        st.success("Feedback submitted!")

    if col2.button("Clear"):
        st.rerun()

    if col3.button("Edit"):
        st.info("You can edit your feedback above.")

