import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.set_page_config(page_title="Driver Risk Assessment", page_icon="🚗")

st.title("🚗 Driver Risk Assessment App")
st.write(
    "Adjust the driving parameters below to predict the driver's risk tier."
)


# 1. Train model automatically inside the app
@st.cache_resource
def load_trained_model():
  np.random.seed(42)
  n_samples = 1200
  brakes = np.random.randint(0, 10, n_samples)
  speeding = np.random.uniform(0, 50, n_samples)
  turns = np.random.randint(0, 8, n_samples)

  score = np.clip(100 - (brakes * 4 + turns * 3 + speeding * 0.8), 0, 100)
  risk = np.where(score > 75, 0, np.where(score > 50, 1, 2))

  df = pd.DataFrame({
      "harsh_brakes": brakes,
      "sharp_turns": turns,
      "speeding_percent": speeding,
      "risk_level": risk,
  })

  X = df[["harsh_brakes", "sharp_turns", "speeding_percent"]]
  y = df["risk_level"]

  clf = RandomForestClassifier(n_estimators=100, random_state=42)
  clf.fit(X, y)
  return clf


model = load_trained_model()

# 2. Sliders for user input
st.subheader("Trip Telemetry Inputs")
brakes_input = st.slider(
    "Harsh Braking Events (per trip)", min_value=0, max_value=12, value=2
)
turns_input = st.slider(
    "Aggressive Turns (per trip)", min_value=0, max_value=10, value=1
)
speeding_input = st.slider(
    "Time Spent Speeding (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5
)

# 3. Predict on Button Click
if st.button("Calculate Risk Tier"):
  sample = pd.DataFrame([{
      "harsh_brakes": brakes_input,
      "sharp_turns": turns_input,
      "speeding_percent": speeding_input,
  }])

  pred = model.predict(sample)[0]
  probs = model.predict_proba(sample)[0]

  labels = {
      0: ("Low Risk (Safe Driver)", "🟢 Safe"),
      1: ("Medium Risk (Moderate Driver)", "🟡 Normal"),
      2: ("High Risk (Aggressive Driver)", "🔴 Dangerous"),
  }

  status_text, badge = labels[pred]

  st.divider()
  st.subheader("Result")
  st.markdown(f"### Assessed Profile: **{status_text}**")

  st.write("**Model Probabilities:**")
  col1, col2, col3 = st.columns(3)
  col1.metric("Safe", f"{probs[0]*100:.1f}%")
  col2.metric("Moderate", f"{probs[1]*100:.1f}%")
  col3.metric("Dangerous", f"{probs[2]*100:.1f}%")