import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.title("Weather Event Prediction App")

# Upload model
uploaded_model = st.sidebar.file_uploader("Upload your model.pkl", type="pkl")
if uploaded_model:
    models = joblib.load(uploaded_model)
    st.sidebar.success(f"Loaded {len(models)} models")
    st.sidebar.write("Available weather events:", list(models.keys()))
else:
    st.warning("Please upload model.pkl")
    models = None

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type="csv")
if uploaded_file and models:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(data.head())

    # Extract year
    if "Date/Time" not in data.columns:
        st.error("No 'Date/Time' column found in CSV")
    else:
        data['Date/Time'] = pd.to_datetime(data['Date/Time'], errors='coerce')
        data.dropna(subset=['Date/Time'], inplace=True)
        data['Year'] = data['Date/Time'].dt.year

        # Predict single event
        event = st.selectbox("Select Weather Event", list(models.keys()))

        if st.button("Predict for 2025"):
            model = models[event]
            pred_2025 = int(model.predict(np.array([[2025]]))[0])
            st.write(f"Predicted count for **{event}** in 2025: **{pred_2025}**")

        # Top 5 predicted events
        if st.button("Show Top 5 Predicted Weather Events for 2025"):
            summary = []
            for event_name, model in models.items():
                pred = int(model.predict(np.array([[2025]]))[0])
                summary.append([event_name, pred])

            summary_df = pd.DataFrame(summary, columns=["Weather Event", "Predicted Count"])
            top5_df = summary_df.sort_values(by="Predicted Count", ascending=False).head(5)

            st.write("### Top 5 Predicted Weather Events in 2025")
            st.dataframe(top5_df)

            # Plot bar chart
            plt.figure(figsize=(8,5))
            sns.barplot(x="Predicted Count", y="Weather Event", data=top5_df, palette="viridis")
            plt.title("Top 5 Predicted Weather Events in 2025")
            st.pyplot(plt)
else:
    st.write("Upload both CSV dataset and model.pkl to enable predictions.")
