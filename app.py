# gradio app for Diabetes Prediction

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the trained model (pipeline)
with open("diabetes_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Prediction Logic Function
def predict_diabetes(
    gender, age, hypertension, heart_disease,
    smoking_history, bmi, hba1c_level, blood_glucose_level
):
    # Pack inputs into DataFrame
    # Column names must exactly match training CSV
    input_df = pd.DataFrame([[
        gender, age, hypertension, heart_disease,
        smoking_history, bmi, hba1c_level, blood_glucose_level
    ]],
    columns=[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        return f"Diabetes Detected (Risk Probability: {probability:.2f})"
    else:
        return f"No Diabetes Detected (Risk Probability: {probability:.2f})"

# 3. Gradio App Interface
inputs = [
    gr.Radio(["Male", "Female", "Other"], label="Gender"),
    gr.Number(label="Age", value=40),
    gr.Radio([0, 1], label="Hypertension (0 = No, 1 = Yes)"),
    gr.Radio([0, 1], label="Heart Disease (0 = No, 1 = Yes)"),
    gr.Dropdown(
        ["never", "former", "current", "not current", "ever"],
        label="Smoking History"
    ),
    gr.Number(label="BMI"),
    gr.Number(label="HbA1c Level"),
    gr.Number(label="Blood Glucose Level")
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction System"
)

app.launch(share=True)
