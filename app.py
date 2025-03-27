import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set app title and icon
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")

# App title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health metrics using machine learning.
The model was trained on the Pima Indians Diabetes Dataset.
""")

# Add sidebar with info
st.sidebar.header("About")
st.sidebar.info("""
This prediction model uses Logistic Regression algorithm to predict diabetes risk.
Enter your health metrics on the left and click 'Predict' to see results.
""")

# Add image
image = Image.open('diabetes_image.jpg')  # You can add your own image
st.image(image, caption='Diabetes Risk Assessment', use_column_width=True)

# Create input form
st.header("Enter Your Health Metrics")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of times pregnant", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose level (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin level (mu U/ml)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("Body Mass Index (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    
    submitted = st.form_submit_button("Predict Diabetes Risk")

# When form is submitted
if submitted:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)
    
    # Display results
    st.header("Prediction Results")
    
    if prediction[0] == 0:
        st.success(f"Low Risk: You are likely NOT diabetic ({(probability[0][0]*100):.2f}% probability)")
    else:
        st.error(f"High Risk: You are likely diabetic ({(probability[0][1]*100):.2f}% probability)")
    
    # Show probability meter
    st.subheader("Risk Probability")
    st.progress(int(probability[0][1] * 100))
    st.write(f"{probability[0][1]*100:.2f}% probability of having diabetes")
    
    # Interpretation
    st.subheader("Interpretation")
    if probability[0][1] < 0.3:
        st.info("Low diabetes risk - Maintain healthy lifestyle")
    elif probability[0][1] < 0.7:
        st.warning("Moderate diabetes risk - Consider consulting a doctor")
    else:
        st.error("High diabetes risk - Please consult a healthcare professional")

# Add footer
st.markdown("---")
st.markdown("""
**Note:** This app provides predictions based on a machine learning model and should not replace professional medical advice.
""")