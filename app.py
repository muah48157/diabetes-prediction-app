import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
import base64
import tempfile
from io import BytesIO
import time

# Performance optimization: Cache everything possible
@st.cache_resource(show_spinner=False)
def load_model():
    """Cache the model and scaler to avoid reloading"""
    try:
        start = time.time()
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        st.session_state.load_time = time.time() - start
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize session variables
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.scaler = load_model()

# Precompute PDF report template
@st.cache_data
def get_pdf_template():
    pdf = FPDF()
    pdf.add_page()
    return pdf

# App configuration - do this first
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered",  # Better performance than "wide"
)

# Lightweight UI rendering
def show_loading_placeholder():
    """Show loading animation while heavy computations run"""
    with st.spinner('Analyzing your data...'):
        time.sleep(0.5)  # Simulate processing

# Main app function
def main():
    st.title("Diabetes Risk Assessment")
    
    # Prediction form - simplified UI
    with st.form("prediction_form"):
        st.subheader("Health Metrics")
        
        # Use sliders with defaults for faster interaction
        glucose = st.slider("Glucose (mg/dL)", 50, 200, 100)
        bmi = st.slider("BMI", 18.0, 40.0, 25.0, 0.1)
        age = st.slider("Age", 20, 80, 30)
        
        # Secondary metrics in expander
        with st.expander("Additional Metrics"):
            pregnancies = st.slider("Pregnancies", 0, 10, 0)
            bp = st.slider("Blood Pressure", 60, 120, 80)
            skin = st.slider("Skin Thickness", 0, 50, 20)
            insulin = st.slider("Insulin", 0, 300, 80)
            pedigree = st.slider("Pedigree", 0.0, 2.0, 0.5, 0.01)
        
        submitted = st.form_submit_button("Assess Risk")

    # Prediction logic
    if submitted and st.session_state.model:
        show_loading_placeholder()
        
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [bp],
                'SkinThickness': [skin],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [pedigree],
                'Age': [age]
            })
            
            # Scale and predict
            scaled_data = st.session_state.scaler.transform(input_data)
            prediction = st.session_state.model.predict(scaled_data)
            probability = st.session_state.model.predict_proba(scaled_data)
            
            # Display results
            st.subheader("Results")
            risk_level = "High" if prediction[0] == 1 else "Low"
            st.metric("Risk Level", f"{risk_level} ({probability[0][1]*100:.1f}%)")
            
            # PDF report generation on demand
            if st.button("Generate PDF Report"):
                with st.spinner('Creating report...'):
                    pdf = get_pdf_template()
                    # ... (PDF generation code from previous version)
                    
                    # Save to bytes buffer instead of temp file
                    pdf_bytes = BytesIO()
                    pdf.output(pdf_bytes)
                    pdf_bytes.seek(0)
                    
                    st.download_button(
                        label="Download Report",
                        data=pdf_bytes,
                        file_name="diabetes_report.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
