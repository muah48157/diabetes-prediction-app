import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import tempfile
import os

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
local_css("style.css")  # We'll create this file next

# App header with theme toggle
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🩺 Diabetes Risk Assessment")
with col2:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Apply theme
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #1e1e1e;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "About"])

with tab1:
    # Prediction form
    with st.form("prediction_form"):
        st.subheader("Enter Your Health Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.slider("Number of times pregnant", 0, 20, 1)
            glucose = st.slider("Glucose level (mg/dL)", 0, 300, 100)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 150, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        
        with col2:
            insulin = st.slider("Insulin level (mu U/ml)", 0, 1000, 80)
            bmi = st.slider("Body Mass Index (kg/m²)", 0.0, 70.0, 25.0, 0.1)
            diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
            age = st.slider("Age (years)", 1, 120, 30)
        
        submitted = st.form_submit_button("Predict Diabetes Risk", use_container_width=True)

    # Prediction logic
    if submitted and model is not None:
        try:
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
            
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)
            
            # Display results in an expandable section
            with st.expander("See Prediction Results", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gauge chart
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.barh(['Risk'], [probability[0][1]*100], color='#FF4B4B' if prediction[0] == 1 else '#2ECC71')
                    ax.set_xlim(0, 100)
                    ax.set_title('Diabetes Risk Score')
                    st.pyplot(fig)
                    
                with col2:
                    if prediction[0] == 0:
                        st.success(f"✅ Low Risk ({(probability[0][0]*100):.1f}%)")
                    else:
                        st.error(f"⚠️ High Risk ({(probability[0][1]*100):.1f}%)")
                    
                    st.metric("Probability", f"{probability[0][1]*100:.1f}%")
                
                # Interpretation
                st.subheader("Recommendations")
                if probability[0][1] < 0.3:
                    st.info("""
                    **Low Risk** - Maintain your healthy lifestyle:
                    - Regular exercise
                    - Balanced diet
                    - Annual checkups
                    """)
                elif probability[0][1] < 0.7:
                    st.warning("""
                    **Moderate Risk** - Consider consulting a doctor:
                    - Monitor glucose levels
                    - Improve diet and exercise
                    - Regular health screenings
                    """)
                else:
                    st.error("""
                    **High Risk** - Please consult a healthcare professional:
                    - Immediate medical consultation recommended
                    - Lifestyle changes advised
                    - Regular monitoring needed
                    """)
            
            # Generate PDF report
            def create_pdf_report(input_data, prediction, probability):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Title
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="Diabetes Risk Assessment Report", ln=1, align='C')
                pdf.ln(10)
                
                # Patient Data
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Patient Data", ln=1)
                pdf.set_font("Arial", size=12)
                
                for col, value in input_data.items():
                    pdf.cell(200, 10, txt=f"{col}: {value[0]}", ln=1)
                
                # Results
                pdf.ln(10)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Assessment Results", ln=1)
                pdf.set_font("Arial", size=12)
                
                risk = "High Risk" if prediction[0] == 1 else "Low Risk"
                pdf.cell(200, 10, txt=f"Risk Level: {risk}", ln=1)
                pdf.cell(200, 10, txt=f"Probability of Diabetes: {probability[0][1]*100:.1f}%", ln=1)
                
                # Recommendations
                pdf.ln(10)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Recommendations", ln=1)
                pdf.set_font("Arial", size=12)
                
                if probability[0][1] < 0.3:
                    pdf.multi_cell(200, 10, txt="Maintain your healthy lifestyle with regular exercise, balanced diet, and annual checkups.")
                elif probability[0][1] < 0.7:
                    pdf.multi_cell(200, 10, txt="Consider consulting a doctor. Monitor glucose levels and improve diet and exercise.")
                else:
                    pdf.multi_cell(200, 10, txt="Please consult a healthcare professional immediately. Lifestyle changes and regular monitoring are strongly advised.")
                
                return pdf
            
            # Download button for PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
                pdf = create_pdf_report(input_data, prediction, probability)
                pdf.output(tmpfile.name)
                
                with open(tmpfile.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                
                st.download_button(
                    label="📄 Download Full Report (PDF)",
                    data=f,
                    file_name="diabetes_risk_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with tab2:
    # Data analysis section
    st.subheader("Data Insights")
    
    try:
        df = pd.read_csv('diabetes2.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Cases", len(df))
            st.metric("Diabetic Cases", df['Outcome'].sum())
            st.metric("Non-Diabetic Cases", len(df) - df['Outcome'].sum())
        
        with col2:
            st.write("**Dataset Overview**")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Interactive visualization
        st.subheader("Interactive Analysis")
        x_axis = st.selectbox("X-axis", df.columns[:-1])
        y_axis = st.selectbox("Y-axis", df.columns[:-1])
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            x=df[x_axis],
            y=df[y_axis],
            c=df['Outcome'],
            alpha=0.7,
            cmap='viridis'
        )
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{x_axis} vs {y_axis}")
        plt.colorbar(scatter, label='Diabetes Outcome')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

with tab3:
    # About section
    st.subheader("About This App")
    
    st.markdown("""
    ### Diabetes Risk Assessment Tool
    
    This application predicts the likelihood of diabetes based on health metrics using machine learning.
    The model was trained on the Pima Indians Diabetes Dataset.
    
    **How it works:**
    - Enter your health metrics in the Prediction tab
    - The model analyzes your risk factors
    - Get immediate results with recommendations
    - Download a detailed PDF report
    
    **Disclaimer:**
    This tool provides predictions based on statistical models and should not replace professional medical advice.
    Always consult with a healthcare provider for medical diagnosis and treatment.
    """)
    
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit")

# Footer
st.markdown("---")
st.caption("© 2023 Diabetes Prediction App | For educational purposes only")
