import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model_file = 'heart_performance_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Set the banner
st.image("dataset-cover.jpg", use_container_width=True)

# App Title
st.title("Heart Disease Prediction App")
st.write("""
This application uses a machine learning model to predict the likelihood of heart disease based on patient data.
""")

# Initialize session state for storing patient records
if "patient_data" not in st.session_state:
    st.session_state.patient_data = pd.DataFrame(columns=[
        "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg",
        "thalachh", "exng", "oldpeak", "slp", "caa", "thall", "Prediction"
    ])

# Sidebar for user input
st.sidebar.header("Input Patient Data")

def get_user_input():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3], format_func=lambda x: [
        "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
    trtbps = st.sidebar.number_input("Resting Blood Pressure (trtbps)", min_value=50, max_value=250, value=140)
    chol = st.sidebar.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=250)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", [0, 1, 2], format_func=lambda x: [
        "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
    thalachh = st.sidebar.number_input("Maximum Heart Rate Achieved (thalachh)", min_value=50, max_value=250, value=150)
    exng = st.sidebar.selectbox("Exercise Induced Angina (exng)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=2.0)
    slp = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (slp)", [0, 1, 2])
    caa = st.sidebar.slider("Number of Major Vessels (caa)", min_value=0, max_value=4, value=1)
    thall = st.sidebar.selectbox("Thalassemia (thall)", [0, 1, 2, 3], format_func=lambda x: [
        "No Data", "Normal", "Fixed Defect", "Reversible Defect"][x])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trtbps": trtbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalachh": thalachh,
        "exng": exng,
        "oldpeak": oldpeak,
        "slp": slp,
        "caa": caa,
        "thall": thall
    }
    return pd.DataFrame([data])

# Collect user input
input_df = get_user_input()
# Display user input
st.subheader("Patient Data")
st.write(input_df)

# Predict and record data
if st.button("Predict and Record"):
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_text = "More chance of heart disease" if prediction == 1 else "Less chance of heart disease"
    input_df["Prediction"] = prediction_text
    
    # Add to session state
    st.session_state.patient_data = pd.concat([st.session_state.patient_data, input_df], ignore_index=True)
    
    # Display result
    st.success(f"Prediction: {prediction_text}")

# Show patient records
if not st.session_state.patient_data.empty:
    st.subheader("Patient Records")
    st.write(st.session_state.patient_data)
    
    # Generate dynamic plots
    st.subheader("Data Visualization")
    col1, col2 = st.columns(2)

    # Age Distribution
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.patient_data["age"], bins=10, kde=True, color="blue", ax=ax)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    # Cholesterol vs. Prediction
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x="Prediction", y="chol", data=st.session_state.patient_data, palette="Set2", ax=ax)
        ax.set_title("Cholesterol Levels by Prediction")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Cholesterol")
        st.pyplot(fig)
    
    # Resting Blood Pressure Distribution
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.patient_data["trtbps"], bins=10, kde=True, color="green", ax=ax)
        ax.set_title("Resting Blood Pressure Distribution")
        ax.set_xlabel("Resting Blood Pressure (trtbps)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Heart Rate vs. Prediction
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x="Prediction", y="thalachh", data=st.session_state.patient_data, palette="mako", ax=ax)
        ax.set_title("Heart Rate by Prediction")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Heart Rate")
        st.pyplot(fig)
else:
    st.info("No patient data recorded yet. Use the sidebar to add a new patient.")
