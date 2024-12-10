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

# Sidebar for user input
st.sidebar.header("Input Patient Data")

# Input fields for patient data
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

input_df = get_user_input()

# Display user input
st.subheader("Patient Data")
st.write(input_df)

# Prediction Button
if st.button("Predict"):
    # Perform prediction
    prediction = model.predict(input_df)[0]
    prediction_text = "More chance of heart disease" if prediction == 1 else "Less chance of heart disease"

    # Display prediction
    st.subheader("Prediction")
    st.write(prediction_text)

# Display charts
if st.checkbox("Show Exploratory Data Analysis (EDA)"):
    st.subheader("Correlation Heatmap")
    heart_data = pd.read_csv("heart.csv")
    correlation_matrix = heart_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Chest Pain Type vs Heart Disease Output")
    fig, ax = plt.subplots()
    sns.countplot(x="cp", hue="output", data=heart_data, ax=ax, palette="Set2")
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    features = input_df.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="mako")
    st.pyplot(fig)