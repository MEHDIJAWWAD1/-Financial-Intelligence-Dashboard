import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="Salary Analytics App",
    page_icon="💼",
    layout="wide"
)

loaded_model = keras.models.load_model('salary_model.keras')

with open('gender_encoder.pkl', 'rb') as f:
    loaded_gender_encoder = pickle.load(f)

with open('geography_encoder.pkl', 'rb') as f:
    loaded_geo_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    loaded_feature_names = pickle.load(f)

st.title("💼 Financial Intelligence Dashboard")
st.write("Deep Learning Estimated Salary Optimization Tool")
st.write("---")

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("📋 Primary Customer Attributes")
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=100, value=38)
        credit_score = st.number_input("CreditScore", min_value=300, max_value=850, value=650)
        
    with sub_col2:
        balance = st.number_input("Balance ($)", min_value=0.0, value=75000.0, step=1000.0)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
        num_of_products = st.selectbox("NumOfProducts", [1, 2, 3, 4], index=1)
        
    st.write(" ")
    st.write("**Secondary Account Status Indicators:**")
    has_cr_card = st.selectbox("HasCrCard", [0, 1], index=1)
    is_active_member = st.selectbox("IsActiveMember", [0, 1], index=1)
    exited = st.selectbox("Exited", [0, 1], index=0)

with col2:
    st.subheader("🎯 Optimization Model Output")
    st.write("Process feedforward structural matrices through dense neural network layers:")
    st.write(" ")
    
    if st.button("Predict Estimated Salary", type="primary", use_container_width=True):
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'Exited': exited
        }
        
        df = pd.DataFrame([input_data])
        df['Gender'] = loaded_gender_encoder.transform(df[['Gender']]).ravel()
        
        geo_array = loaded_geo_encoder.transform(df[['Geography']])
        geo_columns = loaded_geo_encoder.get_feature_names_out(['Geography'])
        geo_df = pd.DataFrame(geo_array, columns=geo_columns, index=df.index)
        
        df = pd.concat([df.drop(columns='Geography'), geo_df], axis=1)
        df = df.reindex(columns=loaded_feature_names, fill_value=0)
        
        scaled_features = loaded_scaler.transform(df)
        
        predicted_salary = float(loaded_model.predict(scaled_features, verbose=0)[0][0])
        
        st.metric(label="Calculated Estimated Salary", value=f"${predicted_salary:,.2f}")
        st.success("Regression iteration pipeline complete.")
    else:
        st.info("System awaiting execution prompt. Set attributes on the left, then trigger regression tracking.")