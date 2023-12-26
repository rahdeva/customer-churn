import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def create_scaler(dataset, feature):
    mmScaler = MinMaxScaler()
    return mmScaler.fit(dataset[feature].values.reshape(-1, 1))

def create_lEncoder(dataset, feature):
    lEncoder = LabelEncoder()
    return lEncoder.fit(dataset[feature])

df = pd.read_csv('dataset_raw.csv')
model_path = 'final_model.pkl'

def run():
    st.title("Customer Churn Prediction")
    
    age = st.number_input("Age", format="%d", step=1)
    gender = st.selectbox("Gender", df['Gender'].unique())
    geo = st.selectbox("Geography", df['Geography'].unique())
    numofprod = st.selectbox('Number of Products', df['NumOfProducts'].unique())
    hascc = st.selectbox("Has Credit Card", df['HasCrCard'].unique())
    active = st.selectbox("Is Active Member", df['IsActiveMember'].unique())
    creditscore = st.number_input("Credit Score", format="%.2f", step=50.)
    tenure = st.number_input("Tenure", format="%d", step=1)
    balance = st.number_input("Balance", format="%.2f")
    estsalary = st.number_input("Estimated Salary", format="%.2f", step=200.)
    point = st.number_input("Point Earned", format="%.2f", step=50.)

    if st.button("Predict"):
        data = {
            'NumberofProducts': int(numofprod),
            'HasCreditCard': int(hascc),
            'IsActiveMember': int(active),
            'Geography': str(geo),
            'Gender': str(gender),
            'CreditScore': int(creditscore),
            'Tenure': int(tenure),
            'Age': int(age),
            'Balance': float(balance),
            'EstimatedSalary': float(estsalary),
            'PointEarned': int(point)
        }

        # Create transformer for input data
        lEncoder_gender = create_lEncoder(df, 'Gender')
        lEncoder_geography = create_lEncoder(df, 'Geography')
        mmScaler_cs = create_scaler(df, 'CreditScore')
        mmScaler_t = create_scaler(df, 'Tenure')
        mmScaler_a = create_scaler(df, 'Age')
        mmScaler_b = create_scaler(df, 'Balance')
        mmScaler_es = create_scaler(df, 'EstimatedSalary')
        mmScaler_pe = create_scaler(df, 'Point Earned')

        # Apply transformations to input data
        data['Gender'] = lEncoder_gender.transform([data['Gender']])[0]
        data['Geography'] = lEncoder_geography.transform([data['Geography']])[0]
        data['CreditScore'] = mmScaler_cs.transform([[data['CreditScore']]])[0][0]
        data['Tenure'] = mmScaler_t.transform([[data['Tenure']]])[0][0]
        data['Age'] = mmScaler_a.transform([[data['Age']]])[0][0]
        data['Balance'] = mmScaler_b.transform([[data['Balance']]])[0][0]
        data['EstimatedSalary'] = mmScaler_es.transform([[data['EstimatedSalary']]])[0][0]
        data['PointEarned'] = mmScaler_pe.transform([[data['PointEarned']]])[0][0]

        # Load the model
        model = pickle.load(open(model_path, 'rb'))

        # Predict the outcome
        prediction = model.predict(np.array([list(data.values())]))

        if prediction[0] == 0:
            st.caption(f"The prediction from the model: {prediction[0]}")
            st.success("The model predicts the customer is not likely to churn.")
        else:
            st.success("The model predicts the customer is likely to churn.")
            st.caption(f"The prediction from the model: {prediction[0]}")

if __name__ == '__main__':
    run()
