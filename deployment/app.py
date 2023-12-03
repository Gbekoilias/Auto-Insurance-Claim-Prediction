import streamlit as st
import pandas as pd  # If you need it for data handling
from sklearn.externals import joblib  # If you're using joblib for model saving/loading
import pickle
import joblib

#load model using pickle
model_file_path='C:\Users\DONKAMS\Downloads\Project_STA2017\PIPEmodelGBC.sav'
model = pickle.load(open(model_file_path, 'rb'))

#create the user interface
st.title('AutoInsurance Prediction')
st.subheader('The essential app for your car insurance')
st.write('This app predicts the likelihood of a customer to buy an auto insurance policy.')
st.write('Please fill in the form below to get your prediction')

# Example input fields for user input
age = st.number_input('Enter Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Select Gender', ['Male', 'Female'])

# Prepare user input as required by your model
# For example, assuming you have 'age' and 'gender' as features
input_data = pd.DataFrame({'Age': [age], 'Gender': [gender]})  # Assuming 'Age' and 'Gender' as features

# Make predictions
prediction = model.predict(input_data)  # Replace with your actual prediction logic

# Display prediction to the user
st.subheader('Prediction Result:')
if prediction == 1:  # Assuming binary classification
    st.write('The prediction is Positive')
else:
    st.write('The prediction is Negative')
st.write('Thank you for using our app')


