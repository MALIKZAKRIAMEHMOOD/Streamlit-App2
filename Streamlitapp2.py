import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
# import pickle
import joblib

model = joblib.load('K-Nearest Neighborsmodel.pkl')

with open('accuracy.txt', 'r') as file:
  accuracy = file.read()

st.title(f"Car Prediction App")
st.write(f"Model {accuracy}")

st.header("Real_Time Prediction")

test_data = pd.read_csv('car data.csv')

x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

input_data = []
for col in x_test.columns:
  input_value = st.number_input(f"Input from {col}", value=0.0)
  input_data.append(input_value)

input_df = pd.DataFrame([input_data], columns = x_test.columns)

if st.button("Predict"):
  prediction = model.predict(input_df)

st.header("Accuracy Plot")
# Assuming accuracy is in the form of 'Accuracy: <value>'
try:
    # Split the accuracy string and safely convert the second part to float
    accuracy_value = float(accuracy.split(':')[1].strip())
    st.bar_chart([accuracy_value])
except (IndexError, ValueError) as e:
    st.error(f"An error occurred: {e}")
    st.write("Accuracy string may be malformed or missing.")
