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
# Assuming 'Car_Name' is the column containing the car names
car_names = data['Car_Name'].unique()

# Display the car names in the Streamlit app
st.title('List of Car Names')

# Display car names using st.write
st.write(car_names)

# Or display car names in a table format
st.dataframe(car_names)

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
st.bar_chart([float(accuracy.split(':')[5])])
