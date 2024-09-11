import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the trained model
model = joblib.load('K-Nearest Neighborsmodel.pkl')

# Load the test data
test_data = pd.read_csv('car data.csv')

# Assuming the last column in the dataset is the target
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Generate test predictions
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy to file
with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}\n')

print(f'Accuracy: {accuracy}')
