pip install -r requirements.txt
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

try:
    # Load the trained model
    model = joblib.load('K-Nearest Neighborsmodel.pkl')
    print("Model loaded successfully.")

    # Load the test data
    test_data = pd.read_csv('car data.csv')
    print("Test data loaded successfully.")
    
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

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
