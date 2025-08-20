import joblib
import os

# Define base path for model loading
base_path = os.path.dirname(__file__)

# Load models
temperature_model = joblib.load(os.path.join(base_path, 'T_Anomaly_model.pkl'))
current_model = joblib.load(os.path.join(base_path, 'I_Anomaly_model.pkl'))
voltage_model = joblib.load(os.path.join(base_path, 'U_Anomaly_model.pkl'))

def predict_anomalies(temperature_value, current_value, voltage_value):
    try:
        # Ensure the inputs are numeric
        if not all(isinstance(value, (int, float)) for value in [temperature_value, current_value, voltage_value]):
            raise TypeError("All inputs must be numeric (int or float).")

        # Reshape the inputs to match the model's expected input format (2D array)
        input_data = [[temperature_value, current_value, voltage_value]]  # One feature per instance

        # Use the trained models to make predictions
        T_Anomaly = temperature_model.predict(input_data)
        I_Anomaly = current_model.predict(input_data)
        U_Anomaly = voltage_model.predict(input_data)

        # Extract and return the first value from each prediction (assuming it returns an array)
        # Convert the value to an integer (if it's a binary classification, i.e., 0 or 1)
        return int(T_Anomaly[0]), int(I_Anomaly[0]), int(U_Anomaly[0])

    except Exception as e:
        print(f"Error: {e}")
        raise

# Test section (optional for standalone testing)
if __name__ == "__main__":
    # Example test with numeric input
    result = predict_anomalies(25.0, 12.5, 220.0)
    print(f"Predictions: Temperature Anomaly={result[0]}, Current Anomaly={result[1]}, Voltage Anomaly={result[2]}")
