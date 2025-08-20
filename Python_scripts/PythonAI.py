import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load the Dataset
data = pd.read_csv("data.csv")

# Step 2: Split Data into Inputs (X) and Outputs (y)
X = data[["T_Val", "I_Val", "U_Val"]]  # Inputs
y = data[["T_Anomaly", "I_Anomaly", "U_Anomaly"]]  # Outputs (multi-label classification)

# Step 3: Train/Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and Train Random Forest Models (One for Each Output)
models = {}
for column in y_train.columns:
    print(f"Training model for {column}...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[column])  # Train model for this specific output
    models[column] = model

# Step 5: Evaluate the Models
for column, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Results for {column}:")
    print(classification_report(y_test[column], y_pred))
    print(f"Accuracy: {accuracy_score(y_test[column], y_pred)}")

# Step 6: Save the Models for Later Use
for column, model in models.items():
    joblib.dump(model, f"{column}_model.pkl")  # Save each model as a separate file
print("Models saved successfully!")
