import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_data():
    file_path = "California_Houses.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}. Please ensure the file exists in the directory.")
    df = pd.read_csv(file_path)
    return df

# Load dataset
df = load_data()

# Standardizing column names to remove extra spaces
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Selecting relevant features and target
features = [
    "median_income", "median_age", "tot_rooms", "tot_bedrooms", "population", "households", 
    "distance_to_coast", "distance_to_la", "distance_to_sandiego", "distance_to_sanfrancisco"
]
target = "median_house_value"

# Ensure all required columns exist in the dataset
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    raise KeyError(f"Missing columns in dataset: {missing_cols}")

# Drop rows with missing values
df = df.dropna()

# Splitting data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict based on user input
def predict_house_value():
    print("Enter details for the place you want to predict the house value for:")
    user_input = []
    for feature in features:
        value = float(input(f"Enter value for {feature.replace('_', ' ')}: "))
        user_input.append(value)
    
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)[0]
    print(f"Predicted Median House Value: ${prediction:,.2f}")

# Call function to get user input and predict value
predict_house_value()
