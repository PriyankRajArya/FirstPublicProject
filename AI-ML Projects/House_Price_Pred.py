import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Create dataset
data = {
    "Size (sqft)": [1000, 1500, 1200, 1800, 1600, 1100, 1300, 1700, 1400, 1600],
    "Location": ["Delhi", "Mumbai", "Bangalore", "Mumbai", "Delhi", "Bangalore", "Delhi", "Mumbai", "Bangalore", "Delhi"],
    "Rooms": [2, 3, 2, 4, 3, 2, 3, 3, 3, 4],
    "Price": [5000000, 8000000, 6000000, 9500000, 7500000, 5800000, 6700000, 9000000, 7300000, 8200000]
}
df = pd.DataFrame(data)

# Step 3: Prepare features and target
X = df[["Size (sqft)", "Location", "Rooms"]]
y = df["Price"]

# Step 4: One-hot encode the 'Location' column
preprocessor = ColumnTransformer([
    ("location_encoder", OneHotEncoder(), ["Location"])
], remainder="passthrough")

# Step 5: Build the pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Predict on test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Step 10: Get user input for prediction
print("\n--- House Price Prediction ---")
try:
    size = float(input("Enter size (in sqft): "))
    location = input("Enter location (Delhi, Mumbai, Bangalore): ").strip().capitalize()
    rooms = int(input("Enter number of rooms: "))

    # Create DataFrame for prediction
    new_house = pd.DataFrame({
        "Size (sqft)": [size],
        "Location": [location],
        "Rooms": [rooms]
    })

    # Predict price
    predicted_price = model.predict(new_house)[0]
    print(f"\nPredicted price for {int(size)} sqft, {rooms}-room house in {location}: ₹{int(predicted_price)}")

except Exception as e:
    print("Error:", e)
