
# ============================================================
# PROJECT 2: STUDENT GRADE PREDICTION USING MACHINE LEARNING
# Algorithm: Linear Regression
# Libraries: Pandas, Matplotlib, Scikit-learn
# ============================================================

# Pandas: Load and prepare the dataset
import pandas as pd

# Matplotlib: Create charts
import matplotlib.pyplot as plt

# Scikit-learn: Split data, train a model, and evaluate it
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Convert categorical columns into numerical columns
from pandas import get_dummies

# Load the CSV file
student = pd.read_csv("student_data.csv")

# Display the dataset size
print("\n--- Dataset Information ---")
print("Number of students:", student.shape[0])
print("Number of columns:", student.shape[1])

# Display the first five rows
print("\n--- First Five Rows ---")
print(student.head())

# Remove duplicate rows to avoid repeated records
student = student.drop_duplicates().copy()

# Remove rows where the target grade is missing
student = student.dropna(subset=["G3"]).copy()

# G3 is the value the model will learn to predict
y = student["G3"]

# Features contain all other available columns
# G3 must not be included as an input feature
X = student.drop(columns=["G3"])

print("\n--- Feature and Target Information ---")
print("Number of records:", len(student))
print("Number of input features before encoding:", X.shape[1])
print("Target variable: G3")

# Machine learning models need numerical input.
# Convert categorical columns, such as school and gender,
# into numerical indicator columns.

X = pd.get_dummies(X, drop_first=True, dtype=int)

# Handle any remaining missing feature values using medians.
# This is a simple baseline approach for this project.
X = X.fillna(X.median())

print("\n--- Prepared Feature Data ---")
print("Number of features after encoding:", X.shape[1])
print(X.head())

# Training data teaches the model.
# Testing data checks its performance on unseen records.
# 80% of data is used for training.
# 20% of data is used for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- Data Split ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Create the model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("\n--- Model Training ---")
print("Linear Regression model trained!")

# Predict final grades for the test records
y_pred = model.predict(X_test)

print("\n--- Sample Predictions ---")

# Display the first 10 actual and predicted grades
results = pd.DataFrame({
    "Actual Grade": y_test,
    "Predicted Grade": y_pred
})

print(results.head(10).round(2))

# MAE: Average absolute difference between actual
# and predicted grades. Lower is better.
mae = mean_absolute_error(y_test, y_pred)

# RMSE: Penalizes larger prediction errors more strongly.
# Lower is better.
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# R-squared: Measures performance relative to predicting
# the average target value. A higher value is not always
# guaranteed, and it can be negative on test data.
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")

# Create a scatter plot comparing actual and predicted grades
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

# Add a reference line for perfect predictions
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual Grades vs Predicted Grades")
plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade (G3)")

plt.tight_layout()
plt.show()

# Linear Regression assigns a coefficient to each feature.
# Coefficients describe the model's estimated relationship
# between each feature and the predicted grade, holding
# the other included features constant.

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

# Sort by absolute coefficient size for inspection
coefficients["Absolute Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print("\n--- Largest Linear Regression Coefficients ---")
print(coefficients.head(10).round(3))

# Visualize the ten largest coefficients
top_features = coefficients.head(10).sort_values(
    by="Coefficient"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.title("Largest Linear Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# Save the test results to a new CSV file
results.to_csv("prediction_results.csv", index=False)

print("\nPrediction results saved to prediction_results.csv")

print("\nStudent Grade Prediction project completed!")