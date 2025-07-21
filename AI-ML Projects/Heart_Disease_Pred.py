import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset from the local file
file_path = "C:/Users/bhavn/OneDrive/Desktop/programming/heart.csv"  # Update this path
df = pd.read_csv(file_path)

# Step 2: Check for missing values and handle them
print("Missing values per column before imputation:\n", df.isnull().sum())

# Fill missing values using the mean for numerical columns
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Check for missing values after imputation
print("\nMissing values per column after imputation:\n", df_imputed.isnull().sum())

# Step 3: Split data into features and target variable
X = df_imputed.drop('target', axis=1)  # Features (everything except 'target')
y = df_imputed['target']  # Target variable ('target' column)

# Step 4: Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize features (StandardScaler normalizes the data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)

# Step 7: Make predictions using Logistic Regression
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# Evaluate Logistic Regression model
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log_reg))

# Step 8: Hyperparameter Tuning for Logistic Regression (Grid Search)
param_grid_log_reg = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
grid_search_log_reg = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_log_reg, cv=5)
grid_search_log_reg.fit(X_train_scaled, y_train)

# Best hyperparameters for Logistic Regression
print("\nBest Parameters for Logistic Regression:", grid_search_log_reg.best_params_)

# Step 9: Evaluate the best Logistic Regression model (after hyperparameter tuning)
best_log_reg_model = grid_search_log_reg.best_estimator_

# Evaluate best Logistic Regression model
y_pred_best_log_reg = best_log_reg_model.predict(X_test_scaled)
print("\nBest Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_best_log_reg))
print("\nBest Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_best_log_reg))
