import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Step 1: Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Step 2: Data Exploration

# Display the first few rows of the training dataset
print("First few rows of the training data:")
print(train_df.head())

# Display summary information about the training dataset
print("\nTraining data information:")
print(train_df.info())

# Display basic statistics for numerical columns
print("\nSummary statistics of numerical features:")
print(train_df.describe())

# Check for missing values in the training dataset
print("\nMissing values in the training data:")
print(train_df.isnull().sum())

# Check for missing values in the test dataset
print("\nMissing values in the test data:")
print(test_df.isnull().sum())

# Explore the distribution of the target variable
print("\nDistribution of the target variable (Transported):")
print(train_df['Transported'].value_counts())

# Visualize correlations between numerical features
sns.heatmap(train_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Visualize distributions of important numerical features
for column in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df[column].dropna(), kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Explore relationships between categorical features and the target variable
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for column in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=column, hue='Transported', data=train_df)
    plt.title(f'{column} vs Transported')
    plt.show()






# Step 3: Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Step 4: Preprocessing

# Handle missing values with KNN Imputation (for numerical data)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=5)
train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

# Fill missing categorical data with a placeholder (e.g., "Unknown")
categorical_cols = train_df.select_dtypes(include=[object]).columns
train_df[categorical_cols] = train_df[categorical_cols].fillna("Unknown")
test_df[categorical_cols] = test_df[categorical_cols].fillna("Unknown")

# One-Hot Encoding for categorical features
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Align the columns of train and test datasets
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# Step 5: Prepare the target and features
X = train_df.drop(columns=['Transported'])
y = train_df['Transported'].astype(int)  # Convert target to binary (0 or 1)

# Step 6: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Step 9: Make predictions on the test set
test_predictions = xgb_model.predict(test_df)

# Step 10: Prepare the submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': test_predictions.astype(bool)
})

submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
