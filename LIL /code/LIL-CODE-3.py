#Random forest with the age classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = '/home/rajarajan/Desktop/LIL CODE/dataset/heart.csv'  
heart_df = pd.read_csv(file_path)

# Define a function to calculate the max heart rate threshold based on age
def calculate_max_hr(age):
    return 220 - age

# Apply threshold to classify if MaxHR is normal (below calculated threshold) or above normal
heart_df['MaxHR_Normal'] = heart_df.apply(lambda row: 1 if row['MaxHR'] <= calculate_max_hr(row['Age']) else 0, axis=1)

# Classify age into categories
def classify_age(age):
    if age < 30:
        return 'Young'
    elif 30 <= age <= 60:
        return 'Middle-Aged'
    else:
        return 'Elderly'

# Create a new column for age category
heart_df['Age_Group'] = heart_df['Age'].apply(classify_age)

# One-hot encoding for age group categorical feature
heart_df = pd.get_dummies(heart_df, columns=['Age_Group'], drop_first=True)

# Select the features (MaxHR_Normal, Age, Age_Group categories) and target (Cholesterol)
# Note: 'Age_Group_Elderly' is excluded because it's not present in the dataset
X = heart_df[['MaxHR', 'MaxHR_Normal', 'Age', 'Age_Group_Middle-Aged', 'Age_Group_Young']]
y = heart_df['Cholesterol']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Calculate the R² score
r2 = r2_score(y_test, y_pred)

# Convert R² score to a percentage
accuracy_percentage = r2 * 100
print(f'Accuracy of the model: {accuracy_percentage:.2f}%')

# Function to create plots for different age groups
def plot_maxhr_predictions_for_age_range(start_age, end_age, increment=10):
    plt.figure(figsize=(10, 6))
    for age in range(start_age, end_age + 1, increment):
        # Filter data by age range
        age_group_data = heart_df[heart_df['Age'] == age]
        X_age_group = age_group_data[['MaxHR', 'MaxHR_Normal', 'Age', 'Age_Group_Middle-Aged', 'Age_Group_Young']]
        y_age_group = age_group_data['Cholesterol']
        
        if not age_group_data.empty:
            # Predict for the current age group
            y_pred_age = rf_regressor.predict(X_age_group)
            
            # Scatter plot of actual vs predicted Cholesterol for this age group
            sns.scatterplot(x=y_age_group, y=y_pred_age, label=f'Age {age}', marker='o')
    
    # Line representing perfect predictions
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    
    # Add labels and title
    plt.xlabel('Actual Cholesterol')
    plt.ylabel('Predicted Cholesterol')
    plt.title(f'Actual vs Predicted Cholesterol for Ages {start_age}-{end_age}')
    plt.legend()
    plt.show()

# Plot MaxHR predictions for ages 30 to 70, with a difference of 10 years
plot_maxhr_predictions_for_age_range(30, 70, increment=10)
