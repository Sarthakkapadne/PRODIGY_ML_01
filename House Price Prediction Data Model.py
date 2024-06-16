import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('task1 houses dataset.csv')

# print(data.head())

# Define features and target variable
X = data[['Squarefoot', 'Bedrooms', 'Bathrooms']]
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Example new data
new_data = pd.DataFrame({
    'Squarefoot': [60000],
    'Bedrooms': [5],
    'Bathrooms': [7]
})

# Make predictions
new_predictions = model.predict(new_data)

print(new_predictions)




