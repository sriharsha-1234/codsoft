import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\sriha\\OneDrive\\Desktop\\codsoft files\\IMDb Movies India.csv', encoding='ISO-8859-1')
print(df.head())
# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Distribution of ratings
import matplotlib.pyplot as plt

plt.hist(df['Rating'], bins=20)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')
plt.show()
# Drop or fill missing values as appropriate
df.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Encode genres, directors, and actors
df['Genre'] = df['Genre'].str.get_dummies('|')
df['Actors'] = df['Actors'].str.get_dummies('|')

# Example of Label Encoding for 'Director'
label_encoder = LabelEncoder()
df['Director'] = label_encoder.fit_transform(df['Director'])
# Assuming 'Genre' and 'Actors' are already one-hot encoded
X = df[['Genre', 'Director', 'Actors']]
y = df['Rating']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
from sklearn.model_selection import GridSearchCV

# Example: Grid search for Linear Regression parameters (if applicable)
param_grid = {
    'normalize': [True, False]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')
from sklearn.ensemble import RandomForestRegressor

# Initialize and train a RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'Random Forest R^2 Score: {rf_r2}')

# Get feature importances from RandomForest
importances = rf_model.feature_importances_
features = X.columns

# Create a DataFrame for visualization
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

print(importances_df)
