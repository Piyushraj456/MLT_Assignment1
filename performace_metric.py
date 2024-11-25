import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


train_data = pd.read_csv('train.csv');


train_data_cleaned = train_data.dropna(subset=['SalePrice'])


relevant_features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageCars']
target = 'SalePrice'


data_cleaned = train_data_cleaned.dropna(subset=relevant_features)


X = data_cleaned[relevant_features]
y = data_cleaned[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


coefficients = pd.DataFrame({
    'Feature': relevant_features,
    'Coefficient': model.coef_
})
print(coefficients)