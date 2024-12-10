# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score


cardio_data = pd.read_csv('cardio_train.csv', sep=';')

print("Dataset Info:")
print(cardio_data.info())
print("\nFirst 5 Rows:")
print(cardio_data.head())


cardio_data.drop_duplicates(inplace=True)


cardio_data['age'] = (cardio_data['age'] / 365).astype(int)


print("\nData Summary:")
print(cardio_data.describe())


# Gender distribution
gender_distribution = cardio_data['gender'].value_counts()
print("\nGender Distribution:")
print(gender_distribution)

# age ranges
bins = [10, 20, 30, 40, 50, 60, 70, 100]
labels = ['10-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
cardio_data['age_range'] = pd.cut(cardio_data['age'], bins=bins, labels=labels, right=False)

# Count by age range
age_distribution = cardio_data['age_range'].value_counts()
print("\nAge Distribution:")
print(age_distribution)


# Total cardiovascular cases
cardio_cases = cardio_data[cardio_data['cardio'] == 1]
cardio_case_count = cardio_cases.shape[0]
print(f"\nTotal Cardiovascular Cases: {cardio_case_count}")

# Cardiovascular cases by age range
cardio_by_age = cardio_cases['age_range'].value_counts()
print("\nCardiovascular Cases by Age Range:")
print(cardio_by_age)

# Cardiovascular cases by gender
cardio_by_gender = cardio_cases['gender'].value_counts()
print("\nCardiovascular Cases by Gender:")
print(cardio_by_gender)


# Bar chart for age range
cardio_by_age.plot(kind='bar', title='Cardiovascular Cases by Age Range', color='skyblue')
plt.xlabel('Age Range')
plt.ylabel('Number of Cases')
plt.show()


cardio_by_gender.plot(kind='pie', title='Cardiovascular Cases by Gender', autopct='%1.1f%%')
plt.ylabel('') 
plt.show()


# Features and target
X = cardio_data.drop(['cardio', 'id', 'age_range'], axis=1)  
y = cardio_data['cardio']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")


print("\nInsights:")
print("- The gender distribution shows how many males and females are represented.")
print("- Age range distribution helps in identifying which group is most affected.")
print("- The Logistic Regression model provides a basic understanding of cardiovascular prediction.")

