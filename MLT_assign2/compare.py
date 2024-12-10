import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('cardio_train.csv', delimiter=';')


print("Columns in the dataset:", df.columns)


X = df.drop(columns=['id', 'cardio']) 
y = df['cardio']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "K-NN": KNeighborsClassifier(),
    "ANN": MLPClassifier(max_iter=500)  
}


for model_name, model in models.items():
    print(f"Training {model_name}...")
  
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    

    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)
