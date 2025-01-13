import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=43)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.joblib")

    test_accuracy = model.score(X_test, y_test)
    y_pred= model.predict(X_test)

    return test_accuracy, y_pred, y_test

def evaluate_model(y_test, y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=load_iris().target_names))

    print("Confusion Metrics:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    accuracy, y_test, y_pred  = train_model()
    print(f"Model accuracy: {accuracy:.2f}")

    evaluate_model(y_test, y_pred)
