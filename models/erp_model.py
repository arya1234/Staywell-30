 # The fit method trains the model on the input data X and labels y, the predict method returns binary predictions for the input X, and the predict_proba method returns the predicted probabilities of the positive class. You will need to modify this code to implement your own machine learning models.


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ERPPredictor:
    def __init__(self):
        # Define candidate models
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForest': RandomForestClassifier(),
            'XGBoost': XGBClassifier()
        }
        self.best_model = None

    def fit(self, X, y):
        # Split data for testing model performance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_score = 0

        # Train each model and evaluate its performance
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            # Keep track of the best model
            if score > best_score:
                best_score = score
                self.best_model = model
                self.best_model_name = name
        
        print(f"Best Model: {self.best_model_name} with Accuracy: {best_score:.4f}")

    def predict(self, X):
        if self.best_model is None:
            raise Exception("Model is not fitted yet. Please call the fit() method first.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise Exception("Model is not fitted yet. Please call the fit() method first.")
        return self.best_model.predict_proba(X)
