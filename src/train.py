from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

from src.feature_engineering import prepare_features

def train_model():

    X, y = prepare_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_model.pkl")

    return model, X_test, y_test