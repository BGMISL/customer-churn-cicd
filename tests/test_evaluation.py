import os

from src.train import train_model
from src.evaluate import evaluate_model

def test_evaluation():

    model, X_test, y_test = train_model()

    precision, recall = evaluate_model(model, X_test, y_test)

    assert precision >= 0.5
    assert recall >= 0.5

    assert os.path.exists("models/churn_model.pkl")