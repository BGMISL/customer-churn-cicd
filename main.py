from src.train import train_model
from src.evaluate import evaluate_model

model, X_test, y_test = train_model()

precision, recall = evaluate_model(model, X_test, y_test)

print("Customer churn model trained successfully")