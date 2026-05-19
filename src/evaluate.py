from sklearn.metrics import precision_score, recall_score

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return precision, recall