from src.load_data import load_data

def prepare_features():

    df = load_data()

    X = df[["tenure", "monthly_charges", "total_charges"]]

    y = df["churn"]

    return X, y