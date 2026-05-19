from src.feature_engineering import prepare_features

def test_features():

    X, y = prepare_features()

    assert X.shape[0] > 0
    assert y.shape[0] > 0