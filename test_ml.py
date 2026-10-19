import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import compute_model_metrics, train_model
import numpy as np


# Fixture
@pytest.fixture
def mock_census_data():
    """
    Fixture that contains a sample mock census data.
    """
    data = pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28],
            "workclass": [
                "State-gov",
                "Self-emp-not-inc",
                "Private",
                "Private",
                "Private",
            ],
            "fnlwgt": [77516, 83311, 215646, 234721, 338409],
            "education": [
                "Bachelors",
                "Bachelors",
                "HS-grad",
                "11th",
                "Bachelors",
            ],
            "education-num": [13, 13, 9, 7, 13],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Married-civ-spouse",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Handlers-cleaners",
                "Prof-specialty",
            ],
            "relationship": [
                "Not-in-family",
                "Husband",
                "Not-in-family",
                "Husband",
                "Wife",
            ],
            "race": ["White", "White", "White", "Black", "Black"],
            "sex": ["Male", "Male", "Male", "Male", "Female"],
            "capital-gain": [2174, 0, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40, 40],
            "native-country": [
                "United-States",
                "United-States",
                "United-States",
                "United-States",
                "Cuba",
            ],
            "salary": ["<=50K", "<=50K", ">50K", "<=50K", ">50K"],
        }
    )
    return data


# Test one
def test_process_data(mock_census_data):
    """
    This function tests if the preprocessing function process_data correctly
    outputs the expected rows, entries, and columns
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        mock_census_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    # Confirms no rows were dropped during preprocessing
    assert len(X) == len(mock_census_data)
    assert len(y) == len(mock_census_data)
    # Confirms feature columns were successfully created after one-hot encoding
    assert X.shape[1] > 0


# Test two
def test_train_model(mock_census_data):
    """
    This function tests if train_model() returns the expected algorithm
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, _, _ = process_data(
        mock_census_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# Test three
def test_metrics(mock_census_data):
    """
    This function tests if the metrics function returns the exact outputs
    for precision, recall, and fbeta
    """
    # Actual values
    y = np.array([1, 1, 1, 0])
    preds = np.array([1, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Confirms exact metrics
    assert precision == 1.0
    np.testing.assert_allclose(recall, 2 / 3, atol=1e-4)
    np.testing.assert_allclose(fbeta, 0.8, atol=1e-4)
