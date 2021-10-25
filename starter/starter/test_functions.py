import pandas as pd
from joblib import load
from ml.data import process_data
import pytest
import os


@pytest.fixture
def data():
    """
    Get the training data
    """
    try:
        df = pd.read_csv(os.path.join(
            os.getcwd(), "starter/data/census_train_cleaned.csv"))
    except FileNotFoundError:
        df = pd.read_csv(os.path.join(
            os.getcwd(), "data/census_train_cleaned.csv"))

    return df


def test_cat_data(data):
    """
    Check split have same number of rows for X and y
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

    cats = list(data.select_dtypes(include='object').columns)[:-1]

    assert cats == cat_features


def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """

    if os.path.exists(os.path.join(os.getcwd(),
                                   "starter/model/encoder_test.joblib")):
        encoder = load(os.path.join(
            os.getcwd(), "starter/model/encoder_test.joblib"))
    else:
        encoder = load(os.path.join(
            os.getcwd(), "model/encoder_test.joblib"))

    if os.path.exists(os.path.join(os.getcwd(),
                                   "starter/model/lb_test.joblib")):
        lb = load(os.path.join(
            os.getcwd(), "starter/model/lb_test.joblib"))
    else:
        lb = load(os.path.join(
            os.getcwd(), "model/lb_test.joblib"))

    cats = list(data.select_dtypes(include='object').columns)[:-1]

    X_test, y_test, _, _ = process_data(
        data,
        cats,
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder(data):
    """
    Check split have same number of rows for X and y
    """

    if os.path.exists(os.path.join(os.getcwd(),
                                   "starter/model/encoder_test.joblib")):
        encoder_test = load(os.path.join(
            os.getcwd(), "starter/model/encoder_test.joblib"))
    else:
        encoder_test = load(os.path.join(
            os.getcwd(), "model/encoder_test.joblib"))

    if os.path.exists(os.path.join(os.getcwd(),
                                   "starter/model/lb_test.joblib")):
        lb_test = load(os.path.join(
            os.getcwd(), "starter/model/lb_test.joblib"))
    else:
        lb_test = load(os.path.join(
            os.getcwd(), "model/lb_test.joblib"))

    cats = list(data.select_dtypes(include='object').columns)[:-1]

    _, _, encoder, lb = process_data(
        data,
        cats,
        label="salary", training=True)

    _, _, _, _ = process_data(
        data,
        cats,
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()