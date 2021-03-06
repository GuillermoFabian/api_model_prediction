# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from .ml.data import process_data
from .ml.model import train_model as tr
from .ml.model import compute_model_metrics
import logging



# Add the necessary imports for the starter code.

def train_model():
    # Add code to load in the data.
    data = pd.read_csv('data/census_data_cleaned.csv')

    # Optional enhancement, use K-fold cross validation
    # instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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

    categories = list(test.select_dtypes(include='object').columns)[:-1]

    print(categories)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )


    # Proces the test data with the process_data function.

    # Train and save a model.
    model = tr(X_train, y_train)


    dump(model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")
