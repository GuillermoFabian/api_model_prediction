import pandas as pd
from joblib import load
from .ml.model import compute_model_metrics
from .ml.data import process_data
import logging


def get_score():
    """
    Execute score checking
    """
    test_data = pd.read_csv('./data/census_data_cleaned.csv')

    trained_model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    slice_values = []

    categories = list(test_data.select_dtypes(include='object').columns)[:-1]

    for cat in categories:
        for cls in test_data[cat].unique():
            df_temp = test_data[test_data[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categories,
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = ("Precision {:2.2f}, Recall {:2.2f}, F1 {:2.2f}".format(
                prc, rcl, fb))
            logging.info(line)
            slice_values.append(line)

    with open('data/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')
