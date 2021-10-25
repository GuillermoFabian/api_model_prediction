from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import inference
from pandas.core.frame import DataFrame
import os


class Census(BaseModel):
    workclass: str = Field(..., example="Never-married")
    education: str = Field(..., example="Bachelors")
    marital_status: str = Field(...,
                                alias="marital-status",
                                example="Divorced")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    native_country: str = Field(...,
                                alias="native-country",
                                example="United-States")
    age: int = Field(..., example=35)
    hours_per_week: int = Field(..., alias="hours-per-week", example=45)


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Hello"}


@app.post('/predict')
async def predict(data: Census):

    try:
        model = load(os.path.join(
            os.getcwd(), "starter/model/model.joblib"))
    except FileNotFoundError:
        model = load(os.path.join(os.getcwd(), "starter/model/model.joblib"))

    try:
        encoder = load(os.path.join(
            os.getcwd(), "starter/model/encoder.joblib"))
    except FileNotFoundError:
        encoder = load(os.path.join(os.getcwd(), "starter/model/encoder.joblib"))

    try:
        lb = load(os.path.join(
            os.getcwd(), "starter/model/lb.joblib"))
    except FileNotFoundError:
        lb = load(os.path.join(os.getcwd(), "starter/model/lb.joblib"))

    data = data.dict(by_alias=True)
    df = DataFrame(data, index=[0])

    columns = ["workclass",
               "education",
               "marital-status",
               "occupation",
               "relationship",
               "race",
               "sex",
               "native-country",
               "age",
               "hours-per-week",
               ]

    categorical_cols = columns[: -2]

    X, _, _, _ = process_data(
        df,
        categorical_cols,
        encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    return {"prediction": y}
