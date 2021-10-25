from fastapi.testclient import TestClient
from app_server import app
client = TestClient(app)


def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello"}


def test_get_bad_url():
    response = client.get("/bad_url")
    assert response.status_code != 200


def test_post_hi_salary():
    response = client.post("/predict", json={
        "workclass": "Private",
        "education": "Prof-school",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "age": 45,
        "hours-per-week": 40

    })
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_post_low_salary():
    response = client.post("/predict", json={
        "workclass": "Private",
        "education": "Some-college",
        "marital-status": "Separated",
        "occupation": "Adm-clerical",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "native-country": "United-States",
        "age": 21,
        "hours-per-week": 40
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
