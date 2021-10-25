import requests


data = {
    "workclass": "Private",
    "education": "Prof-school",
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States",
    "age": 35,
    "hours-per-week": 40,

}
response = requests.post(
    'https://url/predict', json=data)

assert response.status_code == 200

print("Response code: %s" % response.status_code)
print("Response body: %s" % response.json())