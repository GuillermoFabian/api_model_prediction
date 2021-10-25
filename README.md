## Api Deployment - FASTApi

Deploying a machine learning model on Heroku using Git and DVC. 

The model is a simple classification model based on the Census Income Data Set.
Also, implementing CI and CD to ensure the pipeline passes unit tests and PEP8 format requirements.
An API will be written using FastAPI and requested via the requests module. 

### Execution

ML training and test: python main.py --choice train_model

Model score Check score:  python main.py --choice get_score

Running the entire pipeline:  python main.py --choice all

Test API If testing FastAPi serving on locally : uvicorn app_server:app --reload

Checking Heroku deployed API: python heroku_api_test.py