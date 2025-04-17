import json
import pytest
import requests

with open('./config/app.json') as f:
    config = json.load(f)

def test_default_prediction():
    """
    Test for the /default_prediction endpoint with valid input data.
    Verify is the request has been successfully completed.
    Verify is the key prediction exists in response.json.
    Verify is the prediction is of type int.
    Verify if for the input data the prediction is 0.
    """
    response = requests.post(url=f"http://localhost:{config['service_port']}/default_prediction", json={
        'LIMIT_BAL': 30000.0,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 2,
        'AGE': 25,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 8864.0,
        'BILL_AMT2': 10062.0,
        'BILL_AMT3': 11581.0,
        'BILL_AMT4': 12580.0,
        'BILL_AMT5': 13716.0,
        'BILL_AMT6': 14828.0,
        'PAY_AMT1': 1500.0,
        'PAY_AMT2': 2000.0,
        'PAY_AMT3': 1500.0,
        'PAY_AMT4': 1500.0,
        'PAY_AMT5': 1500.0,
        'PAY_AMT6': 2000.0
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)
    assert response.json()["prediction"] == 0

def test_model_params():
    """
    Test for the /model_params endpoint.
    Verify if the request has been successfully completed.
    """

    response = requests.get(url=f"http://localhost:{config['service_port']}/model_params")
    assert response.status_code == 200

def test_model_metrics():
    """
    Test for the /model_metrics endpoint.
    Verify if the request has been successfully completed.
    """

    response = requests.get(url=f"http://localhost:{config['service_port']}/model_metrics")
    assert response.status_code == 200

