import json
import pytest
import requests

with open('./config/app.json') as f:
    config = json.load(f)

def test_default_prediciton():
    """
    Test for the /should_loan endpoint with valid input data.
    It should return a prediction in the response.
    """
    response = requests.post(f"http://localhost:{config['service_port']}/should_loan", json={
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