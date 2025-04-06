import json
import pytest
import pandas as pd
import mlflow


@pytest.fixture(scope='module')
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")
    model_name = config['model_name']
    model_version = config['model_version']
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_no_default(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
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
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_default(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 30000.0,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 2,
        'AGE': 25,
        'PAY_0': 7,
        'PAY_2': 7,
        'PAY_3': 7,
        'PAY_4': 7,
        'PAY_5': 7,
        'PAY_6': 7,
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
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 1


def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 30000.0,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 2,
        'AGE': 25,
        'PAY_0': 7,
        'PAY_2': 7,
        'PAY_3': 7,
        'PAY_4': 7,
        'PAY_5': 7,
        'PAY_6': 7,
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
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )


def test_model_gender(model: mlflow.pyfunc.PyFuncModel):
    
    # input values for sex = male
    input_male = pd.DataFrame.from_records([{
        'LIMIT_BAL': 30000.0,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 2,
        'AGE': 25,
        'PAY_0': 7,
        'PAY_2': 7,
        'PAY_3': 7,
        'PAY_4': 7,
        'PAY_5': 7,
        'PAY_6': 7,
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
    }])

    # inpute values for sex = female
    input_female = pd.DataFrame.from_records([{
        'LIMIT_BAL': 30000.0,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 2,
        'AGE': 25,
        'PAY_0': 7,
        'PAY_2': 7,
        'PAY_3': 7,
        'PAY_4': 7,
        'PAY_5': 7,
        'PAY_6': 7,
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
    }])

    prediction_male = model.predict(data=input_male) # prediciton for sex = male
    prediction_female = model.predict(data=input_female) # prediction for sex = female
    assert prediction_male == prediction_female