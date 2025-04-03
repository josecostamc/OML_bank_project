import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from pydantic import BaseModel
import pandas as pd
import json
import uvicorn

#Load the application configuration
with open('./config/app.json') as f:
    config_file = json.load(f)
    model_name = config_file['model_name']
    model_version = config_file['model_version']
    tracking_base_url = config_file['tracking_base_url']
    tracking_port = config_file['tracking_port']
    service_base_url = config_file['service_base_url']
    service_port = config_file['service_port']

# Create a FastAPI application
app = fastapi.FastAPI()

# Define the inputs expected and a default value in the request body as JSON
class Request(BaseModel):

    """
    ID: ID of each client
    LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
    SEX: Gender (1=male, 2=female)
    EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
    MARRIAGE: Marital status (1=married, 2=single, 3=others)
    AGE: Age in years
    PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
    PAY_2: Repayment status in August, 2005 (scale same as above)
    PAY_3: Repayment status in July, 2005 (scale same as above)
    PAY_4: Repayment status in June, 2005 (scale same as above)
    PAY_5: Repayment status in May, 2005 (scale same as above)
    PAY_6: Repayment status in April, 2005 (scale same as above)
    BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
    BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
    BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
    BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
    BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
    BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
    PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
    PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
    PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
    PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
    PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
    PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)    
    """

    LIMIT_BAL: float = 30000.0
    SEX: int = 1
    EDUCATION: int = 2
    MARRIAGE: int = 2
    AGE: int = 25
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = 0
    PAY_6: int = 0
    BILL_AMT1: float = 8864.0
    BILL_AMT2: float = 10062.0
    BILL_AMT3: float = 11581.0
    BILL_AMT4: float = 12580.0
    BILL_AMT5: float = 13716.0
    BILL_AMT6: float = 14828.0
    PAY_AMT1: float = 1500.0
    PAY_AMT2: float = 2000.0
    PAY_AMT3: float = 1500.0
    PAY_AMT4: float = 1500.0
    PAY_AMT5: float = 1500.0
    PAY_AMT6: float = 2000.0


# Add CORS middleware to allow all origins, methods, and headers for local testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

# function to be called when the app starts running
@app.on_event('startup')
async def startup_event():
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.

    """         
    mlflow.set_tracking_uri(f"{tracking_base_url}:{tracking_port}")    

    # Load the registered model specified in the configuration
    model_uri = f"models:/{model_name}@{model_version}" 
    app.model = mlflow.pyfunc.load_model(model_uri = model_uri)  

    # Gives access to the MLflow client
    app.client = mlflow.tracking.MlflowClient(tracking_uri=f"{tracking_base_url}:{tracking_port}")
        
    print(f"Loaded model {model_uri}")    

@app.post('/default_prediction')
async def predict(input: Request):  
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
    return {"prediction": prediction.tolist()[0]}   

@app.get("/model_params")
async def get_params():
    """
    Endpoint to return the parameters of the model.

    Returns:
        dict: A dictionary with the model parameters.
    """
    # Get champion model run id
    run_id = app.model.metadata.to_dict()['run_id'] 

    # Get model parameters
    model_params = app.client.get_run(run_id).data.params     

    return model_params

@app.get("/model_metrics")
async def get_metrics():
    """
    Endpoint to return the metrics of the model.

    Returns:
        dict: A dictionary with the model metrics.
    """

    # Get champion model run id
    run_id = app.model.metadata.to_dict()['run_id']

    # Get model metrics
    model_metrics = app.client.get_run(run_id).data.metrics 
    
    return model_metrics

if __name__ == "__main__":
    uvicorn.run(app=app, port=service_port, host='0.0.0.0')