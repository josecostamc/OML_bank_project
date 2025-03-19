import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from pydantic import BaseModel, conint
import pandas as pd
import json
import uvicorn

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)

# Create a FastAPI application
app = fastapi.FastAPI()


# Define the inputs expected in the request body as JSON
class Request(BaseModel):

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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.
    """
        
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load the registered model specified in the configuration
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri = model_uri)
    
    print(f"Loaded model {model_uri}")

@app.post("/has_diabetes")
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

