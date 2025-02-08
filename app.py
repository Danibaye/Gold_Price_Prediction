from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the input data structure
class PredictionInput(BaseModel):
    Volume: float
    Open: float
    High: float
    Low: float
    year: int
    month: int
    day: int
    weekday: int
    is_weekend: bool

    class Config:
        json_schema_extra = {
            "example": {
                "Volume": 150000.0,
                "Open": 2300.0,
                "High": 2350.0,
                "Low": 2280.0,
                "year": 2024,
                "month": 7,
                "day": 5,
                "weekday": 4,  # Friday
                "is_weekend": False
            }
        }

# Get the name of X columns
features =  ['Volume', 'Open', 'High', 'Low', 'year', 'month', 'day', 'weekday', 'is_weekend']

# Initialize FastAPI app
app = FastAPI(title="Gold Price Prediction API")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="static")  # Set Jinja2 directory to "static"

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define paths for persisted model and preprocessor files
MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

# Load the trained model and preprocessor
model = None
scaler = None
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler = preprocessor['scaler']
    model = joblib.load(MODEL_PATH)
    logging.info("Model and preprocessor loaded successfully.")

except FileNotFoundError as e:
    logging.error(f"Model or preprocessor file not found: {e}")
    raise  # Re-raise the exception to stop the app if the files are essential
except Exception as e:
    logging.error(f"Error loading model or preprocessor: {e}")
    raise  # Re-raise if loading fails

# Define the health check endpoint
@app.get("/health", summary="Health Check Endpoint")
async def health_check():
    """
    Endpoint to check the health status of the API.
    """
    return {"status": "ok"}

# Define the prediction endpoint
@app.post("/predict", summary="Gold Price Prediction Endpoint")
async def predict(
    request: Request,
    Volume: float = Form(...),
    Open: float = Form(...),
    High: float = Form(...),
    Low: float = Form(...),
    year: int = Form(...),
    month: int = Form(...),
    day: int = Form(...),
    weekday: int = Form(...),
    is_weekend: bool = Form(...),
):
    """
    Endpoint to predict the gold price based on input features.
    """
    logging.info(f"Received prediction request")
    try:
        # Create the PredictionInput object
        input_data = PredictionInput(
            Volume=Volume,
            Open=Open,
            High=High,
            Low=Low,
            year=year,
            month=month,
            day=day,
            weekday=weekday,
            is_weekend=is_weekend,
        )

        # Convert input data into pandas DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Extract the data from the dictionary to ensure only the data we expect is used,
        #   AND in the correct order. Note that it also must have same shape so that predict() is valid.
        input_for_model = [input_df[feature].iloc[0] for feature in features]

        # Transform
        data = scaler.transform([input_for_model])

        # Make the prediction
        prediction = model.predict(data)
        logging.info(f"Prediction successful: {prediction[0]}")

        # Return the prediction as HTML
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction[0]})

    except ValueError as ve:
        logging.error(f"Data validation or scaling error: {ve}")
        return templates.TemplateResponse("index.html", {"request": request, "error": str(ve)})
    except Exception as e:
        logging.exception(f"Prediction error: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})


# Define the health check endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "error": None})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)