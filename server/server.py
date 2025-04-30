import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Load the pre-trained model
model = joblib.load('diabetes_model_logisticregression.joblib')

class_names = ["Not Diabetes", "Diabetes"]

# Define the FastAPI app
app = FastAPI()

# Pydantic model for request validation
class DiabetesData(BaseModel):
    age: int
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    gender: str
    smoking_history: str

@app.get('/')
def read_root():
    return {"message": "Logistic Regression model deployment"}

@app.post('/predict')
def predict(data: DiabetesData):
    try:
        # Convert the input data to a pandas DataFrame
        df = pd.DataFrame([data.dict()])

        # Make predictions using the model
        prediction = model.predict(df)[0]  # Get the prediction for the single input

        # Map the prediction to the class label
        predicted_class = class_names[prediction]

        # Return the prediction result as JSON
        return {"prediction": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


