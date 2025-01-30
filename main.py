from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict(input: DiabetesInput):
    input_data = np.array([[
        input.Pregnancies, input.Glucose, input.BloodPressure, input.SkinThickness,
        input.Insulin, input.BMI, input.DiabetesPedigreeFunction, input.Age
    ]])
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}

# To run this application, use the command:
# uvicorn main:app --reload