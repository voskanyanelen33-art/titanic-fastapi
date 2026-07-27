from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
model = joblib.load("titanic_pipeline.pkl")


@app.get("/")
def home():
    return {
        "message": "API works"
    }

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Fare: int
    Embarked: str

@app.post("/predict")
def predict(data: Passenger):

    passenger = pd.DataFrame([data.model_dump()])

    prediction = model.predict(passenger)

    return {
        "prediction": int(prediction[0])
    }
