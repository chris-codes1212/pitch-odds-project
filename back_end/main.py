from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import time
import pandas as pd

from . import utils
from .PitchSimulation import PitchSimulation

app = FastAPI(title='Pitch-by-Pitch MLB Betting')

# Try and load model pipeline
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "money-ball"
    model, labels = utils.load_production_model(
        ENTITY,
        PROJECT,
    )
    print("Model Loaded Successfully")
except FileNotFoundError:
    print("Error: unable to load model pipeline")
    model = None

# Startup event to print if model is not loaded
@app.on_event("startup")
def startup_event():
    if model is None:
        print("WARNING: Model is not loaded")

    # df = pd.read_parquet("heldout_games.parquet")
    s3_uri = "s3://statcast-mlb-raw/pitches/heldout_games.parquet"
    df = pd.read_parquet(s3_uri)
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    app.state.simulation = PitchSimulation(df)



# Health get endpoint
@app.get("/health")
async def root():
    return {"status": "ok"}

# Create a class for the /predict endpoint
class PredictInput(BaseModel):
    comment: str

@app.post("/predict")
def predict():
    sim = app.state.simulation

    try:
        row = sim.next_pitch()
    except StopIteration:
        return {"message": "Simulation complete"}

    input_df = pd.DataFrame([row])
    #probs = utils.add_vig(model.predict_proba(input_df)[0])
    probs = model.predict_proba(input_df)[0]

    probabilities = {
        labels[0]: float(probs[0]),
        labels[1]: float(probs[1]),
        labels[2]: float(probs[2]),
        labels[3]: float(probs[3])
    }

    # Drop unwanted class
    allowed_classes = ["strike", "ball", "hit"]
    filtered_probs = {k: probabilities[k] for k in allowed_classes}

    # Renormalize
    total = sum(filtered_probs.values())
    normalized_probs = {k: v / total for k, v in filtered_probs.items()}


    return {
        "pitch": row.to_dict(),
        "probabilities": normalized_probs
    }
