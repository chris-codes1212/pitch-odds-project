from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import pandas as pd
import statsapi

from .user_class import User

from . import utils
from .PitchSimulation import PitchSimulation
from . import sessions

import sys
import os

sys.path.append(os.path.abspath(".."))
import src.data_loader as data_loader

class PeekRequest(BaseModel):
    user_id: str

class BetRequest(BaseModel):
    user_id: str
    bet: str
    amount: float
    odds: int


app = FastAPI(title='Pitch-by-Pitch MLB Betting')

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#
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
    
        # create outcome variable
    df['outcome_coarse'] = df.apply(data_loader.map_outcome_coarse, axis=1)
    df = df.reset_index(drop=True)

    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    app.state.df = df
    app.state.simulation = PitchSimulation(df)

# Health get endpoint
@app.get("/health")
async def root():
    return {"status": "ok"}

# Create a class for the /predict endpoint
class PredictInput(BaseModel):
    comment: str

# create predict endpoint that takes in a user id and returns the probabilities for the next pitch outcome, as well as the current pitch information
@app.post("/predict")
def predict(req: PeekRequest):
    sim = app.state.simulation
    
    user = sessions.get_user(req.user_id)

    i = int(user.get_pitch_index())
    df = app.state.df
    row = df.iloc[i]

    input_df = pd.DataFrame([row[PitchSimulation.FEATURES]])
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

    user.advance_pitch()

    return {
        "pitch": row[PitchSimulation.FEATURES].to_dict(),
        "probabilities": normalized_probs
    }


@app.post("/bet")
def place_bet(req: BetRequest):
    user = sessions.get_user(req.user_id)
    user.place_bet(req)

    # For simplicity, we assume the bet is always on the correct outcome and the odds are correct
    # In a real application, you would want to validate the bet against the current probabilities and outcomes
    i = user.get_pitch_index()

    df = app.state.df
    row = df.iloc[i]

    if row['outcome_coarse'] == req.bet:
        won = True
    else:
        won = False

    user.update_bankroll(won=won, amount=req.amount * (req.odds / 100)) # Simplified payout calculation

    return {"status": "bet placed", "new_bankroll": user.get_bankroll()}


@app.get("/balance")
def get_balance(user_id: str):
    user = sessions.get_user(user_id)
    return {"balance": user.get_bankroll()}

@app.get("/live_games")
def get_live_games():
    live_games = []
    # Placeholder for live games endpoint
    # 1. Get today's games to find a live game_pk
    today = statsapi.schedule()
    for game in today:
        #print(f"Game ID: {game['game_id']} | {game['away_name']} @ {game['home_name']} | Status: {game['status']}")
        if game['status'] in ['Live', 'In Progress']:
            live_games.append({
                "game_id": game['game_id'],
                "away_team": game['away_name'],
                "home_team": game['home_name'],
                "status": game['status']
            })
            #print(f"  Found live game: {game['game_id']} - {game['away_name']} @ {game['home_name']}")
    return {"live_games": live_games}