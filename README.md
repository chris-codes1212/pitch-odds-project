
# Pitch Odds Project
*A real-time pitch-by-pitch MLB betting odds simulator powered by machine learning*

## Overview

This project simulates a **live pitch-by-pitch betting interface for Major League Baseball**. The application predicts the probability of outcomes for each pitch (strike, ball, hit, or foul) and converts those probabilities into betting odds in a user-facing interface.

The goal is to demonstrate what a **micro-betting interface for baseball** might look like if users could place wagers on every pitch of a game.

The system combines:

- machine learning modeling
- real-time API predictions
- a frontend betting interface
- cloud deployment
- CI/CD automation

### Application URL

http://mlbalb-1808435537.us-east-1.elb.amazonaws.com/

---

# Application Architecture

Frontend (Streamlit)
        ↓
Application Load Balancer
        ↓
Backend API (FastAPI)
        ↓
ML Model (XGBoost pipeline)
        ↓
Held-out pitch dataset

The backend serves predictions for each pitch, while the frontend displays betting odds and allows users to simulate placing bets.

---

# Frontend

The frontend is built with **Streamlit** and provides an interactive interface where users can:

- view the current pitch state
- see updated betting odds
- simulate placing bets on outcomes

Odds are derived from model probabilities and adjusted using a sportsbook margin (vig).

Example UI flow:

1. A pitch context is loaded
2. The backend returns probabilities for pitch outcomes
3. Probabilities are converted to American betting odds
4. Users choose a bet and advance to the next pitch

Each browser session receives a unique user ID so the backend can track simulation state.

---

# Backend API

The backend is implemented using **FastAPI** and exposes endpoints for:

GET  /health  
POST /predict

The `/predict` endpoint:

1. Retrieves the current pitch index for a user session
2. Loads the corresponding pitch from the dataset
3. Generates prediction probabilities using the trained model
4. Filters and normalizes probabilities
5. Advances the simulation to the next pitch

---

# Machine Learning Model

The prediction model is an **XGBoost multi-class classifier** that estimates probabilities for:

- strike
- ball
- hit
- non-strike foul

Model training and deployment use **Weights & Biases** artifact versioning.

The backend loads the **production model artifact** at startup.

---

## Feature Engineering

Several feature engineering techniques were implemented:

### Empirical Bayes Shrinkage

Player-level performance rates are estimated using a **beta-binomial empirical Bayes approach** to stabilize estimates for players with limited observations.

Examples include:

- pitcher strike rate
- pitcher hit rate
- batter hit rate

These are implemented as custom **scikit-learn transformers**.

### Player Archetype Clustering

Pitchers and batters are clustered using **K-Means** based on performance rate features. Distances to cluster centers are used as additional features to represent player archetypes.

### Additional Engineered Features

Other features include:

- count state (`balls-strikes`)
- game context
- fielding alignment
- handedness matchups

The feature engineering logic is encapsulated inside a **scikit-learn pipeline** so it can be used consistently during both training and inference.

---

# Dataset

Pitch data is obtained using the **Statcast API via pybaseball**.

The training pipeline:

1. pulls pitch-level data
2. maps pitch descriptions to simplified outcomes
3. splits data into train / validation / test sets

Outcome classes:

- strike
- ball
- hit
- non-strike foul

The live application currently runs on a **held-out dataset of historical pitches**.

---

# CI/CD Pipeline

The project is deployed using a fully automated **AWS CI/CD pipeline**.

Pipeline stages:

GitHub push (main)
        ↓
Source stage
        ↓
Test stage (CodeBuild)
        ↓
Build stage (Docker images)
        ↓
Push images to ECR
        ↓
Deploy stage (ECS Fargate)

### Test Stage

Unit tests are executed using CodeBuild before new images are built.

### Build Stage

Two Docker images are built:

- frontend container
- backend container

Images are tagged with `:latest` and pushed to **Amazon ECR**.

### Deploy Stage

Deployment updates the ECS service by:

1. creating a new task definition
2. referencing the new image versions
3. starting new containers on **AWS Fargate**

The application is exposed through an **Application Load Balancer**.

---

# Repository Structure

backend/
    main.py
    sessions.py
    utils.py
    PitchSimulation.py
    requirements.txt

frontend/
    app.py
    requirements.txt

src/
    data_loader.py
    preprocessing.py
    transformers.py
    pipeline.py
    model.py
    train.py

notebooks/
    pitch_by_pitch.ipynb

tests/
    test_main_predict.py

buildspec.yml
buildspec.test.yml

---

# Local Development

### Backend

pip install -r backend/requirements.txt
uvicorn backend.main:app --reload

### Frontend

pip install -r frontend/requirements.txt
streamlit run frontend/app.py

---

# Future Improvements

### Real-time pitch data

Replace the held-out dataset with **live Statcast pitch feeds** so the application can simulate betting during real MLB games.

### Simulated bankroll system

Allow users to:

- receive a starting bankroll
- place simulated bets
- track winnings and losses

### Additional feature engineering

Future modeling improvements may include:

- pitch type information
- pitcher fatigue effects
- batter/pitcher matchup history
- pitch sequencing

### Model calibration

Improve probability calibration for betting applications.

---

# Technologies Used

Machine Learning

- XGBoost
- scikit-learn
- pybaseball
- pandas
- numpy

Backend

- FastAPI
- DynamoDB
- boto3

Frontend

- Streamlit

Infrastructure

- AWS CodePipeline
- AWS CodeBuild
- Amazon ECS (Fargate)
- Amazon ECR
- Application Load Balancer

Experiment Tracking

- Weights & Biases

---

# Author

Christopher Thompson

Machine Learning | Data Science | Cloud Engineering
