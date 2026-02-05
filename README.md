# MLB Live-Pitch Odds Model

This repository contains a machine learning pipeline designed to generate real-time, pitch-by-pitch betting odds for Major League Baseball (MLB). By leveraging high-fidelity Statcast data, the model predicts the probability of discrete pitch outcomes (Strikes, Balls, Hits) to provide a foundation for live betting interfaces.

## 🚀 Project Overview
Traditional baseball betting often focuses on game-level or inning-level outcomes. This project aims to increase fan engagement by modeling the **stochastic** nature of individual pitches. The core challenge addressed here is the high degree of variance in baseball data; we solve this using advanced feature engineering rather than just complex algorithms.

## 🛠️ Current Status: Phase 1 (Core ML Pipeline)
At present, the project is centered around a comprehensive Jupyter Notebook (`pitch_by_pitch.ipynb`) that handles:
* **Data Acquisition:** Automated ingestion of MLB Statcast data via `pybaseball`.
* **Empirical Bayes Shrinkage:** A Beta-Binomial framework used to "shrink" noisy, small-sample player statistics toward league averages to produce "true-skill" features.
* **Unsupervised Learning (K-Means):** Clustering pitchers and batters into distinct "Archetypes" (e.g., "Zone-Pounders" vs. "Wild Aces") to capture non-linear playing styles.
* **Supervised Learning:** A Random Forest Classifier optimized for **Negative Log Loss** to ensure well-calibrated outcome probabilities.

## 📊 Key Results
* **Performance:** Model Log Loss of **1.18**, successfully outperforming the naive baseline of 1.22.
* **Feature Importance:** The model identified "Situational Gravity" (the count) as the primary driver of outcomes, with two-strike counts (1-2, 3-2, etc.) accounting for over 50% of the predictive power.

## 📁 Repository Structure
```text
├── pitch_by_pitch.ipynb    # Core research, feature engineering, and model training
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
Note: This repository is actively expanding to include a dedicated backend (API) and frontend (Dashboard) service.


⚙️ Installation & Usage
Prerequisites
Python 3.11+

Setup
Clone the repository:

Bash
git clone [https://github.com/your-username/mlb-pitch-odds.git](https://github.com/your-username/mlb-pitch-odds.git)
cd mlb-pitch-odds
Install dependencies: It is recommended to use a virtual environment. You can now install all required packages (including pybaseball, scikit-learn, xgboost, and pandas) using the provided requirements file:

Bash
python3.11 -m pip install -r requirements.txt
Run the model: Launch the notebook to fetch data and begin training:

Bash
jupyter notebook pitch_by_pitch.ipynb

🛤️ Roadmap
[ ] Phase 2: Integrate physical metrics (Mean Spin Rate, Extension) using Normal-Normal EB Shrinkage.

[ ] Backend: Develop a FastAPI service to serve model predictions in real-time.

[ ] Frontend: Build a React-based "Live Game" dashboard to visualize shifting odds during an at-bat.

[ ] Monitoring: Implement Weights & Biases (W&B) for experiment tracking and Data Drift detection.
