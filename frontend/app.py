import streamlit as st
import requests
import os
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ---------------------------
# Backend Helpers
# ---------------------------

def fetch_prediction():
    response = requests.post(f"{BACKEND_URL}/predict")
    response.raise_for_status()
    return response.json()


def wait_for_backend():
    for _ in range(10):
        try:
            r = requests.get(f"{BACKEND_URL}/health")
            r.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            time.sleep(2)
    return False


# ---------------------------
# App Logic
# ---------------------------

def initialize_state():
    if "probs" not in st.session_state:
        data = fetch_prediction()
        st.session_state.probs = data["probabilities"]
        st.session_state.pitch = data["pitch"]
        st.session_state.selected_bet = None


def render_pitch_info():
    st.subheader("Current Pitch Info")
    st.write(st.session_state.pitch)

def prob_to_betting_odds(p):
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    else:
        return round(100 * (1 - p) / p)

def add_vig(probs, vig=0.05):
    """
    probs: list of probabilities [strike, ball, hit]
    vig: bookmaker margin (0.05 = 5%)
    """
    total = sum(probs)
    target_total = total * (1 + vig)  # increase sum to include vig
    probs_with_vig = [p / total * target_total for p in probs]
    return probs_with_vig


def render_odds_buttons(vig=0.08):
    st.subheader("Current Betting Odds")

    # Grab the current probabilities from session state
    probs = st.session_state.probs

    # Apply vig: convert dict values to list, then back to dict
    labels = list(probs.keys())
    probs_list = list(probs.values())
    probs_with_vig = add_vig(probs_list, vig=vig)
    probs_with_vig_dict = {label: p for label, p in zip(labels, probs_with_vig)}

    cols = st.columns(len(probs_with_vig_dict))

    for i, (label, prob) in enumerate(probs_with_vig_dict.items()):
        is_selected = st.session_state.selected_bet == label

        american_odds = prob_to_betting_odds(prob)
        button_label = f"{label.upper()} ({american_odds:+})"

        if cols[i].button(
            button_label,
            key=f"bet_{label}",
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.selected_bet = label


def render_place_bet():
    if st.session_state.selected_bet:
        st.success(f"Selected: {st.session_state.selected_bet}")

        if st.button("Place Bet"):
            data = fetch_prediction()
            st.session_state.probs = data["probabilities"]
            st.session_state.pitch = data["pitch"]
            st.session_state.selected_bet = None
            st.rerun()


# ---------------------------
# Main App
# ---------------------------

def run_app():
    st.title("Live Pitch Odds Interface")
    st.subheader("An app to demonstrate live pitch betting odds")

    initialize_state()

    render_pitch_info()
    render_odds_buttons()
    render_place_bet()


if __name__ == "__main__":
    wait_for_backend()
    run_app()