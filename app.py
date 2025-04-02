# Streamlit IPL Match Winner Predictor App

import streamlit as st
import pandas as pd
from prediction import predict_winner

# Load venue list from matches.csv
matches = pd.read_csv("data/matches.csv")
venues = sorted(matches["venue"].dropna().unique().tolist())

# Set up the Streamlit UI
st.set_page_config(page_title="IPL Match Winner Predictor", layout="centered")
st.title("🏏 IPL Match Winner Predictor")
st.markdown("Enter match details to predict the winner based on historical IPL data.")

# Full team names as used in matches.csv
teams = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad"
]

# Input fields
team1 = st.selectbox("Team 1 (Batting First)", teams)
team2 = st.selectbox("Team 2 (Batting Second)", [t for t in teams if t != team1])

venue = st.selectbox("Venue", venues)

season = st.selectbox("Season", list(range(2008, 2024)))

toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.radio("Toss Decision", ["bat", "field"])

# Predict button
if st.button("Predict Winner"):
    winner = predict_winner(
        team1=team1,
        team2=team2,
        venue=venue,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        season=season
    )
    st.success(f"🏆 Predicted Winner: {winner}")
