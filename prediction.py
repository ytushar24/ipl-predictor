# Predict Match Winner from User Input (FINAL FIXED VERSION)

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model and feature columns
model = joblib.load("./data/match_winner_model.pkl")
X_base = pd.read_csv("./data/match_winner_features.csv")
trained_columns = X_base.columns.tolist()

# Load original team names from matches.csv to decode prediction
matches = pd.read_csv(r"C:\\Users\\ytush\\Downloads\\archive\\matches.csv")
unique_teams = matches["winner"].dropna().unique()
unique_teams.sort()  # Ensures consistency with label encoding

def predict_winner(team1, team2, venue, toss_winner, toss_decision, season):
    input_data = pd.DataFrame(data=[np.zeros(len(trained_columns))], columns=trained_columns)

    fields = {
        f"season_{season}": 1,
        f"team1_{team1}": 1,
        f"team2_{team2}": 1,
        f"toss_winner_{toss_winner}": 1,
        f"toss_decision_{toss_decision}": 1,
        f"venue_{venue}": 1
    }

    print("\nActivated fields from input:")
    for col in fields:
        if col in input_data.columns:
            input_data.at[0, col] = 1
            print(f"✅ {col}")
        else:
            print(f"❌ {col} — not in model features")

    # Predict label (integer)
    pred_label = model.predict(input_data)[0]

    # Convert label to team name using sorted team list
    if pred_label < len(unique_teams):
        predicted_team = unique_teams[pred_label]
    else:
        predicted_team = f"Unknown team (label {pred_label})"

    return predicted_team

# Example usage
if __name__ == "__main__":
    result = predict_winner(
        team1="Mumbai Indians",
        team2="Royal Challengers Bangalore",
        venue="Wankhede Stadium",
        toss_winner="Royal Challengers Bangalore",
        toss_decision="bat",
        season=2019
    )
    print("\n🏆 Predicted Winner:", result)
