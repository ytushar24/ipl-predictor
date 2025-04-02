# Feature Engineering for Match Winner Prediction

import pandas as pd
import numpy as np
import os

# Update this path to point to your actual matches.csv file
MATCHES_PATH = r"C:\Users\ytush\Downloads\archive\matches.csv"

# Check if file exists
if not os.path.exists(MATCHES_PATH):
    raise FileNotFoundError(f"File not found: {MATCHES_PATH}")

# Load matches dataset
matches = pd.read_csv(MATCHES_PATH)

# --- STEP 1: Basic cleanup ---
# Drop matches with no result or missing winner
df = matches.dropna(subset=["winner"])
df = df[df["result"] != "no result"]

# Simplify column names (now including 'season')
cols_needed = ["season", "team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]
df = df[cols_needed].copy()

# Convert season from 'YYYY/YY' to integer year (e.g., '2020/21' -> 2020)
df["season"] = df["season"].astype(str).str.extract(r'(\d{4})').astype(int)

# --- STEP 2: Encode categorical variables ---
# One-hot encode all categorical features including season
df = pd.get_dummies(df, columns=["season", "team1", "team2", "toss_winner", "toss_decision", "venue"], drop_first=True)

# --- STEP 3: Create target column ---
df["label"] = df["winner"]
df = df.drop(columns=["winner"])

# Encode target variable (label encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# --- STEP 4: Output features and labels ---
X = df.drop(columns=["label"])
y = df["label"]

# Make sure data folder exists
os.makedirs("./data", exist_ok=True)

# Save for modeling phase
X.to_csv("./data/match_winner_features.csv", index=False)
y.to_csv("./data/match_winner_labels.csv", index=False)

print("✅ Feature engineering completed. Features and labels saved to ./data/")
print(f"Shape of X: {X.shape}")
print(f"Classes: {list(le.classes_)}")
