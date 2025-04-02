# IPL Starter EDA Notebook with Inline Comments

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- STEP 0: Define paths and validate ---
# These should point to your actual dataset location
MATCHES_PATH = r"C:\Users\ytush\Downloads\archive\matches.csv"
DELIVERIES_PATH = r"C:\Users\ytush\Downloads\archive\deliveries.csv"

# Check if files exist, raise error if not
if not os.path.exists(MATCHES_PATH):
    raise FileNotFoundError(f"File not found: {MATCHES_PATH}")
if not os.path.exists(DELIVERIES_PATH):
    raise FileNotFoundError(f"File not found: {DELIVERIES_PATH}")

# Load the IPL datasets into pandas DataFrames
matches = pd.read_csv(MATCHES_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)

# Set the plot style for better visuals
sns.set(style="whitegrid")

# --- STEP 1: Understand the Data ---
# Check the number of rows and columns in each dataset
print("Matches Dataset Shape:", matches.shape)
print("Deliveries Dataset Shape:", deliveries.shape)

# Display the column names for both datasets
print("\nMatches Columns:", matches.columns.tolist())
print("\nDeliveries Columns:", deliveries.columns.tolist())

# Show first few rows to get a glimpse of the structure and values
print("\nSample Matches Data:")
print(matches.head())

print("\nSample Deliveries Data:")
print(deliveries.head())

# --- STEP 2: Missing Values ---
# Check for missing values in each column of the matches dataset
print("\nMissing Values in Matches:")
print(matches.isnull().sum())

# Check for missing values in each column of the deliveries dataset
print("\nMissing Values in Deliveries:")
print(deliveries.isnull().sum())

# --- STEP 3: Matches per Season ---
# Plot the number of matches played in each IPL season
plt.figure(figsize=(10,5))
sns.countplot(data=matches, x="season")
plt.title("Number of Matches per Season")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# --- STEP 4: Toss Winner vs Match Winner ---
# Create a new column to check if toss winner also won the match
matches["toss_match_win"] = matches["toss_winner"] == matches["winner"]

# Calculate the proportion of such matches
toss_win_rate = matches["toss_match_win"].mean()
print(f"\nPercentage of times toss winner won the match: {toss_win_rate * 100:.2f}%")

# --- STEP 5: Most Successful Teams ---
# Plot the number of matches won by each team
plt.figure(figsize=(10,5))
matches['winner'].value_counts().plot(kind='bar')
plt.title("Most Match Wins by Team")
plt.xlabel("Team")
plt.ylabel("Wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- STEP 6: Venue Analysis ---
# Display top 10 venues where most IPL matches were played
plt.figure(figsize=(10,5))
matches['venue'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Venues with Most Matches")
plt.xlabel("Venue")
plt.ylabel("Match Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- STEP 7: Example Player Performance (can be expanded later) ---
# Calculate the total runs scored by each batsman and display the top 10
batsman_runs = deliveries.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).head(10)

# Print the top 10 run scorers
print("\nTop 10 Run Scorers:")
print(batsman_runs)

# Plot the top 10 run scorers as a bar chart
batsman_runs.plot(kind='bar')
plt.title("Top 10 Batsmen (Total Runs)")
plt.xlabel("Player")
plt.ylabel("Runs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
