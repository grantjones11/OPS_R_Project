import pandas as pd
from pybaseball import batting_stats, fielding_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Load Batting Data ---
batting_df = batting_stats(2019, 2024)
batting_df['Season'] = batting_df['Season'].astype(int)

# --- Load Fielding Data ---
fielding_frames = []
for year in range(2019, 2025):
    df = fielding_stats(year)
    if 'Pos' in df.columns:
        df['Season'] = year
        fielding_frames.append(df[['Name', 'Team', 'Season', 'Pos']])
fielding_df = pd.concat(fielding_frames, ignore_index=True)

# --- Clean Names and Team Abbreviations ---
for df in [batting_df, fielding_df]:
    df['Name'] = df['Name'].astype(str).str.strip()
    df['Team'] = df['Team'].astype(str).str.strip()

# --- Merge and Compute Primary Position ---
merged_df = pd.merge(batting_df, fielding_df, on=['Name', 'Team', 'Season'], how='left')
merged_df = merged_df.rename(columns={'Pos_y': 'Pos'})
merged_df = merged_df.dropna(subset=['Pos'])

primary_pos_df = (
    merged_df.groupby("Name")["Pos"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index().rename(columns={"Pos": "Primary_Pos"})
)
merged_df = pd.merge(merged_df, primary_pos_df, on="Name", how="left")

# --- Compute Metrics ---
for col in ['H', 'BB', 'HBP', 'R', 'OBP', 'SLG']:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

merged_df['TOB'] = merged_df['H'] + merged_df['BB'] + merged_df['HBP']
merged_df['OPS'] = merged_df['OBP'] + merged_df['SLG']

team_stats = merged_df.groupby("Team")[['R', 'TOB']].sum().reset_index()
team_stats['RSPTOB'] = team_stats['R'] / (team_stats['TOB'] + 1e-5)
league_avg_rsp = team_stats['R'].sum() / (team_stats['TOB'].sum() + 1e-5)
team_stats['RPR'] = team_stats['RSPTOB'] / league_avg_rsp

merged_df = pd.merge(merged_df, team_stats[['Team', 'RPR']], on='Team', how='left')
merged_df['OPS_R'] = merged_df['OPS'] * merged_df['RPR']

# --- Aggregate for Active Players ---
most_recent = merged_df['Season'].max()
active_df = merged_df[merged_df['Season'] == most_recent]
active_names = active_df['Name'].unique()
filtered_df = merged_df[merged_df['Name'].isin(active_names)]

filtered_df = filtered_df.sort_values(by=['Name', 'Season'])

agg = filtered_df.groupby('Name').agg({
    'R': 'sum',
    'TOB': 'sum',
    'OBP': 'mean',
    'SLG': 'mean',
    'OPS': 'mean',
    'OPS_R': 'mean',
    'Primary_Pos': lambda x: x.iloc[-1],
    'Team': lambda x: x.iloc[-1]
}).reset_index()

# --- Final Team Stats ---
team_stats_final = agg.groupby("Team")[['R', 'TOB']].sum().reset_index()
team_stats_final['RSPTOB'] = team_stats_final['R'] / (team_stats_final['TOB'] + 1e-5)
league_avg_rsp = team_stats_final['R'].sum() / (team_stats_final['TOB'].sum() + 1e-5)
team_stats_final['RPR'] = team_stats_final['RSPTOB'] / league_avg_rsp

# --- Trade Simulation Function ---
def simulate_replacement_trade(new_player, old_player, from_team, to_team, team_df):
    df = team_df.copy()
    df['R'] = df['R'].astype(float)
    df['TOB'] = df['TOB'].astype(float)

    df.loc[df['Team'] == from_team, ['R', 'TOB']] -= new_player[['R', 'TOB']].values
    df.loc[df['Team'] == from_team, ['R', 'TOB']] += old_player[['R', 'TOB']].values
    df.loc[df['Team'] == to_team, ['R', 'TOB']] -= old_player[['R', 'TOB']].values
    df.loc[df['Team'] == to_team, ['R', 'TOB']] += new_player[['R', 'TOB']].values

    df['RSPTOB'] = df['R'] / (df['TOB'] + 1e-5)
    league_avg = df['R'].sum() / (df['TOB'].sum() + 1e-5)
    df['RPR'] = df['RSPTOB'] / league_avg

    old_rpr = team_df.loc[team_df['Team'] == to_team, 'RPR'].values[0]
    new_rpr = df.loc[df['Team'] == to_team, 'RPR'].values[0]
    return new_rpr - old_rpr

# --- Generate Simulated Samples for Model Training ---
team_samples = []
all_teams = agg['Team'].unique()

for target_team in all_teams:
    starter_df = agg[agg['Team'] == target_team]
    for _, candidate in agg.iterrows():
        if candidate['Team'] == target_team or candidate['TOB'] <= 10:
            continue
        pos = candidate['Primary_Pos']
        match = starter_df[starter_df['Primary_Pos'] == pos]
        if not match.empty:
            starter = match.iloc[0]
            delta = simulate_replacement_trade(candidate, starter, candidate['Team'], target_team, team_stats_final)
            team_samples.append({
                'Target_Team': target_team,
                'Name': candidate['Name'],
                'Current_Team': candidate['Team'],
                'Primary_Pos': pos,
                'OPS_R': candidate['OPS_R'],
                'Starter_OPS_R': starter['OPS_R'],
                'OBP': candidate['OBP'],
                'SLG': candidate['SLG'],
                'Predicted_RPR_Gain': delta
            })

# --- Train Random Forest Model ---
train_df = pd.DataFrame(team_samples).dropna()

X = train_df[['OPS_R', 'Starter_OPS_R', 'OBP', 'SLG']]
y = train_df['Predicted_RPR_Gain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# --- Predict Using Trained Model ---
ml_samples = []
for target_team in all_teams:
    starter_df = agg[agg['Team'] == target_team]
    for _, candidate in agg.iterrows():
        if candidate['Team'] == target_team or candidate['TOB'] <= 10:
            continue
        pos = candidate['Primary_Pos']
        match = starter_df[starter_df['Primary_Pos'] == pos]
        if not match.empty:
            starter = match.iloc[0]
            features = pd.DataFrame([{
                'OPS_R': candidate['OPS_R'],
                'Starter_OPS_R': starter['OPS_R'],
                'OBP': candidate['OBP'],
                'SLG': candidate['SLG']
            }])
            pred_gain = rf_model.predict(features)[0]
            ml_samples.append({
                'Target_Team': target_team,
                'Name': candidate['Name'],
                'Current_Team': candidate['Team'],
                'Primary_Pos': pos,
                'OPS_R': candidate['OPS_R'],
                'Starter_OPS_R': starter['OPS_R'],
                'Predicted_RPR_Gain': pred_gain,
                'Is_Upgrade': pred_gain > 0
            })

# --- Save Results ---
ml_df = pd.DataFrame(ml_samples)
ml_recommendations = ml_df[ml_df['Is_Upgrade']].sort_values(
    by=['Target_Team', 'Predicted_RPR_Gain'], ascending=[True, False]
)
ml_recommendations.to_csv("ml_trade_recommendations.csv", index=False)

ml_recommendations.head(10)
