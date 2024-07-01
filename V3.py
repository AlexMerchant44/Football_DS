import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Adjust pandas display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Don't truncate column values
pd.set_option('display.width', 1000)  # Increase display width

player_valuations = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\player_valuations.csv", low_memory=False)
filtered_games = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\filtered_games.csv",low_memory=False)
filtered_game_lineups = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\filtered_game_lineups.csv", low_memory=False)
filtered_game_events = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\filtered_game_events.csv", low_memory=False)

sorted_filtered_games = filtered_games.sort_values(by='game_id').reset_index(drop=True)
sorted_filtered_game_lineups = filtered_game_lineups.sort_values(by='game_id').reset_index(drop=True)
sorted_filtered_game_events = filtered_game_events.sort_values(by='game_id').reset_index(drop=True)

X = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\X_df.csv", low_memory=False)
'''
# Group by 'game_id' and aggregate player IDs into lists
game_lineups_dict = sorted_filtered_game_lineups.groupby('game_id')['player_id'].apply(list).to_dict()

total_valuations = []

for game_id, player_ids in game_lineups_dict.items():
    total_valuation = player_valuations.loc[player_valuations['player_id'].isin(player_ids), 'market_value_in_eur'].sum()
    total_valuations.append({'game_id': game_id, 'total_valuation': total_valuation})

# Step 3: Create the new DataFrame X
X = pd.DataFrame(total_valuations)

# Save the filtered_game_lineups DataFrame to a CSV file
X.to_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\X_df.csv", index=False)

print('saved file')

'''
# Changing date format
sorted_filtered_games['date_formatted'] = pd.to_datetime(sorted_filtered_games['date'])

# Extract features from the datetime objects
X['year'] = sorted_filtered_games['date_formatted'].dt.year
X['month'] = sorted_filtered_games['date_formatted'].dt.month
X['day'] = sorted_filtered_games['date_formatted'].dt.day

X['home_club_id'] = sorted_filtered_games['home_club_id']
X['away_club_id'] = sorted_filtered_games['away_club_id']

X['home_club_formation'] = sorted_filtered_games['home_club_formation']
X['away_club_formation'] = sorted_filtered_games['away_club_formation']


# Ensure the 'home_club_formation' column is treated as strings and handle NaNs
X['home_club_formation'] = X['home_club_formation'].astype(str).fillna('')

# Delete anything after a space in the 'home_club_formation' column
X['home_club_formation'] = X['home_club_formation'].str.split(' ').str[0]

# Define a regular expression to extract the formation components
formation_pattern = r'^(\d+)-(\d+)-(\d+)(?:-(\d+))?$'

# Extract the formation components into new columns
formation_cols = X['home_club_formation'].str.extract(formation_pattern, expand=True)

# Rename the columns to defenders, midfielders, attackers, forwards
formation_cols.columns = ['hc_defenders', 'hc_midfielders', 'hc_attackers', 'hc_forwards']

# Replace NaNs in 'forwards' with 0 and convert all columns to integers
formation_cols = formation_cols.fillna(0).astype(int)

# Concatenate the new columns with the original DataFrame
X = pd.concat([X, formation_cols], axis=1)

# Ensure the 'home_club_formation' column is treated as strings and handle NaNs
X['away_club_formation'] = X['away_club_formation'].astype(str).fillna('')

# Delete anything after a space in the 'home_club_formation' column
X['away_club_formation'] = X['away_club_formation'].str.split(' ').str[0]

# Define a regular expression to extract the formation components
formation_pattern = r'^(\d+)-(\d+)-(\d+)(?:-(\d+))?$'

# Extract the formation components into new columns
formation_cols = X['away_club_formation'].str.extract(formation_pattern, expand=True)

# Rename the columns to defenders, midfielders, attackers, forwards
formation_cols.columns = ['ac_defenders', 'ac_midfielders', 'ac_attackers', 'ac_forwards']

# Replace NaNs in 'forwards' with 0 and convert all columns to integers
formation_cols = formation_cols.fillna(0).astype(int)

# Concatenate the new columns with the original DataFrame
X = pd.concat([X, formation_cols], axis=1)

X['total_goals'] = sorted_filtered_games['home_club_goals'] + sorted_filtered_games['away_club_goals']

X1 = X[['total_valuation', 'year', 'month', 'day', 'home_club_id', 'away_club_id', 'hc_defenders',
         'hc_midfielders', 'hc_attackers', 'hc_forwards', 'ac_defenders', 'ac_midfielders',
         'ac_attackers', 'ac_forwards']]
y = X['total_goals']

# Split the data in an 80:20 split, randomly assigning 80% of the rows to the training data
X_train, X_test_val, y_train, y_test_val = train_test_split(X1,y, test_size=0.2, shuffle = True, random_state=0)

# Split the leftover 20% data in half and assign randomly 10% to validation and 10% to testing
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=0)

rf_model = RandomForestRegressor(random_state=0, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predict on the testing data
y_pred = rf_model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")


