import pandas as pd

# Adjust pandas display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Don't truncate column values
pd.set_option('display.width', 1000)  # Increase display width

filtered_game_lineups = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\filtered_game_lineups.csv", low_memory=False)

filtered_games = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\filtered_games.csv", low_memory=False)

X = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\game_features_df.csv", low_memory=False)

filtered_game_lineups_sorted = filtered_game_lineups.sort_values(by='game_id').reset_index(drop=True)
filtered_games_sorted = filtered_games.sort_values(by='game_id').reset_index(drop=True)
X_sorted = X.sort_values(by='game_id').reset_index(drop=True)

# Changing date format
filtered_games_sorted['date_formatted'] = pd.to_datetime(filtered_games_sorted['date'])

# Extract features from the datetime objects
X['year'] = filtered_games_sorted['date_formatted'].dt.year
X['month'] = filtered_games_sorted['date_formatted'].dt.month
X['day'] = filtered_games_sorted['date_formatted'].dt.day

X['home_club_id'] = filtered_games_sorted['home_club_id']
X['away_club_id'] = filtered_games_sorted['away_club_id']

X['home_club_position'] = filtered_games_sorted['home_club_position']
X['away_club_position'] = filtered_games_sorted['away_club_position']

X['home_club_formation'] = filtered_games_sorted['home_club_formation']
X['away_club_formation'] = filtered_games_sorted['away_club_formation']



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

print(filtered_games_sorted.head(1))
