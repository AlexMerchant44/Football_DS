import pandas as pd

# Adjust pandas display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Don't truncate column values
pd.set_option('display.width', 1000)  # Increase display width

game_lineups = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\game_lineups.csv")

games = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\games.csv")

X = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\game_features_df.csv")

# Changing date format
game_lineups['date_formatted'] = pd.to_datetime(game_lineups['date'])

# Extract features from the datetime objects
game_lineups['year'] = game_lineups['date_formatted'].dt.year
game_lineups['month'] = game_lineups['date_formatted'].dt.month
game_lineups['day'] = game_lineups['date_formatted'].dt.day

filtered_game_lineups = game_lineups[~game_lineups['type'].str.contains('substitutes', case=False, na=False)]

# Get the counts of each unique id in the column
id_counts = filtered_game_lineups['game_id'].value_counts()

# Find game IDs that occur exactly 36 times
game_ids_to_keep = id_counts[id_counts == 22].index

# Filter the DataFrames to keep only rows with these game IDs
filtered_game_lineups = filtered_game_lineups[game_lineups['game_id'].isin(game_ids_to_keep)]
filtered_games = games[games['game_id'].isin(game_ids_to_keep)]

#filtered_game_lineups.to_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\filtered_game_lineups.csv", index=False)
#filtered_games.to_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\filtered_games.csv", index=False)

#print('saved file')

'''

# Group by 'gameid' and aggregate players into lists
grouped = filtered_game_lineups.groupby('game_id')['player_id'].apply(list)
# Create the dictionary with game IDs as keys and player lists as values
game_players_dict = grouped.to_dict()

# Initialize an empty DataFrame with game IDs as index
game_features_df = pd.DataFrame(index=game_players_dict.keys())

# Iterate through the dictionary and populate the DataFrame
for game_id, player_list in game_players_dict.items():
    for i, player_id in enumerate(player_list, start=1):
        column_name = f'player{i}'  # Generate column names like 'player1', 'player2', etc.
        game_features_df.loc[game_id, column_name] = player_id

# Reset index to make 'game_id' a regular column
game_features_df = game_features_df.reset_index().rename(columns={'index': 'game_id'})

# Print the resulting DataFrame

# Save the filtered_game_lineups DataFrame to a CSV file
game_features_df.to_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\game_features_df.csv", index=False)

print('saved file')

'''


X['year'] = filtered_game_lineups['year']
X['month'] = filtered_game_lineups['month']
X['day'] = filtered_game_lineups['day']

X['home_club_id'] = filtered_games['home_club_id']
X['away_club_id'] = filtered_games['away_club_id']

X['home_club_position'] = filtered_games['home_club_position']
X['away_club_position'] = filtered_games['away_club_position']

X['home_club_formation'] = filtered_games['home_club_formation']

'''
print(X.shape)
print(filtered_game_lineups.shape)
print(filtered_games.shape)

print(filtered_games.loc[filtered_games['game_id'] == 2317258].index)

'''

'''

# Fill NaN values with empty strings in the home_club_formation column
X['home_club_formation'] = X['home_club_formation'].fillna('')

# Define a regular expression to match valid formations (e.g., "4-3-2-1" or "3-4-3")
valid_formation_pattern = r'^\d+(-\d+){2,3}$'

# Filter the DataFrame to keep only rows with valid formations
X = X[X['home_club_formation'].str.match(valid_formation_pattern)]

# Split the valid formations into separate columns
formation_split = X['home_club_formation'].str.split('-', expand=True)

# Rename the columns appropriately, accommodating up to 4 components
formation_split.columns = ['defenders', 'midfielders', 'attackers', 'forwards']

# Fill missing values with 0 if any formation has fewer than 4 components
formation_split = formation_split.apply(pd.to_numeric)

# Concatenate the split columns with the valid DataFrame
X = pd.concat([X, formation_split], axis=1)

# Drop the original formation column if desired
X.drop('home_club_formation', axis=1, inplace=True)


#redo x to have 11 players only. Fix formation settings

'''