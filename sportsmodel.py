import pandas as pd

game_lineups = pd.read_csv("C:\\Users\AlexM\\OneDrive\\Game Datasets\\game_lineups.csv")

games = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\Game Datasets\\games.csv")

# Changing data format
game_lineups['date_formatted'] = pd.to_datetime(game_lineups['date'])

# Extract features from the datetime objects
game_lineups['year'] = game_lineups['date_formatted'].dt.year
game_lineups['month'] = game_lineups['date_formatted'].dt.month
game_lineups['day'] = game_lineups['date_formatted'].dt.day

# Get the counts of each unique id in the column
id_counts = game_lineups['game_id'].value_counts()

# Find game IDs that occur exactly 36 times
game_ids_to_keep = id_counts[id_counts == 36].index

# Filter the DataFrames to keep only rows with these game IDs
filtered_game_lineups = game_lineups[game_lineups['game_id'].isin(game_ids_to_keep)]
filtered_games = games[games['game_id'].isin(game_ids_to_keep)]

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
print(game_features_df)