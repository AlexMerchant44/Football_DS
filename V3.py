import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
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
average_goals_df = pd.read_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\average_goals_df.csv", low_memory=False)

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

# Merge average goals for home clubs
X = X.merge(average_goals_df, how='left', left_on='home_club_id', right_on='club_id')
X.rename(columns={'average_club_goals': 'home_club_average_goals'}, inplace=True)
X.drop(columns=['club_id'], inplace=True)  # Drop the redundant 'club_id' column

# Merge average goals for away clubs
X = X.merge(average_goals_df, how='left', left_on='away_club_id', right_on='club_id')
X.rename(columns={'average_club_goals': 'away_club_average_goals'}, inplace=True)
X.drop(columns=['club_id'], inplace=True)  # Drop the redundant 'club_id' column

'''

# Initialize dictionaries to store total goals and game count for each club
total_goals = {}
game_count = {}

# Iterate through games dataset
for index, row in X.iterrows():
    home_club = row['home_club_id']
    away_club = row['away_club_id']
    goals = row['total_goals']
    
    # Update total goals and game count for home club
    if home_club in total_goals:
        total_goals[home_club] += goals
        game_count[home_club] += 1
    else:
        total_goals[home_club] = goals
        game_count[home_club] = 1
    
    # Update total goals and game count for away club
    if away_club in total_goals:
        total_goals[away_club] += goals
        game_count[away_club] += 1
    else:
        total_goals[away_club] = goals
        game_count[away_club] = 1

# Calculate average goals per game for each club
average_goals = {}
for club_id in total_goals:
    average_goals[club_id] = total_goals[club_id] / game_count[club_id]

# Create a new dataset with club_id and average_club_goals
average_goals_df = pd.DataFrame(list(average_goals.items()), columns=['club_id', 'average_club_goals'])

# Save the filtered_game_lineups DataFrame to a CSV file
average_goals_df.to_csv("C:\\Users\\AlexM\\OneDrive\\GameDatasets\\average_goals_df.csv", index=False)

print('saved file')
'''



X1 = X[['total_valuation', 'year', 'month', 'day', 'home_club_average_goals', 'away_club_average_goals', 'hc_defenders',
         'hc_midfielders', 'hc_attackers', 'hc_forwards', 'ac_defenders', 'ac_midfielders',
         'ac_attackers', 'ac_forwards']]
y = X['total_goals']


'''
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
'''
#tune it, try classifier if not, find average goals an then say is it greater than or equal to this.
'''
#change data split
train_errors = []
val_errors = []
splits = np.arange(0.1, 0.9, 0.1)

for split in splits:
    # Split the data
    X1_train, X1_test_val, y1_train, y1_test_val = train_test_split(X1, y, test_size=split, shuffle=True, random_state=0)
    X1_val, X1_test, y1_val, y1_test = train_test_split(X1_test_val, y1_test_val, test_size=0.5, random_state=0)

    # Create a new model using only the optimal features
    rf_model1 = RandomForestRegressor(random_state=0)
    rf_model1.fit(X1_train, y1_train)

    train_error = mean_absolute_error(y1_train, rf_model1.predict(X1_train))
    val_error = mean_absolute_error(y1_val, rf_model1.predict(X1_val))

    train_errors.append(train_error)
    val_errors.append(val_error)

# Plot the errors
plt.figure()
plt.plot(splits, train_errors, 'b', label='Training MAE')
plt.plot(splits, val_errors, 'r', label='Validation MAE')
plt.xlabel('Training Data Split', fontsize=14)
plt.ylabel('Mean Absolute Error', fontsize=14)
plt.legend(fontsize=12)
plt.title('Model MAE vs. Training Data Split', fontsize=16)
plt.show()

#change n estimators
train_errors = []
val_errors = []
estimators = np.arange(100, 1000, 100)

for estimator in estimators:
    # Split the data
    X1_train, X1_test_val, y1_train, y1_test_val = train_test_split(X1, y, test_size=0.1, shuffle=True, random_state=0)
    X1_val, X1_test, y1_val, y1_test = train_test_split(X1_test_val, y1_test_val, test_size=0.5, random_state=0)

    # Create a new model using only the optimal features
    rf_model1 = RandomForestRegressor(random_state=0, n_estimators=estimator, n_jobs=-1)
    rf_model1.fit(X1_train, y1_train)

    train_error = mean_absolute_error(y1_train, rf_model1.predict(X1_train))
    val_error = mean_absolute_error(y1_val, rf_model1.predict(X1_val))

    train_errors.append(train_error)
    val_errors.append(val_error)

# Plot the errors
plt.figure()
plt.plot(estimators, train_errors, 'b', label='Training MAE')
plt.plot(estimators, val_errors, 'r', label='Validation MAE')
plt.xlabel('Training Data Split', fontsize=14)
plt.ylabel('Mean Absolute Error', fontsize=14)
plt.legend(fontsize=12)
plt.title('Model MAE vs. Training Data Split', fontsize=16)
plt.show()

'''

#change n estimators
train_errors = []
val_errors = []
min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9]

for nr in min_samples_split:
    # Split the data
    X1_train, X1_test_val, y1_train, y1_test_val = train_test_split(X1, y, test_size=0.1, shuffle=True, random_state=0)
    X1_val, X1_test, y1_val, y1_test = train_test_split(X1_test_val, y1_test_val, test_size=0.5, random_state=0)

    # Create a new model using only the optimal features
    rf_model1 = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1, min_samples_split=nr)
    rf_model1.fit(X1_train, y1_train)

    train_error = mean_absolute_error(y1_train, rf_model1.predict(X1_train))
    val_error = mean_absolute_error(y1_val, rf_model1.predict(X1_val))

    train_errors.append(train_error)
    val_errors.append(val_error)

# Plot the errors
plt.figure()
plt.plot(min_samples_split, train_errors, 'b', label='Training MAE')
plt.plot(min_samples_split, val_errors, 'r', label='Validation MAE')
plt.xlabel('Training Data Split', fontsize=14)
plt.ylabel('Mean Absolute Error', fontsize=14)
plt.legend(fontsize=12)
plt.title('Model MAE vs. min samples split', fontsize=16)
plt.show()

#somehow include club ids, change average club goals to be per year, include players somehow. make classifier
