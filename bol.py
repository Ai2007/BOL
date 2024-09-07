import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


all_seasons_path = "/Users/advaiit/Downloads/NBA Data/all_seasons.csv" #Traditional Stats
frogs_path = "/Users/advaiit/Downloads/frogs.csv" #Advanced stats

all_seasons_df = pd.read_csv(all_seasons_path)
frogs_df = pd.read_csv(frogs_path)

# player name merge
merged_df = pd.merge(all_seasons_df, frogs_df, on='Player', suffixes=('_basic', '_advanced'))
merged_df['Primary_Pos'] = merged_df['Pos_basic'].apply(lambda x: x.split('-')[0]) #get the main position

# Calculate averages for DWS, DBPM, FPG, and MPG by position
position_averages = merged_df.groupby('Primary_Pos').agg({
    'DWS': 'mean',
    'DBPM': 'mean',
    'PF': 'mean',
    'MP_basic': 'mean'
}).rename(columns={
    'PF': 'Avg_FPG',
    'MP_basic': 'Avg_MPG'
}).reset_index()


merged_df = pd.merge(merged_df, position_averages, on='Primary_Pos', suffixes=('', '_avg'))


# Weights
def calculate_weights(row):
    weight_spg = 1
    weight_bpg = 1

    # deviations
    dws_dev = row['DWS'] / row['DWS_avg']
    dbpm_dev = row['DBPM'] / row['DBPM_avg']
    fpg_dev = row['PF'] / row['Avg_FPG']
    mpg_dev = row['MP_basic'] / row['Avg_MPG']

    # Apply penalties for below average DWS and DBPM
    if dws_dev < 1:
        weight_spg *= dws_dev
        weight_bpg *= dws_dev
    if dbpm_dev < 1:
        weight_spg *= dbpm_dev
        weight_bpg *= dbpm_dev

    # Apply penalties for above average FPG
    if fpg_dev > 1:
        weight_spg *= 1 / fpg_dev
        weight_bpg *= 1 / fpg_dev

    # Apply rewards for high MPG with low FPG
    if fpg_dev < 1 and mpg_dev > 1:
        weight_spg *= mpg_dev * (1 - fpg_dev)
        weight_bpg *= mpg_dev * (1 - fpg_dev)

    return pd.Series([weight_spg, weight_bpg])


merged_df[['Weight_SPG', 'Weight_BPG']] = merged_df.apply(calculate_weights, axis=1)

# features and target variables
X = merged_df[['DWS', 'DBPM', 'PF', 'MP_basic']]
y_spg = merged_df['STL']
y_bpg = merged_df['BLK']


X_train, X_test, y_spg_train, y_spg_test, y_bpg_train, y_bpg_test = train_test_split(X, y_spg, y_bpg, test_size=0.2,
                                                                                     random_state=42)

model_spg = LinearRegression()
model_spg.fit(X_train, y_spg_train)
model_bpg = LinearRegression()
model_bpg.fit(X_train, y_bpg_train)
y_spg_pred = model_spg.predict(X_test)
y_bpg_pred = model_bpg.predict(X_test)
mse_spg = mean_squared_error(y_spg_test, y_spg_pred)
mse_bpg = mean_squared_error(y_bpg_test, y_bpg_pred)
weights_spg = model_spg.predict(X)
weights_bpg = model_bpg.predict(X)
merged_df['Weight_SPG'] = weights_spg
merged_df['Weight_BPG'] = weights_bpg
merged_df['BOL'] = (merged_df['Weight_SPG'] * merged_df['STL']) + (merged_df['Weight_BPG'] * merged_df['BLK'])

# Keep only TOT values for players on multiple teams
tot_df = merged_df[merged_df['Tm_basic'] == 'TOT']
tot_players = tot_df['Player'].unique()
filtered_df = merged_df[~merged_df['Player'].isin(tot_players) | (merged_df['Tm_basic'] == 'TOT')]
filtered_df = filtered_df.drop_duplicates(subset=['Player', 'Tm_basic'])

# Sort by BOL and get the top 20 players
top_20_bol_players = filtered_df[
    ['Player', 'Primary_Pos', 'STL', 'BLK', 'Weight_SPG', 'Weight_BPG', 'BOL']].sort_values(by='BOL',
                                                                                            ascending=False).head(20)

# Print the top 20 players
print(top_20_bol_players[['Player', 'Primary_Pos', 'STL', 'BLK', 'Weight_SPG', 'Weight_BPG', 'BOL']])
