import pandas as pd

# File paths for the BTC data files (update these if needed)
file_2017 = 'BTC-2017min.csv'
file_2018 = 'BTC-2018min.csv'
file_2019 = 'BTC-2019min.csv'
file_2020 = 'BTC-2020min.csv'
file_2021 = 'BTC-2021min.csv'

# Step 1: Load the BTC datasets for each year
btc_2017 = pd.read_csv(file_2017)
btc_2018 = pd.read_csv(file_2018)
btc_2019 = pd.read_csv(file_2019)
btc_2020 = pd.read_csv(file_2020)
btc_2021 = pd.read_csv(file_2021)

# Step 2: Convert 'unix' to 'date' (if necessary) and drop the 'unix' and 'symbol' columns
btc_2017['date'] = pd.to_datetime(btc_2017['unix'], unit='s')
btc_2017 = btc_2017.drop(columns=['unix', 'symbol'])

btc_2018['date'] = pd.to_datetime(btc_2018['unix'], unit='s')
btc_2018 = btc_2018.drop(columns=['unix', 'symbol'])

btc_2019['date'] = pd.to_datetime(btc_2019['unix'], unit='s')
btc_2019 = btc_2019.drop(columns=['unix', 'symbol'])

btc_2020['date'] = pd.to_datetime(btc_2020['unix'], unit='s')
btc_2020 = btc_2020.drop(columns=['unix', 'symbol'])

btc_2021['date'] = pd.to_datetime(btc_2021['unix'], unit='s')
btc_2021 = btc_2021.drop(columns=['unix', 'symbol'])

# Step 3: Combine data from 2017, 2018, 2019, and 2020 into a single DataFrame (all columns except 'unix' and 'symbol')
combined_train_data = pd.concat([btc_2017, btc_2018, btc_2019, btc_2020], axis=0).reset_index(drop=True)

# Step 4: Calculate the split for the train and test sets using 70% of 2021 as the test set
test_split_2021 = int(len(btc_2021) * 0.7)

# Step 5: Use 70% of 2021 for the test set and the remaining 30% for the training set
train_2021 = btc_2021.iloc[test_split_2021:]
test_2021 = btc_2021.iloc[:test_split_2021]

# Step 6: Combine the training portion of 2021 with the previous years' data to create the final training set
final_train_data = pd.concat([combined_train_data, train_2021], axis=0).reset_index(drop=True)

# Step 7: Save the final training and test datasets as CSV files (with all columns except 'unix' and 'symbol')
final_train_data.to_csv('train_data.csv', index=False)
test_2021.to_csv('test_data.csv', index=False)

# Optional: Print confirmation
print("Train and test datasets have been successfully created and saved as 'train_data.csv' and 'test_data.csv'")
