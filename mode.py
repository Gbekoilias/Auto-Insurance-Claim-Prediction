import json

file_path = 'C:/Users/DONKAMS/Downloads/Project_STA2017/nigeria-state-and-lgas.json'

try:
    with open(file_path) as f:
        data = json.load(f)
    # Proceed with using the 'data' variable containing the JSON content
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except PermissionError:
    print("Permission denied to read the file.")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd

# Assuming df is your original DataFrame and df3 is the DataFrame with correct LGA-State mapping

# Merge df and df3 on 'LGA_Name' to get the correct 'State' values
merged_df = df.merge(df3, how='left', left_on='LGA_Name', right_on='LGA')

# Replace the 'State' column in df with the correct 'State' values from df3
df['State'] = merged_df['state']  # Assuming the column name in df3 for State is 'state'

# Display the updated DataFrame 'df' with corrected 'State' values
print(df)
# Create a mapping dictionary from df3 where LGA is the key and State is the value
mapping_dict = df3.set_index('LGA')['state'].to_dict()

# Replace incorrect State values in df based on the mapping dictionary
df['State'] = df['LGA_Name'].map(mapping_dict).fillna(df['State'])

# Display the updated DataFrame 'df' with corrected 'State' values
print(df)

from fuzzywuzzy import process

# Assuming df is your original DataFrame and df3 is the DataFrame with correct LGA-State mapping

# Create a function to find similar substrings between LGA names
def find_similar_lga(lga_name, lga_list):
    # Find the closest match with a similarity score
    match = process.extractOne(lga_name, lga_list)
    # If similarity score is above a certain threshold, return the matched value
    if match[1] > 80:  # Adjust the similarity threshold as needed (here, 80 is used)
        return match[0]
    else:
        return None

# Create a mapping dictionary using fuzzy string matching
lga_list_df3 = df3['LGA'].tolist()

# Iterate through each LGA_Name in df and find similar LGAs in df3
for index, row in df.iterrows():
    similar_lga = find_similar_lga(row['LGA_Name'], lga_list_df3)
    if similar_lga:
        # Update the State column in df with the corresponding State from df3
        corresponding_state = df3[df3['LGA'] == similar_lga]['state'].values[0]
        df.at[index, 'State'] = corresponding_state

# Display the updated DataFrame 'df' with corrected 'State' values
print(df)

from fuzzywuzzy import process
import json

# Load the LGA-State mapping from the JSON file
with open('your_file_path.json', 'r') as file:
    lga_state_dict = json.load(file)

# Assuming df is your original DataFrame

# Create a function to find similar substrings between LGA names
def find_similar_lga(lga_name, lga_list):
    match = process.extractOne(lga_name, lga_list)
    if match[1] > 80:
        return match[0]
    else:
        return None

# Create a list of LGAs from the loaded dictionary
lga_list_dict = list(lga_state_dict.keys())

# Iterate through each LGA_Name in df and find similar LGAs in the dictionary keys
for index, row in df.iterrows():
    similar_lga = find_similar_lga(row['LGA_Name'], lga_list_dict)
    if similar_lga:
        # Update the State column in df with the corresponding State from the dictionary
        corresponding_state = lga_state_dict[similar_lga]
        df.at[index, 'State'] = corresponding_state

# Display the updated DataFrame 'df' with corrected 'State' values
print(df)


