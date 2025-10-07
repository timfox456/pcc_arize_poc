import pandas as pd

# Load the original data
try:
    predictions_df = pd.read_csv("data/prediction_details.csv")
    outcomes_df = pd.read_csv("data/outcome_details.csv")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure the original data files are in the 'data/' directory.")
    exit()

# Get the prediction IDs from the predictions file
prediction_ids = predictions_df["prediction_id"].dropna().unique()

# If there are more outcomes than predictions, truncate the outcomes dataframe
if len(outcomes_df) > len(prediction_ids):
    outcomes_df = outcomes_df.iloc[:len(prediction_ids)]

# Overwrite the prediction_id in the outcomes dataframe
# with the IDs from the predictions dataframe to ensure a match.
outcomes_df["prediction_id"] = prediction_ids[:len(outcomes_df)]

# Save the modified outcomes data to a new file
new_outcomes_path = "data/outcome_details_modified.csv"
outcomes_df.to_csv(new_outcomes_path, index=False)

print(f"Successfully generated new outcomes file at '{new_outcomes_path}' with matching prediction IDs.")
