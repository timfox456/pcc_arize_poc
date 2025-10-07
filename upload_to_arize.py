import os
import pandas as pd
import uuid
from datetime import datetime
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
import json
from dotenv import load_dotenv
load_dotenv()

# --- 1. Configuration ---
SPACE_ID = os.environ.get("ARIZE_SPACE_ID")
#SPACE_ID = "U3BhY2U6Mjk2MTM6WndkNw=="
API_KEY = os.environ.get("ARIZE_API_KEY")
#API_KEY = "ak-75121a75-1696-434e-a05c-20cafa54bbff-lCJaNNqfk6GW_nhbPq9jtlT1NK78Ed8P"
if not SPACE_ID or not API_KEY:
    raise ValueError("ARIZE_SPACE_ID and ARIZE_API_KEY must be set as environment variables.")
MODEL_ID = "prth"
MODEL_VERSION = "1.0.8"

# --- 2. Initialize Arize Client ---
print("Initializing Arize client...")
arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)
print("Arize client initialized.")

# --- 3. Load and Merge Data ---
print("Loading and merging data...")
try:
    predictions_df = pd.read_csv("data/prediction_details.csv")
    outcomes_df = pd.read_csv("data/outcome_details_modified.csv")
    merged_df = pd.merge(predictions_df, outcomes_df, on="prediction_id", suffixes=("_pred", "_actual"))
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure the data files are in the 'data/' directory.")
    exit()
print("Data loaded and merged.")

# --- 4. Process and Clean Data ---
print("Processing and cleaning data...")

# Parse JSON features
def parse_json(json_string):
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return {}

features_df = merged_df["features"].apply(parse_json).apply(pd.Series)
merged_df = pd.concat([merged_df.drop("features", axis=1), features_df], axis=1)

# Parse prediction and outcome JSON
merged_df["prediction"] = merged_df["prediction"].apply(parse_json)
merged_df["prediction_value"] = merged_df["prediction"].apply(lambda x: x.get("predictionValue"))
merged_df["binarized_prediction"] = merged_df["prediction"].apply(lambda x: x.get("binarizedPrediction"))
merged_df["binarized_outcome"] = merged_df["prediction"].apply(lambda x: x.get("binarizedOutcome"))

# Drop rows with null binarized_outcomes before logging
merged_df.dropna(subset=["binarized_outcome"], inplace=True)



# Convert timestamp
merged_df["prediction_as_of_datetime_utc_pred"] = pd.to_datetime(merged_df["prediction_as_of_datetime_utc_pred"]).astype("int64") // 10**9

print("Data processing complete.")

# --- 5. Define Arize Schema ---
print("Defining Arize schema...")
# Get feature columns dynamically, excluding metadata and id columns
if isinstance(features_df, pd.DataFrame):
    feature_column_names = [col for col in features_df.columns if col != "median_pct_eaten"]
else:
    feature_column_names = [features_df.name]

schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="prediction_as_of_datetime_utc_pred",
    prediction_label_column_name="binarized_prediction",
    actual_label_column_name="binarized_outcome",
    feature_column_names=feature_column_names,
)
print("Schema defined.")

# --- 6. Log Data to Arize ---
print("Logging data to Arize...")

# Drop rows with null prediction_ids before logging
merged_df.dropna(subset=["prediction_id"], inplace=True)
merged_df["prediction_id"] = merged_df["prediction_id"].astype(str)
merged_df.reset_index(drop=True, inplace=True)


response = arize_client.log(
    dataframe=merged_df,
    schema=schema,
    model_id=MODEL_ID,
    model_version=MODEL_VERSION,
    model_type=ModelTypes.SCORE_CATEGORICAL,
    environment=Environments.PRODUCTION,
)

if response.status_code == 200:
    print(f"✅ Successfully logged data for model {MODEL_ID} to Arize!")
else:
    print(f'❌ Logging failed with status code {response.status_code} and message "{response.text}"')

print("Script finished.")
