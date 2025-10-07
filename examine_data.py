
import pandas as pd
import json

def examine_data():
    """
    Analyzes the prediction and outcome data to identify potential issues.
    """
    print("--- Analyzing Data Quality ---")

    try:
        predictions_df = pd.read_csv("data/prediction_details.csv")
        outcomes_df = pd.read_csv("data/outcome_details.csv")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the data files are in the 'data/' directory.")
        return

    # --- 1. Row Counts ---
    print(f"Total rows in prediction_details.csv: {len(predictions_df)}")
    print(f"Total rows in outcome_details.csv: {len(outcomes_df)}")

    # --- 2. Missing Required Data ---
    missing_required_data = predictions_df["is_missing_required_data"].sum()
    missing_percentage = (missing_required_data / len(predictions_df)) * 100
    print(f"\nPercentage of predictions with missing required data: {missing_percentage:.2f}%")

    # --- 3. Null Outcomes ---
    def count_null_outcomes(df):
        null_outcomes = 0
        for _, row in df.iterrows():
            try:
                prediction_json = json.loads(row["prediction"])
                if prediction_json.get("binarizedOutcome") is None:
                    null_outcomes += 1
            except (json.JSONDecodeError, TypeError):
                null_outcomes += 1
        return null_outcomes

    null_outcomes_count = count_null_outcomes(outcomes_df)
    null_outcomes_percentage = (null_outcomes_count / len(outcomes_df)) * 100
    print(f"Percentage of null outcomes: {null_outcomes_percentage:.2f}%")

    # --- 4. Feature Sparsity ---
    print("\n--- Feature Sparsity Analysis ---")
    def analyze_feature_sparsity(df):
        feature_null_counts = {}
        total_rows = len(df)

        for _, row in df.iterrows():
            try:
                features = json.loads(row["features"])
                for feature, value in features.items():
                    if feature not in feature_null_counts:
                        feature_null_counts[feature] = 0
                    if value is None:
                        feature_null_counts[feature] += 1
            except (json.JSONDecodeError, TypeError):
                continue

        print("Percentage of null values for each feature:")
        for feature, null_count in feature_null_counts.items():
            percentage = (null_count / total_rows) * 100
            print(f"- {feature}: {percentage:.2f}%")

    analyze_feature_sparsity(predictions_df)

if __name__ == "__main__":
    examine_data()
