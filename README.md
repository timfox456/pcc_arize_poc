# Arize AI data pipeline

This repository implements a synthetic fraud detection pipeline that generates, analyzes, and logs model data to **Arize AI** for monitoring and performance tracking.

---

## Overview

The project simulates a binary fraud classification workflow. It:

1. Generates synthetic prediction data with realistic fraud-related features.
2. Performs analytics on the generated dataset.
3. Logs the results to Arize AI for observability and analysis.

---

## File Structure

| File                    | Description                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `generate_fake_data.py` | Generates synthetic fraud prediction data with relevant features.  |
| `examine_data.py`       | Computes key analytics and prints performance summaries.           |
| `main.py`               | Executes the full pipeline, from data generation to Arize logging. |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Install Dependencies

Ensure Python 3.8 or higher is installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and add:

```
ARIZE_API_KEY=your_arize_api_key
ARIZE_SPACE_ID=your_arize_space_id
```

### 4. Run the Pipeline

```bash
python main.py
```

---

## Output Details

* **Console Output**
  Displays the generated dataset and analytics summary.

* **CSV File**
  A local copy of the dataset is saved as `new_data.csv`.

* **Arize Logging**
  The pipeline logs data to Arize for monitoring model metrics and feature trends.

---

## Analytics Summary

The analytics module (`examine_data.py`) reports:

* Accuracy of model predictions
* Average prediction score
* Actual fraud rate
* Feature distributions and averages

---

## Core Components

### 1. Data Generation (`generate_fake_data.py`)

Produces a 7-day synthetic dataset (default: 300 samples) with fields such as:

* `prediction_id`, `prediction_timestamp`
* `prediction_score`, `actual_label`
* `transaction_amount`, `user_history_days`, `device_type`, `location_risk`

### 2. Data Examination (`examine_data.py`)

Analyzes the dataset to compute metrics and display feature statistics.

### 3. Data Logging (`main.py`)

Defines the Arize schema and logs the dataset using `arize.pandas.logger.Client`.

Example schema:

```python
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="prediction_timestamp",
    prediction_score_column_name="prediction_score",
    actual_label_column_name="actual_label",
    feature_column_names=[
        'transaction_amount', 
        'user_history_days', 
        'device_type', 
        'location_risk'
    ]
)
```