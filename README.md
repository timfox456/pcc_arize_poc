# pcc_arize_poc# Arize Data Upload Project

This project contains a set of Python scripts to generate fake data, analyze its quality, and upload it to the Arize AI platform for model monitoring.

## Project Structure

- `.env`: Configuration file for storing API keys and other secrets.
- `generate_fake_data.py`: Generates a modified outcomes dataset to ensure prediction IDs match.
- `upload_to_arize.py`: Uploads the prediction and outcome data to Arize.
- `examine_data.py`: Analyzes the data for quality issues before uploading.
- `requirements.txt`: A list of Python dependencies required for the project.
- `data/`: Directory containing the raw data files.

## Setup and Installation

Follow these steps to set up your local environment and run the scripts.

### 1. Create a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root of the project and add your Arize credentials:

```
ARIZE_SPACE_ID="YOUR_SPACE_ID"
ARIZE_API_KEY="YOUR_API_KEY"
```

Replace `"YOUR_SPACE_ID"` and `"YOUR_API_KEY"` with your actual Arize Space and API keys.

## How to Run the Scripts

### 1. Generate Modified Data

Before uploading, run the `generate_fake_data.py` script to create a `outcome_details_modified.csv` file. This script ensures that the prediction IDs in the outcomes data align with the prediction data.

```bash
python3 generate_fake_data.py
```

### 2. Examine Data Quality

To analyze the data for potential issues like missing values or sparsity, run the `examine_data.py` script. This will print a report to the console, helping you identify problems before uploading.

```bash
python3 examine_data.py
```

### 3. Upload Data to Arize

After generating the modified data and ensuring its quality, run the `upload_to_arize.py` script to log the data to your Arize account.

```bash
python3 upload_to_arize.py
```

If the upload is successful, you will see a confirmation message in the console. You can then view your data in the Arize web interface.
