import os
import pandas as pd
import uuid
from datetime import datetime
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
import json

# --- 1. Configuration ---
SPACE_ID = "U3BhY2U6Mjk2MTM6WndkNw=="
API_KEY = os.environ.get("ARIZE_API_KEY")
API_KEY = "ak-a05d6e5f-c733-41bc-97ed-f1cc9a1c8934-8qW5ejv_XNni7t17ZDNrvECx6P5Kzt8M%"
if not SPACE_ID or not API_KEY:
    raise ValueError("ARIZE_SPACE_ID and ARIZE_API_KEY must be set as environment variables.")
MODEL_ID = "prth"
MODEL_VERSION = "1.0.8"

from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.utils.constants import GENERATIVE
import pandas as pd

client = ArizeDatasetsClient(api_key="YOUR_API_KEY")
# Create a dataset from a DataFrame add your own data here
df = pd.DataFrame(data)
dataset_id = client.create_dataset(space_id="U3BhY2U6Mjk2MTM6WndkNw==", dataset_name="my_dataset", dataset_type=GENERATIVE, data=df)
