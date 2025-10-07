from generate_fake_data import generate_data
from examine_data import generate_analytics
from arize.utils.types import ModelTypes, Environments, Schema
from arize.pandas.logger import Client
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
client = Client(api_key=os.getenv("ARIZE_API_KEY") , space_id=os.getenv("ARIZE_SPACE_ID"))
data_dir = "data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

df = generate_data()

print(df.head())
df.to_csv("data/synthetic_data.csv")

analytics = generate_analytics(df)

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

response = client.log(
            dataframe=df,
            model_id="demo",
            model_version="1.0",
            model_type=ModelTypes.BINARY_CLASSIFICATION,
            environment=Environments.PRODUCTION,
            schema=schema
        )
