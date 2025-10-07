import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_data():
    load_dotenv()
    samples = 300

    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    timestamps = [start_time + timedelta(seconds=np.random.randint(0, 604800)) for _ in range(samples)]

    data = {
            'prediction_id': [f'pred_{i}' for i in range(samples)],
            'prediction_timestamp': timestamps,
            'prediction_score': np.random.beta(2, 5, samples),  # Fraud probability scores
            'actual_label': np.random.choice([0, 1], samples, p=[0.9, 0.1]),  # Mostly non-fraud
            'transaction_amount': np.random.lognormal(3, 1.5, samples),
            'user_history_days': np.random.randint(1, 1000, samples),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], samples),
            'location_risk': np.random.beta(1, 3, samples)
        }


    df = pd.DataFrame(data)

    df['prediction_score'] = (
            df['prediction_score'] * 0.3 +
            df['location_risk'] * 0.4 +
            (df['transaction_amount'] > 1000).astype(int) * 0.2 +
            (df['user_history_days'] < 30).astype(int) * 0.1
        )

    df['prediction_score'] = np.clip(df['prediction_score'], 0, 1)
    return df




