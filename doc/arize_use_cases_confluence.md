# Arize ML Observability Use Cases for PointClickCare

## Executive Summary

This document outlines two primary integration approaches for implementing Arize ML observability platform with PointClickCare's Patient Risk to Hospital (PRTH) prediction model. These approaches enable comprehensive model monitoring, drift detection, and performance analytics to ensure reliable ML operations in production healthcare environments.

---

## Model Context

**Model Name:** PRTH (Patient Risk to Hospital)
**Current Versions:** 1.0.6, 1.0.7, 1.0.8
**Purpose:** Predict hospitalization risk for patients in skilled nursing facilities
**Model Type:** Binary classification (binarized prediction based on prediction score threshold)

### Key Features Monitored
- **Vital Signs:** Blood pressure (systolic/diastolic), pulse, respiration, temperature, blood sugar, weight, pain level
- **Clinical Data:** Medications (new drugs, total drugs), lab results (abnormal/panic values), NPO status, tube feeding
- **Patient History:** Day of stay, age, severe comorbidities (1-5), clinical notes trends
- **Statistical Aggregates:** Median, variance, trend, and last values for time-series vitals

---

## Use Case 1: Pre-Computed Metrics Approach (Push Model)

### Overview
In this approach, PointClickCare computes model performance metrics, feature drift statistics, and alert conditions internally, then sends the pre-calculated results to Arize for centralized dashboarding, alerting, and historical analysis.

### Architecture

```
┌─────────────────────────────────────┐
│   PointClickCare ML Pipeline        │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Model Inference Engine      │  │
│  │  (PRTH v1.0.x)              │  │
│  └──────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Internal Metrics Engine     │  │
│  │  • Model Performance (F1,    │  │
│  │    AUC, Precision, Recall)   │  │
│  │  • Feature Drift (KS test,   │  │
│  │    p-values per feature)     │  │
│  │  • Data Quality Checks       │  │
│  │  • Custom Business Metrics   │  │
│  └──────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Alert Threshold Logic       │  │
│  │  • Drift p-value < 0.05      │  │
│  │  • Performance degradation   │  │
│  │  • Data quality issues       │  │
│  └──────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │
               │ Push Metrics via API
               ▼
┌─────────────────────────────────────┐
│         Arize Platform              │
│  ┌──────────────────────────────┐  │
│  │  Metrics Ingestion Layer     │  │
│  └──────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Centralized Dashboards      │  │
│  │  • Model performance trends  │  │
│  │  • Feature drift heatmaps    │  │
│  │  • Multi-facility comparison │  │
│  │  • Version performance       │  │
│  └──────────────────────────────┘  │
│              │                      │
│  ┌──────────────────────────────┐  │
│  │  Alerting & Notifications    │  │
│  │  • Email/Slack alerts        │  │
│  │  • Incident management       │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Data Examples from PointClickCare

#### 1. Model Performance Metrics
Based on `baseline_model_accuracy_aggregate.csv`:

```json
{
  "model_name": "prth",
  "model_version": "1.0.7",
  "timestamp": "2025-09-24T19:05:19Z",
  "metrics": {
    "f1_score": 0.5,
    "auc": 0.9,
    "precision": 0.45,
    "recall": 0.58
  },
  "aggregation_period": "daily",
  "total_predictions": 1247
}
```

#### 2. Feature Drift Metrics
Based on `feature_drift_metrics.csv`:

```json
{
  "model_name": "prth",
  "model_version": "1.0.7",
  "prediction_as_of_datetime_utc": "2025-09-06T15:00:06Z",
  "group_id": "2025_9_6_am",
  "aggregation_level": "run",
  "features": [
    {
      "feature_name": "weight_last",
      "ks_stat": 0.7562057,
      "p_value": 0.0000351,
      "drift_detected": true,
      "severity": "high"
    },
    {
      "feature_name": "total_drugs_taken",
      "ks_stat": 0.7952312,
      "p_value": 0.0000076,
      "drift_detected": true,
      "severity": "critical"
    },
    {
      "feature_name": "pulse_median",
      "ks_stat": 0.8723333,
      "p_value": 0.0000002,
      "drift_detected": true,
      "severity": "critical"
    },
    {
      "feature_name": "no_of_new_drug_taken",
      "ks_stat": 0.2947977,
      "p_value": 0.4157642,
      "drift_detected": false,
      "severity": "none"
    }
  ]
}
```

### Implementation Details

#### Data Sent to Arize
1. **Aggregated Performance Metrics** (Daily/Hourly)
   - F1 Score, AUC-ROC, Precision, Recall
   - Confusion matrix values
   - Prediction volume and distribution

2. **Feature Drift Statistics**
   - Kolmogorov-Smirnov (KS) test statistics per feature
   - P-values indicating statistical significance
   - Feature importance scores
   - Distribution comparisons (baseline vs production)

3. **Alert Events**
   - Drift alerts (features with p-value < 0.05)
   - Performance degradation alerts
   - Data quality issues
   - Missing data patterns

4. **Metadata**
   - Model version
   - Facility ID / Organization UUID
   - Timestamp ranges
   - Run identifiers

#### Advantages
✅ **Reduced Arize Costs** - Only aggregated metrics sent, not raw predictions
✅ **Data Privacy** - PHI remains within PCC infrastructure
✅ **Custom Business Logic** - PCC controls alerting thresholds and metrics
✅ **Lower Network Overhead** - Minimal data transfer
✅ **Faster Dashboards** - Pre-computed metrics render instantly
✅ **Flexible Aggregation** - Choose granularity (facility, time period, model version)

#### Challenges
⚠️ **Limited Drill-Down** - Cannot investigate individual predictions in Arize
⚠️ **Duplicate Computation** - PCC must maintain internal metrics pipeline
⚠️ **Manual Alert Tuning** - Threshold management in two systems
⚠️ **Less Arize Features** - Cannot leverage Arize's advanced analytics (cohort analysis, SHAP, etc.)

### Arize API Integration Example

```python
from arize.api import Client
from arize.utils.types import Environments, ModelTypes, Metrics

# Initialize Arize client
arize_client = Client(api_key='YOUR_API_KEY', space_key='YOUR_SPACE_KEY')

# Send pre-computed metrics
response = arize_client.log_bulk_metrics(
    model_id='prth',
    model_version='1.0.7',
    environment=Environments.PRODUCTION,
    metrics=[
        {
            'metric_name': 'f1_score',
            'metric_value': 0.5,
            'timestamp': 1695571519000
        },
        {
            'metric_name': 'auc',
            'metric_value': 0.9,
            'timestamp': 1695571519000
        }
    ]
)

# Send drift metrics
drift_response = arize_client.log_bulk_drift_metrics(
    model_id='prth',
    model_version='1.0.7',
    environment=Environments.PRODUCTION,
    drift_metrics=[
        {
            'feature_name': 'weight_last',
            'ks_stat': 0.7562057,
            'p_value': 0.0000351,
            'timestamp': 1695571519000
        }
    ]
)
```

### Recommended Dashboards in Arize

1. **Model Performance Trends**
   - F1/AUC over time by model version
   - Facility-level performance comparison
   - Prediction volume trends

2. **Feature Drift Heatmap**
   - All features with KS statistics
   - Color-coded by p-value thresholds
   - Time-series drift visualization

3. **Alert Summary**
   - Active drift alerts by severity
   - Performance degradation incidents
   - Alert resolution tracking

---

## Use Case 2: Full Feature Store & Predictions Approach (Raw Data Model)

### Overview
PointClickCare sends complete prediction records with all input features, prediction scores, and actual outcomes to Arize. Arize performs all metric calculations, drift detection, and analysis using its native ML observability capabilities.

### Architecture

```
┌─────────────────────────────────────┐
│   PointClickCare ML Pipeline        │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Model Inference Engine      │  │
│  │  (PRTH v1.0.x)              │  │
│  └──────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Prediction Logger           │  │
│  │  • prediction_id             │  │
│  │  • All input features (50+)  │  │
│  │  • Prediction score          │  │
│  │  • Binarized prediction      │  │
│  │  • Timestamp, metadata       │  │
│  └──────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Actuals Logger (Delayed)    │  │
│  │  • Match prediction_id       │  │
│  │  • Actual hospitalization    │  │
│  │  • Outcome timestamp         │  │
│  └──────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │
               │ Stream via SDK/API
               ▼
┌─────────────────────────────────────┐
│         Arize Platform              │
│  ┌──────────────────────────────┐  │
│  │  Inference Data Ingestion    │  │
│  │  • Real-time streaming       │  │
│  │  • Batch upload support      │  │
│  └──────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Automated Analytics Engine  │  │
│  │  • Auto-compute performance  │  │
│  │  • Auto-detect drift (PSI,   │  │
│  │    KL, JS divergence)        │  │
│  │  • Feature importance        │  │
│  │  • SHAP explainability       │  │
│  │  • Cohort analysis           │  │
│  └──────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Rich Interactive Dashboards │  │
│  │  • Drill down to predictions │  │
│  │  • Feature distributions     │  │
│  │  • Segment performance       │  │
│  │  • Root cause analysis       │  │
│  └──────────────────────────────┘  │
│              │                      │
│  ┌──────────────────────────────┐  │
│  │  Intelligent Alerting        │  │
│  │  • ML-powered thresholds     │  │
│  │  • Anomaly detection         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Data Examples from PointClickCare

#### 1. Prediction Records with Full Features
Based on `prediction_details.csv`:

```json
{
  "prediction_id": "628430151",
  "prediction_timestamp": "2025-09-22T16:00:00Z",
  "model_id": "prth",
  "model_version": "1.0.8",
  "model_type": "binary_classification",

  "features": {
    "no_of_new_drug_taken": 0,
    "total_drugs_taken": 0,
    "num_abnormal_labs_3day": 0,
    "day_of_stay": 64,
    "npo": 0,
    "median_pct_eaten": null,
    "tube_feeding": 0,
    "num_panic_labs_3day": 0,
    "age": 30,
    "weight_last": 165.2,
    "pulse_median": 72.0,
    "pulse_variance": 8.3,
    "pulse_trend": 0.05,
    "bp_systolic_median": 120.0,
    "bp_diastolic_median": 80.0,
    "blood_sugar_median": 95.0,
    "pain_level_median": 2.0,
    "respiration_median": 16.0,
    "temperature_trend": 0.0,
    "severe_comorbidity_1": 0,
    "severe_comorbidity_2": 0,
    "severe_comorbidity_3": 0,
    "severe_comorbidity_4": 0,
    "severe_comorbidity_5": 0
  },

  "prediction_score": 0.156,
  "binarized_prediction": 0,

  "tags": {
    "org_uuid": "470ae66c-bd31-45c2-b040-e9e273efbd2d",
    "facility_id": "37",
    "client_id": "8396081",
    "run_id": "20000344",
    "status": "MISSING_DATA",
    "date_to_year": "2025",
    "date_to_month": "9",
    "date_to_day": "22"
  }
}
```

#### 2. Actual Outcomes (Delayed)
Based on `outcome_details.csv`:

```json
{
  "prediction_id": "627811394",
  "actual_timestamp": "2025-09-21T00:00:00Z",
  "actual_label": 0,
  "outcome_type": "hospitalization_within_7_days"
}
```

### Implementation Details

#### Data Sent to Arize

1. **Prediction Records** (Real-time or Batch)
   - Unique prediction ID
   - All 50+ input features with actual values
   - Prediction score (0-100)
   - Binarized prediction (0 or 1)
   - Timestamp
   - Model version

2. **Actual Outcomes** (Delayed Join)
   - Prediction ID reference
   - Ground truth label
   - Outcome observation timestamp

3. **Metadata & Tags**
   - Organization UUID
   - Facility ID
   - Client ID
   - Run ID
   - Data quality flags (e.g., MISSING_DATA status)
   - Date partitions

#### Advantages
✅ **Full Arize Capabilities** - Leverage all platform features (SHAP, cohort analysis, automated drift)
✅ **Prediction-Level Investigation** - Drill down to individual problematic predictions
✅ **Automated Drift Detection** - Arize computes PSI, KL divergence automatically
✅ **No Custom Metrics Pipeline** - Arize handles all computations
✅ **Advanced Root Cause Analysis** - Slice and dice by any feature or tag
✅ **Model Comparison** - Easily compare v1.0.6 vs v1.0.7 vs v1.0.8
✅ **Explainability** - Built-in SHAP value calculation
✅ **Segment Performance** - Analyze performance by facility, age group, comorbidity

#### Challenges
⚠️ **Higher Costs** - Arize pricing based on prediction volume
⚠️ **Data Privacy Considerations** - PHI data sent to third-party platform (requires BAA)
⚠️ **Network Overhead** - Streaming thousands of predictions daily
⚠️ **Feature Engineering Sync** - Feature definitions must match between PCC and Arize
⚠️ **Delayed Actuals** - Hospitalization outcomes may lag 1-7 days

### Arize SDK Integration Example

```python
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments, Schema
import pandas as pd

# Initialize Arize client
arize_client = Client(api_key='YOUR_API_KEY', space_key='YOUR_SPACE_KEY')

# Load prediction data
predictions_df = pd.read_csv('prediction_details.csv')

# Define schema mapping
schema = Schema(
    prediction_id_column_name='prediction_id',
    timestamp_column_name='prediction_as_of_datetime_utc',
    prediction_score_column_name='prediction_score',
    prediction_label_column_name='binarized_prediction',

    feature_column_names=[
        'no_of_new_drug_taken', 'total_drugs_taken', 'num_abnormal_labs_3day',
        'day_of_stay', 'npo', 'median_pct_eaten', 'tube_feeding',
        'num_panic_labs_3day', 'age', 'weight_last', 'pulse_median',
        'pulse_variance', 'pulse_trend', 'bp_systolic_median',
        'bp_diastolic_median', 'blood_sugar_median', 'pain_level_median',
        'respiration_median', 'temperature_trend', 'severe_comorbidity_1',
        'severe_comorbidity_2', 'severe_comorbidity_3', 'severe_comorbidity_4',
        'severe_comorbidity_5'
    ],

    tag_column_names=[
        'org_uuid', 'facility_id', 'client_id', 'run_id', 'status',
        'date_to_year', 'date_to_month', 'date_to_day'
    ]
)

# Log predictions to Arize
response = arize_client.log(
    dataframe=predictions_df,
    model_id='prth',
    model_version='1.0.8',
    model_type=ModelTypes.BINARY_CLASSIFICATION,
    environment=Environments.PRODUCTION,
    schema=schema
)

# Later: Log actuals when available
actuals_df = pd.read_csv('outcome_details.csv')

actuals_schema = Schema(
    prediction_id_column_name='prediction_id',
    actual_label_column_name='binarized_outcome',
    timestamp_column_name='actual_timestamp'
)

actuals_response = arize_client.log(
    dataframe=actuals_df,
    model_id='prth',
    model_version='1.0.8',
    model_type=ModelTypes.BINARY_CLASSIFICATION,
    environment=Environments.PRODUCTION,
    schema=actuals_schema
)
```

### Recommended Dashboards in Arize

1. **Model Performance Monitor**
   - Accuracy, Precision, Recall, F1, AUC-ROC over time
   - Confusion matrix trends
   - Performance by facility and model version

2. **Feature Drift Analysis**
   - PSI (Population Stability Index) heatmap for all features
   - Distribution comparisons (baseline vs production)
   - Feature importance vs drift correlation

3. **Cohort Performance**
   - Performance by age groups (e.g., <65, 65-75, 75-85, 85+)
   - Performance by day of stay ranges
   - Performance by comorbidity burden
   - Facility-level performance benchmarking

4. **Prediction Troubleshooting**
   - Filter by specific prediction IDs
   - Investigate false positives/negatives
   - SHAP explanations for individual predictions

5. **Data Quality Dashboard**
   - Missing feature values over time
   - Feature value ranges and outliers
   - Records with MISSING_DATA status

---

## Comparison Matrix

| Criterion | Use Case 1: Pre-Computed Metrics | Use Case 2: Full Feature Store |
|-----------|----------------------------------|--------------------------------|
| **Data Volume** | Low (aggregated metrics) | High (all predictions + features) |
| **Arize Cost** | Lower | Higher |
| **Setup Complexity** | Medium (custom metrics pipeline) | Low (SDK integration) |
| **Data Privacy** | Better (PHI stays internal) | Requires BAA with Arize |
| **Investigation Depth** | Limited (aggregate only) | Deep (prediction-level) |
| **Arize Features** | Basic dashboards, alerts | Full platform capabilities |
| **Latency** | Low (pre-computed) | Near real-time |
| **Root Cause Analysis** | Manual | Automated |
| **Model Explainability** | Not available | SHAP, feature importance |
| **Maintenance** | Higher (two systems) | Lower (Arize handles) |

---

## Recommendations

### Short-Term (Pilot Phase)
**Recommendation:** Start with **Use Case 2 (Full Feature Store)** for a single facility or limited model version.

**Rationale:**
- Faster time to value (minimal custom code)
- Leverage Arize's full analytical capabilities
- Learn which metrics and dashboards are most valuable
- Establish data pipeline patterns

**Action Items:**
1. Sign BAA (Business Associate Agreement) with Arize for HIPAA compliance
2. Integrate Arize Python SDK in inference pipeline
3. Set up baseline dataset (e.g., September 2025 predictions)
4. Configure Arize monitors for top 10 most important features
5. Create executive dashboard in Arize

### Medium-Term (Scale Up)
**Recommendation:** Evaluate **Use Case 1 (Pre-Computed Metrics)** if costs become prohibitive or privacy concerns arise.

**Rationale:**
- Better cost control at scale (thousands of predictions/day across facilities)
- Enhanced data privacy (PHI remains on-premise)
- Customizable business metrics specific to PCC workflows

**Action Items:**
1. Develop internal metrics computation pipeline
2. Define aggregation strategies (facility-level, time-based)
3. Create Arize API integration for metric ingestion
4. Migrate alerting rules from Arize to PCC systems
5. Maintain Arize for visualization and historical tracking

### Long-Term (Hybrid Approach)
**Recommendation:** Implement **hybrid model** combining both approaches.

**Strategy:**
- **Use Case 2** for critical model versions and high-value facilities (deep monitoring)
- **Use Case 1** for mature models and lower-risk facilities (cost optimization)
- Centralized Arize dashboards for all models

**Benefits:**
- Optimize cost vs capability tradeoff
- Flexibility to deep-dive when needed
- Scalable to hundreds of models and facilities

---

## Implementation Checklist

### For Use Case 1 (Pre-Computed Metrics)
- [ ] Define core metrics (F1, AUC, drift KS stats)
- [ ] Build internal metrics aggregation pipeline
- [ ] Integrate Arize Metrics API
- [ ] Set up Arize dashboards for aggregated data
- [ ] Configure alert rules in both PCC and Arize
- [ ] Document metric calculation methodology
- [ ] Schedule periodic metrics exports (hourly/daily)

### For Use Case 2 (Full Feature Store)
- [ ] Sign Arize BAA for HIPAA compliance
- [ ] Install Arize Python SDK in inference environment
- [ ] Map feature schema to Arize format
- [ ] Configure prediction_id as primary key
- [ ] Set up delayed actuals logging (7-day window)
- [ ] Define baseline dataset for drift comparison
- [ ] Configure Arize monitors (drift, performance, data quality)
- [ ] Create role-based dashboards in Arize
- [ ] Test end-to-end pipeline with sample data
- [ ] Set up alerting channels (email, Slack, PagerDuty)
- [ ] Document troubleshooting workflows

---

## Key Metrics to Monitor (Both Use Cases)

### Model Performance
- **AUC-ROC:** Target ≥ 0.85
- **F1 Score:** Target ≥ 0.45
- **Precision:** Minimize false positives (alert fatigue)
- **Recall:** Maximize true positives (catch risky patients)

### Feature Drift
- **Critical Features:** weight_last, total_drugs_taken, pulse_median, respiration_median
- **Drift Threshold:** p-value < 0.05 or PSI > 0.2
- **Action:** Retrain model if >20% of features show drift

### Data Quality
- **Missing Data Rate:** Target < 5% of predictions
- **Feature Value Range:** Detect outliers (e.g., age > 120)
- **Prediction Volume:** Alert if daily predictions drop >30%

### Business Metrics
- **Alerts per Facility:** Track alert volume by facility
- **Model Adoption:** Percentage of eligible patients scored
- **Actuals Linkage Rate:** Percentage of predictions with matched outcomes

---

## Appendix: Sample Data Summary

### Datasets Available
1. **baseline_model_accuracy_aggregate.csv** (6 rows)
   - Model versions: 1.0.6, 1.0.7
   - Metrics: F1 (0.24-0.5), AUC (0.75-0.9)

2. **feature_drift_metrics.csv** (8,300+ rows)
   - 51 features monitored
   - KS statistics and p-values per feature per run
   - Significant drift detected in vitals and medications

3. **prediction_details.csv** (14,400+ rows)
   - Full prediction records with features
   - Model version: 1.0.8
   - Status: Many marked as MISSING_DATA

4. **outcome_details.csv** (4,600+ rows)
   - Ground truth labels
   - Binarized predictions and outcomes
   - Actuals for model performance evaluation

5. **synthetic_data.csv** (1,000 rows)
   - Example fraud detection model data
   - Can be used for Arize POC testing

### Model Versioning Observed
- **v1.0.6:** Earlier version, AUC=0.75, F1=0.24
- **v1.0.7:** Improved version, AUC=0.82-0.9, F1=0.24-0.5
- **v1.0.8:** Current production version (most recent predictions)

**Note:** Performance improvement from v1.0.6 to v1.0.7 suggests successful retraining. Version 1.0.8 lacks aggregated metrics in sample data but is actively generating predictions.

---

## Conclusion

Both Arize integration approaches offer distinct advantages for PointClickCare's ML observability needs. **Use Case 2 (Full Feature Store)** provides the richest analytical capabilities and fastest implementation, making it ideal for initial pilots and high-criticality models. **Use Case 1 (Pre-Computed Metrics)** offers better cost efficiency and data privacy at scale, suitable for mature production deployments.

A phased approach—starting with Use Case 2 for deep insights, then optimizing with Use Case 1 where appropriate—will maximize the value of Arize while managing costs and compliance requirements.

**Next Steps:**
1. Schedule alignment meeting with Arize solutions engineer
2. Prioritize model versions and facilities for pilot
3. Initiate BAA execution for HIPAA compliance
4. Define success criteria and KPIs for observability program
5. Allocate engineering resources for SDK integration

---

**Document Version:** 1.0
**Last Updated:** October 15, 2025
**Author:** ML Platform Team
**Reviewers:** Data Science, MLOps, Compliance
