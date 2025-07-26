import pandas as pd
from scipy.stats import ks_2samp
import mlflow
from typing import List, Dict

# تعديل مسار النظام للسماح بالاستيراد من المجلد الرئيسي
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config  # استيراد الإعدادات المركزية


def check_and_log_data_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, numerical_features: List[str]
) -> Dict:
    """
    Checks for data drift between a reference and current dataset using the K-S test.

    Args:
        reference_df: The baseline dataset.
        current_df: The new dataset to compare against the baseline.
        numerical_features: A list of numerical feature names to check for drift.

    Returns:
        A dictionary containing the drift report.
    """
    print("--- Checking for Data Drift ---")
    drift_report = {}
    total_drifted_features = 0

    for feature in numerical_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            print(
                f"Warning: Feature '{feature}' not found in one of the dataframes. Skipping."
            )
            continue

        # Perform the two-sample Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(reference_df[feature], current_df[feature])
        is_drifted = p_value < config["monitoring"]["data_drift"]["alpha"]

        drift_report[feature] = {
            "p_value": float(p_value),
            "is_drifted": bool(is_drifted),
        }

        if is_drifted:
            total_drifted_features += 1
            print(f"Data drift detected in feature: {feature} (p-value: {p_value:.4f})")

    print(
        f"Data drift check complete. {total_drifted_features} out of {len(numerical_features)} features drifted."
    )

    # Log metrics and report to MLflow
    if mlflow.active_run():
        mlflow.log_metric("data_drift_total_drifted_features", total_drifted_features)
        mlflow.log_metric(
            "data_drift_alpha", config["monitoring"]["data_drift"]["alpha"]
        )
        mlflow.set_tag("data_drift_status", "completed")

        # Log the full report as a JSON artifact without creating a local file
        mlflow.log_dict(drift_report, "data_drift_report.json")

    return drift_report
