import pandas as pd
import mlflow
from sklearn.metrics import f1_score
from typing import Any, Dict

# تعديل مسار النظام للسماح بالاستيراد من المجلد الرئيسي
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config  # استيراد الإعدادات المركزية


def check_and_log_concept_drift(
    model: Any, current_df: pd.DataFrame, baseline_f1_score: float
) -> Dict:
    """
    Checks for concept drift by comparing model performance on new data against a baseline.

    Args:
        model: The trained model pipeline (preprocessor + model).
        current_df: The new data for performance evaluation.
        baseline_f1_score: The F1 score of the model on the original test set.

    Returns:
        A dictionary containing the concept drift check results.
    """
    print("--- Checking for Concept Drift ---")

    # Prepare current data
    X_current = current_df.drop(
        [config["data"]["user_id_column"], config["data"]["target_column"]], axis=1
    )
    y_current = current_df[config["data"]["target_column"]]

    # Predict on current data
    y_pred = model.predict(X_current)

    # Calculate performance metrics
    current_f1_score = f1_score(y_current, y_pred, pos_label=1)

    # Check for drift
    drift_detected = current_f1_score < (
        baseline_f1_score
        * config["monitoring"]["concept_drift"]["f1_score_threshold_ratio"]
    )

    print(f"Baseline F1 Score: {baseline_f1_score:.4f}")
    print(f"Current F1 Score on new data: {current_f1_score:.4f}")

    if drift_detected:
        print(
            f"Concept Drift DETECTED. Current F1 score is {current_f1_score:.4f}, which is below the threshold."
        )
    else:
        print("No significant concept drift detected.")

    # Log to MLflow
    if mlflow.active_run():
        mlflow.log_metric("concept_drift_baseline_f1_score", baseline_f1_score)
        mlflow.log_metric("concept_drift_current_f1_score", current_f1_score)
        mlflow.log_metric("concept_drift_detected", int(drift_detected))
        mlflow.set_tag("concept_drift_status", "completed")

    print("Concept drift check complete.")

    return {
        "baseline_f1_score": baseline_f1_score,
        "current_f1_score": current_f1_score,
        "drift_detected": bool(drift_detected),
    }
