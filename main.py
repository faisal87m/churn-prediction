import pandas as pd
import numpy as np
import os
import mlflow
import logging
import subprocess

# Import configuration and custom modules
from config import config
from src.features import create_features
from src.train_model import train_model
from src.data_drift import check_and_log_data_drift
from src.check_concept_drift import check_and_log_concept_drift

# Set up logging for the pipeline
# This sets the logging level to INFO and specifies a format for log messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# إعداد logging
# This is an Arabic comment that translates to "Preparing logging"

# Utility function to get the latest data file from a directory
# Returns the path to the most recently modified .json file
# Used to fetch the most up-to-date dataset for training or monitoring


def get_latest_data_files(data_path, pattern="*.json"):
    """
    Returns the paths to the two most recent data files in the directory.
    Useful for current and reference datasets.
    """
    import glob

    # Find all files matching the pattern
    files = glob.glob(os.path.join(data_path, pattern))
    # Sort files by creation time (most recent first)
    files = sorted(files, key=os.path.getctime, reverse=True)
    if len(files) == 0:
        raise FileNotFoundError("No data files found in the specified directory.")
    elif len(files) == 1:
        # If only one file, use it for both current and reference
        return files[0], files[0]
    else:
        # Return the two most recent files
        return files[0], files[1]


# Main pipeline function to orchestrate the end-to-end process:
# 1. Optionally update the data using an external script
# 2. Load and process both current and reference data
# 3. Check for data drift
# 4. Train a new model
# 5. Check for concept drift
# 6. Log all steps and results to MLflow
def run_pipeline():
    """A simplified pipeline that trains the model on the latest data."""
    # Set MLflow tracking URI
    # This specifies where MLflow will store its tracking data
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Check if data update is enabled in the configuration
    if config["data_update"]["enabled"]:
        # Get the path to the data update script
        script_path = config["data_update"]["script_name"]
        logging.info(f"--- Step 0: Running data update script: {script_path} ---")
        try:
            # Run the data update script using subprocess
            subprocess.run(["python", script_path], check=True)
            logging.info("Data update script completed successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Log an error if the script fails or is not found
            logging.error(f"Failed to run data update script '{script_path}': {e}")
            # Decide if the pipeline should stop if the update fails. For now,
            # # we'll continue.

    # Set the MLflow experiment name
    # This groups related runs together in MLflow
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Start a new MLflow run for the pipeline
    with mlflow.start_run(run_name="Unified_Pipeline_Run") as parent_run:
        logging.info(
            f"Starting unified pipeline. MLflow Parent Run ID: {parent_run.info.run_id}"
        )
        mlflow.set_tag("pipeline_status", "started")

        try:
            # --- 1. Load and Process Data ---
            # Load and process both current and reference data
            logging.info(
                "--- Step 1: Loading and Processing Current and Reference Data ---"
            )
            current_data_file, reference_data_file = get_latest_data_files(
                config["paths"]["data_directory"],
                config["paths"]["training_data_pattern"],
            )

            # Load and process the current data
            df_current_raw = pd.read_json(current_data_file, lines=True)
            df_features_current = create_features(df_current_raw)
            logging.info(
                f"Current features  created with shape: {df_features_current.shape}"
            )

            # Load and process the reference data
            df_reference_raw = pd.read_json(reference_data_file, lines=True)
            df_features_reference = create_features(df_reference_raw)
            logging.info(
                f"Reference features  created with shape: {df_features_reference.shape}"
            )

            # --- 2. Monitoring: Data Drift Check ---
            # Check for data drift between the current and reference data
            logging.info("--- Step 2: Running Data Drift Check ---")
            numerical_features_for_drift = (
                df_features_reference.select_dtypes(include=np.number)
                .columns.drop(["is_churned"], errors="ignore")
                .tolist()
            )
            with mlflow.start_run(run_name="Data_Drift_Check", nested=True):
                logging.info("Comparing current data against reference data for drift.")
                check_and_log_data_drift(
                    df_features_reference,  # Base
                    df_features_current,  # New
                    numerical_features=numerical_features_for_drift,
                )

            # --- 3. Train New Model on Current Data ---
            # Train a new model on the current data
            logging.info("--- Step 3: Training New Model on Current Data ---")
            new_model, new_f1_score = train_model(df_features_current)
            if not new_model:
                raise Exception("Model training failed on current data.")

            # Log the new model to MLflow
            mlflow.sklearn.log_model(new_model, config["mlflow"]["model_artifact_path"])
            logging.info("New best model trained and logged to parent run.")

            # --- 4. Monitoring: Concept Drift Check ---
            # Check for concept drift by comparing the new model's performance on new data
            logging.info("--- Step 4: Running Concept Drift Check ---")
            with mlflow.start_run(run_name="Concept_Drift_Check", nested=True):
                logging.info("Running Concept Drift check.")
                # We check the new model's performance on its own test set (from within train_model) against a threshold.
                # The `best_f1_score` from `train_model` is the performance on the new data's test set.
                check_and_log_concept_drift(
                    new_model, df_features_current, new_f1_score
                )

            logging.info("\nUnified pipeline completed successfully!")
            mlflow.set_tag("pipeline_status", "success")

        except FileNotFoundError as e:
            # Log an error if a required file is not found
            logging.error(f"Error: A required file was not found. {e}")
            mlflow.set_tag("pipeline_status", "failed_file_not_found")
        except Exception:
            # Log an error if any other exception occurs
            logging.exception("An unexpected error occurred in the pipeline:")
            mlflow.set_tag("pipeline_status", "failed_exception")


# Entry point for running the pipeline as a script
if __name__ == "__main__":
    run_pipeline()
