import schedule
import time
import subprocess
import logging
from config import config

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_retraining_job():
    """
    Triggers the main.py script to run the full retraining pipeline.
    """
    logging.info("Scheduler: Kicking off retraining job...")
    try:
        # We use subprocess to run the script in its own process.
        # This is cleaner than importing and calling a main function.
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            check=True,  # Raises an exception if the script fails
        )
        logging.info("Scheduler: Retraining job finished successfully.")
        logging.info(result.stdout)  # Log the output from the script
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Scheduler: Retraining job failed with exit code {e.returncode}."
        )
        logging.error(e.stderr)
    except Exception as e:
        logging.error(f"Scheduler: An unexpected error occurred: {e}")


# --- Scheduling Setup ---

# Get the retraining interval from the config file, default to 1 day if not found.
retrain_interval = config.get("retrain_interval_days", 1)

logging.info(f"Retraining job scheduled to run every {retrain_interval} day(s).")

# Schedule the job. The time is set to 03:00 to match the old cron job.
schedule.every(retrain_interval).days.at("03:00").do(run_retraining_job)

# --- Main Loop ---
# This loop keeps the scheduler running and checks for pending jobs.
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every 60 seconds
