# Customer Churn Prediction Project

This project provides an end-to-end solution for predicting customer churn, starting from raw data processing to serving a model via an API, with support for automated retraining and monitoring.

## Model Selection and Evaluation Strategy

A strategic selection of models was chosen to provide a comprehensive comparison, ranging from a simple baseline to a state-of-the-art algorithm.

### 1. Model Selection Rationale

- **Logistic Regression:** Serves as a fast, interpretable, and robust baseline. Its performance is the minimum threshold that more complex models must exceed.
- **Random Forest:** A powerful ensemble model capable of capturing non-linear relationships and interactions between features without significant risk of overfitting.
- **XGBoost (Extreme Gradient Boosting):** A state-of-the-art algorithm known for its high performance on tabular data. It includes built-in parameters like `scale_pos_weight` to effectively handle the class imbalance inherent in churn prediction problems.

### 2. How the Best Model Was Selected

The key challenge in this project is the class imbalance, where the number of churning users is far smaller than non-churning users.

#### Why Not Accuracy?
Accuracy is a misleading metric in this context. A model that always predicts "no churn" would achieve high accuracy but would be useless for the business goal of identifying at-risk customers.

#### The Deciding Metric: F1-Score
The **F1-Score** for the positive class (churn = 1) was chosen as the primary evaluation metric. It provides a harmonic mean of two critical measures:

- **Precision:** Minimizes false positives. *(Of all users we predict will churn, how many actually do?)*
- **Recall:** Minimizes false negatives. *(Of all users who actually churned, how many did we successfully identify?)*

The F1-Score ensures the selected model strikes a good balance between identifying churners and avoiding false alarms.

#### The Selection Process
The final model was chosen through a clear, systematic process:

1.  All three models were trained on the same training dataset.
2.  Each model was evaluated on the same unseen test set.
3.  The F1-Score for the churn class was calculated for each model.
4.  The model with the highest F1-Score was selected as the best performer and deployed to the API.





## 🚀 Key Features

- **End-to-End Pipeline:** A fully automated pipeline for feature engineering, model training, experiment tracking, and drift checking.
- **API-Driven Predictions:** A FastAPI endpoint (`/predict`) that accepts raw user event data, performs feature engineering, and returns real-time predictions.
- **Scheduled Retraining:** An internal scheduler (`scheduler.py`) periodically retrains the model to keep it current with evolving user behavior.
- **Dockerized Environment:** The entire application is containerized using Docker, ensuring a consistent and portable runtime environment.
- **MLflow Experiment Tracking:** Logs and compares all models (Logistic Regression, Random Forest, XGBoost) to easily select the best-performing one based on metrics.
- **Drift Monitoring:** A simple, integrated system to detect Data Drift and Concept Drift.
- **Automation & Best Practices:** Uses a `Makefile` to simplify common commands and `pre-commit` hooks to ensure code quality.



## 📁 Project Structure

```
.
├── api.py                  # FastAPI application
├── config.py               # Central configuration file
├── Dockerfile              # Docker container definition
├── main.py                 # Main pipeline script
├── Makefile                # Automation commands
├── requirements.txt        # Python dependencies
├── scheduler.py            # Retraining scheduler
└── src/
    ├── check_concept_drift.py # Concept drift logic
    ├── data_drift.py       # Data drift logic
    ├── features.py         # Feature engineering
    └── train_model.py      # Model training
```

## ⚙️ Running with Docker

Docker is required to run this project. All common operations have been simplified using a `Makefile`.

### 1. Build and Run the Container

This is the primary command to get everything running. It builds the Docker image, starts the container, runs the initial training pipeline, and then launches the API and the scheduler.

```bash
sudo make up
```

Once this command completes, the following services will be available:

- **API Endpoint:** `http://localhost:8000`
- **Interactive Docs (Swagger UI):** `http://localhost:8000/docs`

### 2. Other Useful Commands

- `sudo make down`: Stops and removes the running container.
- `sudo make logs`: Tails the container logs to monitor output.
- `sudo make retrain`: Manually triggers the retraining pipeline.

## 💡 API Usage for Predictions

The API is designed to be practical and flexible. It accepts a list of raw user event logs (`events`) instead of pre-computed features. The server handles the feature engineering and prediction internally.

### Example with `curl`

You can send a `POST` request to the `/predict` endpoint with the event data in JSON format.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "events": [
    {"userId": "100", "ts": 1609459200000, "registration": 1606780800000, "page": "NextSong", "sessionId": 10, "gender": "F", "location": "New York, NY", "level": "paid"},
    {"userId": "100", "ts": 1609459440000, "registration": 1606780800000, "page": "NextSong", "sessionId": 10, "gender": "F", "location": "New York, NY", "level": "paid"},
    {"userId": "200", "ts": 1609459800000, "registration": 1609372800000, "page": "Home", "sessionId": 20, "gender": "M", "location": "Los Angeles, CA", "level": "free"}
  ]
}'
```

### Request Breakdown

- `events`: A list of JSON objects, where each object represents a single user event.
- You can send events for multiple users in the same request (e.g., `userId: "100"` and `userId: "200"` in the example).
- The API will group events by `userId`, calculate features for each user, and return an independent prediction.

### Expected Response

The response will be a list of prediction objects, one for each unique user in the request.

```json
{"predictions":[
    {"userId":"100","churn_probability":0.32750241268427444,"prediction":"No Churn"},{"userId":"200","churn_probability":0.6641305819708586,"prediction":"No Churn"}]}
```

- `prediction: 0`: The user is not likely to churn.
- `prediction: 1`: The user is likely to churn.

##  Future Improvements

- **Collect More Data:** Gather a larger and more diverse dataset to improve model generalization and capture a wider range of user behaviors.
- **In-Depth Feature Engineering:** Dive deeper into feature extraction, creating more sophisticated and predictive features from the raw event data.
- **Implement Cross-Validation:** Introduce a robust cross-validation strategy to ensure the model's performance is stable and reliable, especially as more data becomes available.
- **Explore More Complex Models:** Experiment with advanced modeling techniques (e.g., LSTMs, Transformers, or other deep learning architectures) to capture complex temporal patterns in user activity.
