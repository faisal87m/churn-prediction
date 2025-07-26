# Makefile for Churn Prediction Project

# Variables
IMAGE_NAME = churn-prediction-app
TAG = latest
PYTHON = python3

# Phony targets
.PHONY: all install format lint test build run api install-hooks clean

all: install format lint test

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Format code using pre-commit hooks (runs black, ruff, etc.)
format:
	@echo "Formatting code with pre-commit..."
	pre-commit run --all-files

# Lint code using ruff
lint:
	@echo "Linting code with ruff..."
	ruff .

# Retrain the model (full pipeline)
retrain:
	@echo "Running full retraining pipeline..."
	python main.py

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(TAG) .

# Build, run, and attach to logs
up: build
	@echo "Stopping and removing old container if it exists..."
	@-docker stop $(IMAGE_NAME) > /dev/null 2>&1 || true
	@-docker rm $(IMAGE_NAME) > /dev/null 2>&1 || true
	@echo "Starting new container..."
	@docker run -d -p 8000:8000 --name $(IMAGE_NAME) $(IMAGE_NAME):$(TAG)
	@echo "Attaching to logs... Press Ctrl+C to exit."
	@docker logs -f $(IMAGE_NAME)

# Stop and remove the container
down: stop
	@echo "Removing container..."
	@-docker rm $(IMAGE_NAME) > /dev/null 2>&1 || true

# Run the Docker container in detached mode
run:
	@echo "Running Docker container in detached mode..."
	@docker run -d -p 8000:8000 --name $(IMAGE_NAME) $(IMAGE_NAME):$(TAG)

# Stop the Docker container
stop:
	@echo "Stopping Docker container..."
	@docker stop $(IMAGE_NAME) || true

# View logs of the running container
logs:
	@echo "Viewing container logs..."
	docker logs -f $(IMAGE_NAME)

# Run the FastAPI app locally (for development)
api:
	@echo "Starting FastAPI app on http://localhost:8000 ..."
	$(PYTHON) -m uvicorn api:app --reload

# Install pre-commit hooks
deploy-hooks: install-hooks
install-hooks:
	@echo "Installing pre-commit hooks..."
	pre-commit install

# Clean up Docker images and cache
clean:
	@echo "Cleaning up..."
	@docker rmi $(IMAGE_NAME):$(TAG) || true
	@rm -rf __pycache__ src/__pycache__ .pytest_cache
