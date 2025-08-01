# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Start initial training, then run the API and the Python scheduler concurrently.
# The scheduler runs in the background (&), and uvicorn runs in the foreground.
CMD ["sh", "-c", "python main.py && python scheduler.py & uvicorn api:app --host 0.0.0.0 --port 8000"]
