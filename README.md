# Implement-End-to-End-Project-with-HuggingFace

## Project Overview

The primary objective of this project is to enable colleagues within a company to utilize a pre-trained HuggingFace sentiment analysis model seamlessly. This is achieved through creating an API endpoint using FastAPI and containerizing the application with Docker for simplified deployment.

### Step 1: Choosing our HuggingFace Model

1. **Install necessary libraries:**
   ```bash
   pip install transformers torch  # or pip install tensorflow
   ```

2. **Import and define the model using the HuggingFace pipeline:**
   ```python
   from transformers import pipeline
   
   # Load the sentiment analysis pipeline
   pipe = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
   
   # Test the model
   print(pipe("This tutorial is great!"))
   # Output: [{'label': 'POSITIVE', 'score': 0.9998689889907837}]
   ```

3. **Define a function to generate a natural language response:**
   ```python
   def generate_response(prompt: str):
       response = pipe(prompt)
       label = response[0]["label"]
       score = response[0]["score"]
       return f"The '{prompt}' input is {label} with a score of {score}"
   ```

### Step 2: Write API Endpoint for the Model with FastAPI

1. **Install FastAPI and pydantic:**
   ```bash
   pip install fastapi pydantic uvicorn
   ```

2. **Define the FastAPI app:**
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   from transformers import pipeline
   
   # Load the sentiment analysis pipeline
   pipe = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
   
   # Initialize FastAPI
   app = FastAPI()
   
   # Define request model
   class RequestModel(BaseModel):
       input: str
   
   @app.post("/sentiment")
   def get_response(request: RequestModel):
       prompt = request.input
       response = pipe(prompt)
       label = response[0]["label"]
       score = response[0]["score"]
       return f"The '{prompt}' input is {label} with a score of {score}"
   ```

### Step 3: Use Docker to Run Our Model

1. **Create a Dockerfile:**
   ```dockerfile
   # Use an official Python runtime as a parent image
   FROM python:3.10-slim
   
   # Set the working directory in the container
   WORKDIR /sentiment
   
   # Copy the requirements.txt file into the root
   COPY requirements.txt .
   
   # Copy the current directory contents into the container at /app as well
   COPY ./app ./app
   
   # Install any needed packages specified in requirements.txt
   RUN pip install -r requirements.txt
   
   # Make port 8000 available to the world outside this container
   EXPOSE 8000
   
   # Run main.py when the container launches
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run the Docker image:**
   ```bash
   docker build -t sentiment-app .
   docker run -p 8000:8000 --name sentiment-app sentiment-app
   ```

### Installation

To get started with this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/AyaFergany/Implement-End-to-End-Project-with-HuggingFace.git
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the FastAPI app locally:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. Send a POST request to the API endpoint:
   ```bash
   curl -X POST "http://127.0.0.1:8000/sentiment" \
        -H "Content-Type: application/json" \
        -d '{"input":"This tutorial is great!"}'
   ```

---

