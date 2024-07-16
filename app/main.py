from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# You can check any other model in the Hugging Face Hub
pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# We define the app
app = FastAPI()

# We define that we expect our input to be a string
class RequestModel(BaseModel):
   input: str

# Now we define that we accept post requests
@app.post("/sentiment")
def get_response(request: RequestModel):
   prompt = request.input
   response = pipe(prompt)
   label = response[0]["label"]
   score = response[0]["score"]
   return f"The '{prompt}' input is {label} with a score of {score}"