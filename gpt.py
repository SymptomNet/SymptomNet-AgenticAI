from openai import OpenAI
from transformers import pipeline
import requests
from datetime import datetime

# Define your OpenAI API key
client = OpenAI(api_key='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')

# Function to preprocess input (e.g., symptoms from the user)
def preprocess_symptoms(symptom_text):
    # You can add more preprocessing steps here as needed.
    return symptom_text.strip()

# Function to query OpenAI's latest API with GPT-3 or GPT-4 models
def get_treatment(symptom_text):
    symptom_text = preprocess_symptoms(symptom_text)
    
    # Send a prompt to OpenAI's GPT model with medical symptoms
    prompt = f"Given the following symptoms: {symptom_text}, what is the most likely diagnosis and treatment?"
    
    response = client.responses.create(
        model="gpt-3.5-turbo",  # You can switch between gpt-3.5-turbo, gpt-4, etc.
        instructions="You are a helpful medical assistant that knows the treatment solution given a set of symptoms proviede by the user",
        input = prompt
    )
    
    treatment = response.output_text.strip()
    return treatment

# Function to interact with the model in a user-friendly way
def query_medical_symptom(symptom_text):
    try:
        print("Processing your symptoms, please wait...")
        treatment = get_treatment(symptom_text)
        return treatment
    except Exception as e:
        return f"An error occurred while processing your request: {e}"

# Test function
symptom_input = "Provide the 3 most likely diseases given the syumptoms of fever, cough, fatigue, and shortness of breath"
result = query_medical_symptom(symptom_input)
print(f"Suggested treatment: {result}")
