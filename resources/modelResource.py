import pandas as pd
from flask_restful import Resource
from flask import request, jsonify
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load the fine-tuned BioBERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained("./fine_tuned_biobert")
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

class modelsPOSTResource(Resource):
    def post(self):
        message = request.json["message"]
        symptom_text = message.strip()
    
        # Provide a medical context (can be a relevant treatment guideline or other text)
        context = "This is the context containing relevant medical information regarding diagnoses and treatments."

        # Tokenize the input (symptom text as a question)
        inputs = tokenizer.encode_plus(symptom_text, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get the model's prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits

        # Get the most likely answer span
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Decode the answer span
        answer_ids = input_ids[0][start_index : end_index + 1]
        treatment = tokenizer.decode(answer_ids, skip_special_tokens=True)
        return treatment
        