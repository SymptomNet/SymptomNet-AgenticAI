from flask_restful import Resource
from flask import request, jsonify
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load components
tokenizer = AutoTokenizer.from_pretrained("Zabihin/Symptom_to_Diagnosis")
model = TFAutoModelForSequenceClassification.from_pretrained("Zabihin/Symptom_to_Diagnosis")

class modelsPOSTResource(Resource):
    def post(self):
        message = request.json['Message']
        print(f"Processing: {message}")
        
        # Tokenize with guaranteed attention_mask
        inputs = tokenizer(
            message,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # Explicitly extract required inputs (ignore other keys like 'token_type_ids')
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]  # Must exist due to padding
        
        # Pass ONLY the arguments the model expects
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,  # Disable unwanted outputs
            output_hidden_states=False
        )
        
        probabilities = tf.nn.softmax(outputs.logits, axis=1)
        top_k = tf.math.top_k(probabilities, k=3)
        
        res = [model.config.id2label[label_id] for label_id in top_k.indices.numpy()[0]]
        return jsonify({"result": res})