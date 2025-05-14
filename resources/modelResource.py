from flask_restful import Resource
from flask import request, jsonify
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the tokenizer and model from TensorFlow weights
tokenizer = AutoTokenizer.from_pretrained("Zabihin/Symptom_to_Diagnosis")
model = TFAutoModelForSequenceClassification.from_pretrained("Zabihin/Symptom_to_Diagnosis")

class modelsPOSTResource(Resource):
    def post(self):
        message = request.json['Message']
        print(message)
        # Tokenize the input
        inputs = tokenizer(message, return_tensors="tf")
    
        # Get predictions
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1)
        top_k = tf.math.top_k(probabilities, k=3)
        
        # Display the top 3 predicted diagnoses
        res = []
        for _, (label_id, _) in enumerate(zip(top_k.indices[0].numpy(), top_k.values[0].numpy()), 1):
            label = model.config.id2label[label_id]
            res.append(label)
            
        return jsonify({"result": res})