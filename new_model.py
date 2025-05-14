from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Load the tokenizer and model from TensorFlow weights
tokenizer = AutoTokenizer.from_pretrained("Zabihin/Symptom_to_Diagnosis")
model = TFAutoModelForSequenceClassification.from_pretrained("Zabihin/Symptom_to_Diagnosis")

# Define the symptom input
symptoms = "fever, cough, shortness of breath"

# Tokenize the input
inputs = tokenizer(symptoms, return_tensors="tf")

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
probabilities = tf.nn.softmax(logits, axis=1)
top_k = tf.math.top_k(probabilities, k=3)

# Display the top 3 predicted diagnoses
res = []
for idx, (label_id, score) in enumerate(zip(top_k.indices[0].numpy(), top_k.values[0].numpy()), 1):
    label = model.config.id2label[label_id]
    res.append(label)
    
print(res)
