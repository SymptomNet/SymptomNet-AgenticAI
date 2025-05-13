from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load the fine-tuned BioBERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained("./fine_tuned_biobert")
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def get_treatment(symptom_text):
    symptom_text = symptom_text.strip()
    
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

# Test the function with an example input
symptom_input = "What is the treatment for fever and cough?"
result = get_treatment(symptom_input)
if result:
    print(f"Suggested treatment: {result}")
