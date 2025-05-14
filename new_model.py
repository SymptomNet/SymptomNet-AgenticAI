from transformers import pipeline

pipe = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis")

result = pipe("I've been having headaches and migraines, and I can't sleep. My whole body shakes and twitches. Sometimes I feel lightheaded. give me the 3 most likely diagnosis I'm having")

print(result.logits)
