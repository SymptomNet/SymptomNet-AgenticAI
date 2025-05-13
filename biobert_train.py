from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Load the SQuAD dataset (or your own medical dataset)
dataset = load_dataset("squad", split={'train': 'train[:30%]', 'validation': 'validation[:30%]'})

def preprocess_function(examples):
    # Tokenize the question and context
    inputs = tokenizer(
        examples["question"], examples["context"],
        truncation="only_second",  # Adjust truncation strategy as necessary
        padding="max_length",  # Padding to max length for uniformity
        max_length=512,  # Set max length (512 for BERT)
        return_tensors="pt",
        return_offsets_mapping=True  # Important for extracting start and end positions
    )

    # Get the start and end positions for the answer span
    start_positions = []
    end_positions = []

    # Iterate over each example in the batch
    for i in range(len(examples["answers"])):
        # Extract the list of answers for this example
        answers = examples["answers"][i]

        # Extract the start position of the answer and answer text
        answer_start = answers["answer_start"][0]  # Ensure it's treated as a single integer
        answer_text = answers["text"][0]  # Use the first answer text if there are multiple

        # Use tokenizer.encode_plus to correctly tokenize the question and context
        encoding = tokenizer.encode_plus(
            examples["question"][i], examples["context"][i],  # Pass both question and context as a tuple
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            truncation="only_second",  # Ensure truncation if the context is too long
            padding="max_length",  # Pad to max length
            max_length=512,  # Ensure the length does not exceed max length
            return_tensors="pt",  # Return pytorch tensors
            return_offsets_mapping=True  # Important for extracting token start and end positions
        )

        # Get the token positions for the start and end of the answer
        offset_mapping = encoding['offset_mapping'][0]  # Get the offset mapping for this example

        # Find the start and end positions of the answer in the context
        start_char = answer_start
        end_char = start_char + len(answer_text) - 1  # Calculate the end char position

        # Find token positions corresponding to start_char and end_char
        start_token = -1
        end_token = -1

        for idx, (start, end) in enumerate(offset_mapping):
            if start_char >= start and start_char <= end:
                start_token = idx
            if end_char >= start and end_char <= end:
                end_token = idx
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    inputs["start_positions"] = torch.tensor(start_positions)
    inputs["end_positions"] = torch.tensor(end_positions)

    return inputs


# Apply tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set up TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    logging_dir='./logs',  # directory for logs
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    eval_steps=500,  # Check every 500 steps (for eval)
    logging_steps=500,  # Log every 500 steps
    eval_strategy="steps",  # Evaluate every eval_steps
    save_strategy="steps",  # Save checkpoint every save_steps
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="eval_loss",  # Use eval_loss to evaluate the best model
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_biobert")
