import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import os

# Step 1: Load the preprocessed datasets
train_dataset = load_from_disk("../data/processed/train")
valid_dataset = load_from_disk("../data/processed/valid")
print(f"Loaded {len(train_dataset)} training samples and {len(valid_dataset)} validation samples.")

# Step 2: Load the T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Step 3: Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="../models/t5-treatment",
    num_train_epochs=3, # Small dataset, so 3 epochs should be enough
    per_device_train_batch_size=4,
    per_gpu_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True if torch.cuda.is_available() else False, # Enable mixed precision for GPU
)

# Step 5: Define a data collator (optional, Trainer handles this automatically for T5)
# T5 expects input_ids, attention_mask, and labels
def data_collator(features):
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the trained model
os.makedirs("../models/t5-treatment/final", exist_ok=True)
model.save_pretrained("../models/t5-treatment/final")
tokenizer.save_pretrained("../models/t5-treatment/final")
print("Saved trained model to models/t5-treatment/final/")



