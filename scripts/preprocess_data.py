import pandas as pd
from transformers import T5Tokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os

# Step 1: Load the curated dataset
treatment_df = pd.read_csv("../data/processed/treatment_data.csv")
print(f"Loaded {len(treatment_df)} disease-treatment pairs")

# Step 2: Prepare the data for T5 (prefix the input with a task description)
# T5 expects a task prefix like "treat disease:" before the input
treatment_df["input_text"] = "treat disease: " + treatment_df["disease"].astype(str)
treatment_df["target_text"] = treatment_df["treatment"].astype(str)

# Step 3: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(treatment_df[["input_text", "target_text"]])

# Step 4: Load T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
max_input_length = 32 # Disease names are short
max_target_length = 1024 # Treatments can be longer

# Step 5: Tokenize the dataset
def tokenize(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )

    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)

# Step 6: Split into train/validation/test sets (80/10/10)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

train_dataset = train_test_split["train"]
valid_dataset = test_valid_split["train"]
test_dataset = test_valid_split["test"]

print(f"Train set: {len(train_dataset)} samples")
print(f"Validation set: {len(valid_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")

# Step 7: Save the preprocessed datasets
os.makedirs("../data/processed/splits", exist_ok=True)
train_dataset.save_to_disk("../data/processed/splits/train")
valid_dataset.save_to_disk("../data/processed/splits/valid")
test_dataset.save_to_disk("../data/processed/splits/test")
print("Saved preprocessed datasets to data/processed/splits/")