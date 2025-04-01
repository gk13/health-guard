from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the trained model and tokenizer
model_path = "../models/t5-treatment/final"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test inference
disease = "Glaucoma"
input_text = f"treat disease: {disease}"
inputs = tokenizer(input_text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate the treatment
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=512,
    num_beams=4,
    early_stopping=True
)

# Decode the output
treatment = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Treatment for {disease}: {treatment}")