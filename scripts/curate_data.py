import pandas as pd
import os

# Load MedQuAD dataset
medquad_df = pd.read_csv("../data/raw/medquad.csv")
# Clean the question and answer columns
medquad_df["question"] = medquad_df["question"].str.strip()
medquad_df["answer"] = medquad_df["answer"].str.strip()

# Step 2: Filter for treatment-related questions
def is_treatment_question(question):
    question = question.lower()
    return "treatment" in question or "how to treat" in question

treatment_df = medquad_df[medquad_df["question"].apply(is_treatment_question)]

# Step 3: Create the curated dataset with disease and treatment
curated_df = treatment_df[["focus_area", "answer"]]
curated_df = curated_df.rename(columns={"focus_area": "disease", "answer": "treatment"})
# Remove duplicates (some diseases may have multiple treatment answers; we'll take the first one for simplicity)
curated_df = curated_df.drop_duplicates(subset=["disease"], keep="first")

# Step 4: Analyze the dataset
print("Number of diseases with treatment information:", len(curated_df))
print("Sample diseases and treatments:")
print(curated_df.head())

# Step 5: Save to CSV
os.makedirs("../data/processed", exist_ok=True)
curated_df.to_csv("../data/processed/treatment_data.csv", index=False);
print(f"Curated {len(curated_df)} disease-treatment pairs to data/processed/treatment_data.csv")


