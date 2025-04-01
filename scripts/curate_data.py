import pandas as pd
import os

# Load MedQuAD dataset
medquad_df = pd.read_csv("../data/raw/medquad.csv")
# Clean the question and answer columns
medquad_df["question"] = medquad_df["question"].str.strip()
if "answer" in medquad_df.columns:
    medquad_df["answer"] = medquad_df["answer"].str.strip()
else:
    print("Warning: 'answer' column not found in the dataset.")
    
# Step 2: Filter for treatment-related questions
def is_treatment_question(question):
    question = question.lower()
    return "treatment" in question or "how to treat" in question

def is_treatment_answer(answer):
    # Check if the answer contains treatment-related keywords
    answer = answer.lower()
    treatment_keywords = [
        "medication", "surgery", "therapy", "treatment", "manage", "control",
        "cure", "relieve", "drug", "procedure", "lifestyle", "exercise", "diet",
        "insulin", "antibiotics", "antidepressants", "anticonvulsants", "alcohol", 
        "prescription", "medicine", "test", "exam", "shots", "exercise", "food",
        "caffeine", "smoking", "aid"
    ]
    resource_phrases = [
        "these resources address",
        "genetic testing registry",
        "content on this page",
        "insurance",
        "national eye institute",
        "national institutes of health",
        "glaucoma foundation",
        "glaucoma research foundation"
    ]
    # Answer must contain at least one treatment keyword and not be a resource list
    has_treatment_keyword = any(keyword in answer for keyword in treatment_keywords)
    is_resource_list = any(phrase in answer for phrase in resource_phrases)
    return has_treatment_keyword and not is_resource_list

# Filter for treatment questions
treatment_df = medquad_df[medquad_df["question"].apply(is_treatment_question)]
# Further filter for answers that describe treatments
treatment_df = treatment_df[treatment_df["answer"].apply(is_treatment_answer)]


# Step 3: Create the curated dataset with disease and treatment
# Group by disease and combine all treatment answers into a single entry
curated_df = treatment_df.groupby("focus_area")["answer"].apply(lambda x: "".join(x)).reset_index()
curated_df = curated_df.rename(columns={"focus_area": "disease", "answer": "treatment"})

# Step 4: Analyze the dataset
print("Number of diseases with treatment information:", len(curated_df))
print("Sample diseases and treatments:")
print(curated_df.head())

# Step 5: Save to CSV
os.makedirs("../data/processed", exist_ok=True)
curated_df.to_csv("../data/processed/treatment_data.csv", index=False);
print(f"Curated {len(curated_df)} disease-treatment pairs to data/processed/treatment_data.csv")


