import pandas as pd
import os
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from collections import Counter

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for advanced text processing
nlp = spacy.load("en_core_web_sm")

# Load stop words
stop_words = set(stopwords.words('english'))

# Define boilerplate phrases to remove
boilerplate_phrases = [
    "click here", "learn more", "contact us", "all rights reserved",
    "privacy policy", "terms of use", "sign up", "log in", "subscribe now",
    "find out more", "visit our website", "call us", "email us",
    "for more information", "see also", "related links"
]

def clean_html(text):
    """Remove HTML tags and artifacts using BeautifulSoup."""
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    return cleaned_text

def remove_boilerplate(text):
    """Remove common boilerplate phrases."""
    text = text.lower()
    for phrase in boilerplate_phrases:
        text = text.replace(phrase, "")
    return text

def remove_stopwords(text):
    """Remove stop words using NLTK."""
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def reduce_repetition(text, max_repeats=2):
    """Reduce excessive repetition of words or phrases."""
    doc = nlp(text)
    tokens = [token.text for token in doc]
    token_counts = Counter(tokens)
    
    # Reduce repetition of any token that appears more than max_repeats times consecutively
    cleaned_tokens = []
    prev_token = None
    repeat_count = 0
    
    for token in tokens:
        if token == prev_token:
            repeat_count += 1
            if repeat_count < max_repeats:
                cleaned_tokens.append(token)
        else:
            cleaned_tokens.append(token)
            prev_token = token
            repeat_count = 0
    
    return " ".join(cleaned_tokens)

def normalize_text(text):
    """Normalize text by removing extra spaces, newlines, and converting to lowercase."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading/trailing spaces
    text = text.lower()  # Convert to lowercase
    return text

def clean_treatment_text(text):
    """Apply all cleaning steps to the treatment text."""
    # Step 1: Remove HTML tags
    text = clean_html(text)
    
    # Step 2: Remove boilerplate phrases
    text = remove_boilerplate(text)
    
    # Step 3: Remove stop words
    text = remove_stopwords(text)
    
    # Step 4: Reduce repetition
    text = reduce_repetition(text)
    
    # Step 5: Normalize text
    text = normalize_text(text)
    
    return text

# Step 1: Load MedQuAD dataset
medquad_df = pd.read_csv("../data/raw/medquad.csv")

# Clean the question and answer columns
medquad_df["question"] = medquad_df["question"].str.strip()
if "answer" in medquad_df.columns:
    medquad_df["answer"] = medquad_df["answer"].str.strip()
else:
    print("Warning: 'answer' column not found in the dataset.")
    exit()

# Step 2: Filter for treatment-related questions
def is_treatment_question(question):
    if pd.isna(question):
        return False
    question = question.lower()
    return "treatment" in question or "how to treat" in question

def is_treatment_answer(answer):
    if pd.isna(answer):
        return False
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

# Step 3: Clean the answers before grouping
treatment_df["answer"] = treatment_df["answer"].apply(clean_treatment_text)

# Filter out empty or short answers after cleaning
treatment_df = treatment_df[treatment_df["answer"].str.strip() != ""]
treatment_df = treatment_df[treatment_df["answer"].str.len() >= 10]

# Step 4: Create the curated dataset with disease and treatment
# Group by disease and combine all treatment answers into a single entry
curated_df = treatment_df.groupby("focus_area")["answer"].apply(lambda x: " ".join(x)).reset_index()
curated_df = curated_df.rename(columns={"focus_area": "disease", "answer": "treatment"})

# Remove duplicates (some diseases may have multiple treatment answers; we'll take the first one for simplicity)
curated_df = curated_df.drop_duplicates(subset=["disease"], keep="first")

# Step 5: Analyze the dataset
print("Number of diseases with treatment information:", len(curated_df))
print("Sample diseases and treatments:")
print(curated_df.head())

# Step 6: Save to CSV
os.makedirs("../data/processed", exist_ok=True)
curated_df.to_csv("../data/processed/treatment_data.csv", index=False)
print(f"Curated {len(curated_df)} disease-treatment pairs to data/processed/treatment_data.csv")