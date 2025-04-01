# health-guard
ML system that retrieves treatments for diseases using the MedQuAD dataset, with infrastructure automation for training and deployment.

**Dataset**: MedQuAD (47,457 Q&A pairs from Kaggle)

## Setup Instructions:
  1. Create a virtual environment: `python3.11 -m venv venv`
  2. Activate the environment: `source venv/bin/activate`
  3. Install dependencies: `pip install datasets transformers pandas scikit-learn torch accelerate`
  4. Download the MedQuAD dataset from Kaggle and place it in `data/raw/medquad.csv`

## Scripts:
  - `scripts/curate_data.py`: Curates and cleans the raw data.
  - `scripts/preprocess_data.py`: Preprocesses the curated data.
  - `scripts/train_model.py`: Trains the model.
  - `scripts/test_model.py`: Tests the trained model.
