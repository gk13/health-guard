# health-guard
ML system that retrieves treatments for diseases using the MedQuAD dataset, with infrastructure automation for training and deployment. The system will validate treatment claims using Convergence's Proxy-lite.

**Dataset**: MedQuAD (47,457 Q&A pairs from Kaggle)

## Setup Instructions:
  1. Create a virtual environment: `python3.11 -m venv venv`
  2. Activate the environment: `source venv/bin/activate`
  3. Install dependencies: `pip install datasets transformers pandas scikit-learn`
  4. Download the MedQuAD dataset from Kaggle and place it in `data/raw/medquad.csv`
  5. Run the curation script: `python scripts/curate_data.py`
  6. Run the preprocessing script: `python scripts/preprocess_data.py`
