# Political Party Classification from Twitter Posts

DS 504 Data Mining and Decision Support - Team 11

## Team Members

- **Sultanbek Mukhamedkulov** - Data Lead & Baseline Model
- **Baimyrza Kalmyrzayev** - Target Model Development  
- **Mohammad Hashim Halimi** - Evaluation & Metrics
- **Sayed Yasar Ahmad Mushtaq** - Data Preparation & Documentation

## Project Overview

This project classifies political party affiliation (Democrat vs Republican) from Twitter posts using two approaches:

1. **Baseline**: Logistic Regression with TF-IDF features
2. **Target**: Fine-tuned BERTweet model

## Dataset

**Source**: [Political Partisanship Tweets](https://www.kaggle.com/datasets/kapastor/democratvsrepublicantweets)

- ~2.8M tweets from political figures
- ~240k tweets from ordinary users
- Binary classification: Democrat (D) vs Republican (R)
- We used 300k balanced samples (150k per class)

### Data Download

The full dataset is not included in this repository. To download:

```bash
# Download manually from Kaggle
# https://www.kaggle.com/datasets/kapastor/democratvsrepublicantweets
# Extract to: data/tweets_balanced.csv
```

A small sample dataset (`data/sample_tweets.csv`) with 100 tweets is included for testing.

## Setup

### Requirements

- Python 3.12.10
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA GPU (recommended for BERTweet training)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd political-party-classifier

# Create virtual environment and install dependencies using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Alternative: using standard pip
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
```

## Usage

### 1. Train Baseline Model (Logistic Regression)

```bash
python logistic_regression.py
```

**Output:**
- Model: `logistic_regression_pipeline.joblib`
- Training time: ~2-3 minutes (CPU)
- Uses 5-fold cross-validation with grid search

### 2. Train Target Model (BERTweet)

```bash
python bertweet_finetune.py
```

**Output:**
- Model directory: `bertweet_classifier_new/`
- Training time: ~5 hours (GPU required)
- Uses 80/10/10 train/val/test split

### 3. Run Inference

```bash
# Single tweet
python inference.py --model bertweet --text "Your tweet here"
python inference.py --model logreg --text "Your tweet here"

# Batch from file
python inference.py --model bertweet --file data/example_tweets.txt
python inference.py --model logreg --file data/example_tweets.txt
```

## Results

### Performance Comparison

| Metric | Baseline (LogReg) | Target (BERTweet) | Improvement |
|--------|-------------------|-------------------|-------------|
| Accuracy | 75.0% | 82.9% | +7.9% |
| F1-Macro | 75.0% | 82.9% | +7.9% |
| Precision | 75.2% | 83.0% | +7.8% |
| Recall | 75.0% | 82.9% | +7.9% |

### Computational Comparison

| Aspect | Logistic Regression | BERTweet |
|--------|-------------------|----------|
| Training Time | 2-3 minutes | ~5 hours |
| Hardware | CPU | GPU (A40 40GB) |
| Model Size | ~5 MB | ~540 MB |

## Repository Structure

```
root_folder/
├── logistic_regression.py       # Train baseline model
├── bertweet_finetune.py         # Fine-tune BERTweet model
├── inference.py                 # Run predictions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/
│   ├── sample_tweets.csv        # Sample data (100 tweets)
│   ├── example_tweets.txt       # Example inputs for inference
│   └── tweets_balanced.csv      # Full dataset
├── notebooks/                   # Jupyter notebooks
└── results/                     # Output plots and metrics
```

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- `RANDOM_STATE = 42` in all scripts
- Stratified train/test splits
- Same preprocessing pipeline