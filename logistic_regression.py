import warnings
import re
import time
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PARAM_GRID = {
    "tfidf__max_features": [50_000, 100_000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "tfidf__min_df":       [2, 5],
    "clf__C":              [0.01, 0.1, 1.0, 10.0],
    "clf__l1_ratio":       [0.0, 1.0],
}


def load_data(filepath: str):
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {list(df.columns)}")
    print(f"\nParty distribution:\n{df['party'].value_counts()}")
    print(f"\nSource distribution:\n{df['source'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"^rt\s+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Preprocessing] Cleaning tweets...")
    df = df.dropna(subset=["text", "party"]).copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df["label"] = (df["party"].str.strip().str.upper() == "R").astype(int)
    print(f"Final dataset  : {df.shape[0]:,} rows")
    print(f"Label balance  : D={( df['label']==0).sum():,} | R={(df['label']==1).sum():,}")
    return df

def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])

def run_grid_search(pipeline, X_train, y_train, cv: int = 5):
    print("\n[GridSearchCV] Starting hyperparameter search …")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    start = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"\nGrid search done in {elapsed/60:.1f} min")
    print(f"Best params  : {gs.best_params_}")
    print(f"Best CV F1   : {gs.best_score_:.4f}")
    return gs


if __name__ == "__main__":
    df = load_data(filepath="./data/tweets_balanced.csv")
    df = preprocess(df)

    X = df["clean_text"]
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.10,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    pipeline = build_pipeline()
    gs = run_grid_search(pipeline, X_train, y_train, cv=5)
    best_model = gs.best_estimator_

    joblib.dump(best_model, "logistic_regression_pipeline.joblib")