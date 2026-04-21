import warnings
import json
import time
import os
import joblib

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from box import Box

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def build_logreg(config: Box) -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=config.tf_idf.max_features,
        ngram_range=tuple(config.tf_idf.ngram_range),
        sublinear_tf=config.tf_idf.sublinear_tf,
        min_df=config.tf_idf.min_df,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )
    clf = LogisticRegression(
        max_iter=1000,
        solver=config.model.solver,
        random_state=RANDOM_STATE,
        l1_ratio=config.model.l1_ratio
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def train_logreg(model: Pipeline,
                 config: Box,
                 dataset: pd.DataFrame):
    # X = dataset["text"]
    # y = dataset["label"]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if config.dataset == "twitter_partisan":
        stratify_key = dataset['source'] + "_" + dataset['party']
    elif config.dataset == "mbib":
        stratify_key = dataset["label"]

    run_data = {
        "metadata": {
            "model_type": "Logistic Regression",
            "random_state": RANDOM_STATE,
            "config": dict(config)
        },
        "folds": [],
        "final_summary": {}
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, stratify_key)):
        print(f"Processing Fold {fold + 1}...")
        fold_start = time.perf_counter()
        train_set = dataset.iloc[train_idx]
        val_set = dataset.iloc[val_idx]

        train_start = time.perf_counter()
        model.fit(train_set["text"], train_set["label"])
        train_end = time.perf_counter()

        def get_metrics(df):
            preds = model.predict(df["text"])
            probs = model.predict_proba(df["text"])

            acc = accuracy_score(df["label"], preds)
            prec, rec, f1, _ = precision_recall_fscore_support(df["label"], preds, average="macro")
            loss = log_loss(df["label"], probs)

            return {
                "loss": float(loss),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_macro": float(f1),
            }

        fold_end = time.perf_counter()

        fold_entry = {
            "fold": fold + 1,
            "timings": {
                "training_seconds": round(train_end - train_start, 4),
                "total_fold_seconds": round(fold_end - fold_start, 4)
            },
            "train": get_metrics(train_set),
            "validation": get_metrics(val_set),
        }
        run_data["folds"].append(fold_entry)
    
    val_f1s = [f["validation"]["f1_macro"] for f in run_data["folds"]]
    val_losses = [f["validation"]["loss"] for f in run_data["folds"]]

    run_data["final_summary"] = {
        "mean_val_f1": float(np.mean(val_f1s)),
        "std_val_f1": float(np.std(val_f1s)),
        "mean_val_loss": float(np.mean(val_losses))
    }

    log_name = config.config_path.split("/")[-1].replace("config", "logreg_train").replace("yaml", "json")
    log_dir = os.path.join(config.log_dir, "train")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    print(f"Writing logs to {log_path}")
    with open(log_path, "w") as file:
        json.dump(run_data, file, indent=4)

    checkpoint_name = config.config_path.split("/")[-1].replace("config", "logreg_train").replace("yaml", "joblib")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
    joblib.dump(model, checkpoint_path)


def test_logreg(config: Box,
                dataset: pd.DataFrame):
    checkpoint_name = config.config_path.split("/")[-1].replace("config", "logreg_train").replace("yaml", "joblib")
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
    model = joblib.load(checkpoint_path)

    preds = model.predict(dataset["text"])
    probs = model.predict_proba(dataset["text"])

    acc = accuracy_score(dataset["label"], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(dataset["label"], preds, average="macro")
    loss = log_loss(dataset["label"], probs)

    run_data = {
        "metadata": {
            "model_type": "Logistic Regression",
            "random_state": RANDOM_STATE,
            "config": dict(config)
        },
        "metrics": {
            "loss": float(loss),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_macro": float(f1),
        }
    }

    log_name = config.config_path.split("/")[-1].replace("config", "logreg_test").replace("yaml", "json")
    log_dir = os.path.join(config.log_dir, "test")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    print(f"Writing logs to {log_path}")
    with open(log_path, "w") as file:
        json.dump(run_data, file, indent=4)