import os
import time
import json

import torch
import pandas as pd
import numpy as np

from box import Box
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_hypotheses(text: str):
    return [
        f"This tweet expresses support for the Democratic party. Tweet: {text}",
        f"This tweet expresses support for the Republican party. Tweet: {text}",
    ]

def build_deberta_zero_shot(config: Box):
    model_hf_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    config.model_hf_name = model_hf_name
    return None

def inference_deberta(config: Box, dataset: pd.DataFrame):
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_hf_name).to(config.device)

    model.eval()

    predictions = []
    ground_truth = dataset["party"].tolist()

    start_time = time.perf_counter()

    print(f"Starting Zero-Shot Inference on {len(dataset)} samples")

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text = row["text"]

        hypotheses = get_hypotheses(text)

        inputs = tokenizer(
            [text, text],
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        entailment_scores = logits[:, 1]

        pred_idx = torch.argmax(entailment_scores).item()

        if pred_idx == 0:
            answer = "democrat"
            predictions.append("D")
        else:
            answer = "republican"
            predictions.append("R")

        # print(f"{idx}: {answer} | scores={entailment_scores.tolist()}")

    end_time = time.perf_counter()

    acc = accuracy_score(ground_truth, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="macro"
    )

    run_data = {
        "metadata": {
            "model_type": "DeBERTa-ZeroShot-NLI",
            "model_name": config.model_hf_name,
            "timestamp": datetime.now().isoformat(),
            "config": dict(config)
        },
        "results": {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "total_inference_seconds": round(end_time - start_time, 2),
            "avg_seconds_per_sample": round((end_time - start_time) / len(dataset), 4)
        },
        "predictions": predictions
    }

    log_dir = os.path.join(config.log_dir, "zero_shot")
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"deberta_results_{datetime.now().strftime('%H%M%S')}.json"

    with open(os.path.join(log_dir, log_name), "w") as f:
        json.dump(run_data, f, indent=4)

    print(f"\nInference Complete. F1: {f1:.4f} | Acc: {acc:.4f}")