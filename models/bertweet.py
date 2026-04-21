import time
import os
import json
import datetime

import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          DataCollatorWithPadding,
                          EarlyStoppingCallback)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from box import Box


RANDOM_STATE = 42

def tokenize(tokenizer,
             batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def build_bertweet(config: Box):
    model_hf_name = "vinai/bertweet-base"
    config.model_hf_name = model_hf_name
    model = AutoModelForSequenceClassification.from_pretrained(model_hf_name, num_labels=2)
    return model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision": float(prec),
        "recall": float(rec)
    }

def get_best_checkpoint(config: Box) -> str:
    config_num = config.config_path.split("/")[-1].replace("config_", "").replace(".yaml", "")
    
    curr_ckpt_path = os.path.join(config.checkpoint_dir, config_num)
    folds = os.listdir(curr_ckpt_path)

    best_f1 = 0.0
    best_fold = 0
    
    run_path = os.path.join(curr_ckpt_path, "fold_4", "run.json")

    with open(run_path) as file:
        run_data = json.load(file)

        for fold in range(5):
            f1 = run_data["folds"][fold]["validation"]["f1_macro"]

            if f1 > best_f1:
                best_f1 = f1
                best_fold = fold
    best_path = os.path.join(curr_ckpt_path, f"fold_{best_fold}")
    return best_path


def train_bertweet(model,
                   config: Box,
                   dataset: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained(config.model_hf_name, use_fast=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if config.dataset == "twitter_partisan":
        stratify_key = dataset['source'] + "_" + dataset['party']
    elif config.dataset == "mbib":
        stratify_key = dataset["label"]

    run_data = {
        "metadata": {"model_type": "BERTweet", "config": dict(config)},
        "folds": [],
        "histories": []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, stratify_key)):
        print(f"--- FOLD {fold + 1} ---")
        fold_start = time.perf_counter()
        
        train_df = dataset.iloc[train_idx]
        val_df = dataset.iloc[val_idx]
        
        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']])

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=config.model.max_length)
        
        train_ds = train_ds.map(tokenize_function, batched=True)
        val_ds = val_ds.map(tokenize_function, batched=True)

        config_name = config.config_path.split("/")[-1].replace("config", "train").replace(".yaml", "")
        log_dir = os.path.join(config.log_dir, "train", config_name)
        output_dir = os.path.join(log_dir, f"fold_{fold}")
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.model.epochs,
            per_device_train_batch_size=config.model.batch_size,
            per_device_eval_batch_size=config.model.batch_size,
            learning_rate=config.model.learning_rate,
            weight_decay=config.model.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch", # Changed to save weights at best/last epoch
            load_best_model_at_end=True, # Required for EarlyStopping
            metric_for_best_model="f1_macro",
            logging_steps=10,
            report_to="none",
            lr_scheduler_type=config.model.lr_scheduler_type,
            warmup_ratio=config.model.warmup_ratio,
            label_smoothing_factor=config.model.label_smoothing,
        )

        fold_model = build_bertweet(config)
        
        trainer = Trainer(
            model=fold_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        train_start = time.perf_counter()
        trainer.train()
        train_end = time.perf_counter()
        
        history = []
        for entry in trainer.state.log_history:
            if "eval_loss" in entry:
                history.append({
                    "epoch": entry.get("epoch"),
                    "val_loss": entry.get("eval_loss"),
                    "val_f1_macro": entry.get("eval_f1_macro"),
                    "val_accuracy": entry.get("eval_accuracy"),
                    "train_loss": entry.get("loss") 
                })

        checkpoint_path = os.path.join(config.checkpoint_dir, config_name, f"fold_{fold}")
        os.makedirs(checkpoint_path, exist_ok=True)
        trainer.save_model(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)



        val_metrics = trainer.evaluate()
        

        fold_end = time.perf_counter()

        fold_entry = {
            "fold": fold,
            "timings": {
                "training_seconds": round(train_end - train_start, 4),
                "total_fold_seconds": round(fold_end - fold_start, 4)
            },
            "validation": {
                "loss": val_metrics["eval_loss"],
                "f1_macro": val_metrics["eval_f1_macro"],
                "accuracy": val_metrics["eval_accuracy"]
            }
        }
        run_data["folds"].append(fold_entry)

        run_data["histories"].append(history) 

        run_path = os.path.join(checkpoint_path, "run.json")
        with open(run_path, "w") as file:
            json.dump(run_data, file, indent=4)


def test_bertweet(config: Box,
                  dataset: pd.DataFrame):
    model_path = get_best_checkpoint(config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)


    test_ds = Dataset.from_pandas(dataset[['text', 'label']])

    def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=config.model.max_length)

    test_ds = test_ds.map(tokenize_function, batched=True)

    log_dir = os.path.join(config.log_dir, "test")
    os.makedirs(log_dir, exist_ok=True)

    test_args = TrainingArguments(
        output_dir=log_dir,
        per_device_eval_batch_size=config.model.batch_size,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=test_args,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics, # Uses the same metrics function from training
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("Running inference on test set...")
    test_start = time.perf_counter()
    results = trainer.evaluate()
    test_end = time.perf_counter()

    # 6. Save Test Results
    test_results = {
        "metadata": {
            "model_type": "BERTweet",
            "checkpoint_used": model_path,
            "test_timestamp": datetime.now().isoformat()
        },
        "metrics": {
            "loss": float(results["eval_loss"]),
            "accuracy": float(results["eval_accuracy"]),
            "f1_macro": float(results["eval_f1_macro"]),
            "precision": float(results["eval_precision"]),
            "recall": float(results["eval_recall"])
        },
        "timings": {
            "test_duration_seconds": round(test_end - test_start, 4)
        }
    }

    log_name = config.config_path.split("/")[-1].replace("config", "test").replace("yaml", "json")
    with open(os.path.join(log_dir, log_name), "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"Testing complete. F1 Score: {test_results['metrics']['f1_macro']:.4f}")
    print(f"Test logs saved to {log_dir}")


if __name__ == "__main__":
    config = Box({})
    config_path = "/home/gpuhead-2/data_mining/DataMiningCourseProject/configs/bertweet/config_306.yaml"
    config.config_path = config_path
    config.log_dir = "/home/gpuhead-2/data_mining/DataMiningCourseProject/logs/bertweet_mbib"
    config.checkpoint_dir = "/home/gpuhead-2/data_mining/DataMiningCourseProject/checkpoints/bertweet_mbib"

    config_num = config.config_path.split("/")[-1].replace("config_", "").replace(".yaml", "")
    
    curr_ckpt_path = os.path.join(config.checkpoint_dir, config_num)
    folds = os.listdir(curr_ckpt_path)

    best_f1 = 0.0
    best_fold = 0
    
    run_path = os.path.join(curr_ckpt_path, "fold_5", "run.json")

    with open(run_path) as file:
        run_data = json.load(file)

        for fold in range(1, 5):
            f1 = run_data["folds"][fold]["validation"]["f1_macro"]

            if f1 > best_f1:
                best_f1 = f1
                best_fold = fold


    best_path = os.path.join(curr_ckpt_path, f"fold_{best_fold}")
    
    tokenizer = AutoTokenizer.from_pretrained(best_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(best_path)