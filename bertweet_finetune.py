import pandas as pd
import evaluate
import numpy as np

from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback)
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/tweets_balanced.csv")
label_map = {
    "D": 0,
    "R": 1
}

df["label"] = df["party"].map(label_map)

df = df[["text", "label"]].reset_index(drop=True)


train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)


train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_ds = Dataset.from_pandas(val_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]

train_ds.set_format("torch", columns=cols)
val_ds.set_format("torch", columns=cols)
test_ds.set_format("torch", columns=cols)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2)

f1_metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuarcy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro":  f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        "precision": precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall":    recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"],
    }

training_args = TrainingArguments(
    output_dir="./results/results2",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=1e-2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=True,
    gradient_checkpointing=True,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

print(f"\n----Validation results----")
val_results = trainer.evaluate(val_ds)
print(val_results)

print(f"\n----Test results----")
test_results = trainer.evaluate(test_ds)
print(test_results)

trainer.save_model("./bertweet_classifier_new")
tokenizer.save_pretrained("./bertweet_classifier_new")