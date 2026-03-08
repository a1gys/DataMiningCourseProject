import argparse
import joblib

import torch
from transformers import AutoTokenizer ,AutoModelForSequenceClassification

from logistic_regression import clean_text

CLASS_MAP = {
    0: "democrat",
    1: "republican"
}

MODEL_NAME = "algys/bertweet-finetune"
LOGREG_PATH = "./logreg/logistic_regression_pipeline.joblib"

def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer

def predict_bert(texts: list[str],
                 model,
                 tokenizer) -> list[str]:
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.inference_mode():
        outputs = model(**inputs)
    
    preds = outputs.logits.argmax(dim=-1)
    return [CLASS_MAP[pred.item()] for pred in preds]

def load_logreg():
    return joblib.load(LOGREG_PATH)

def predict_logreg(texts: list[str],
                   model) -> list[str]:
    preds = model.predict(texts)
    return [CLASS_MAP[int(pred)] for pred in preds]

def read_inputs(args) -> list[str]:
    if args.text:
        texts = [args.text]
        return texts
    
    elif args.file:
        with open(args.file) as file:
            texts = [line.strip() for line in file if line.strip()]
        return texts

    else:
        raise ValueError("Provide --text or --file")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["bertweet", "logreg"], required=True)
    parser.add_argument("--text", type=str, help="Single input sentence")
    parser.add_argument("--file", type=str, help="Path to text file containing one sentence per line")

    args = parser.parse_args()

    texts = read_inputs(args)

    if args.model == "bertweet":
        model, tokenizer = load_bert()
        preds = predict_bert(texts, model, tokenizer)
    else:
        clean_texts = [clean_text(text) for text in texts]
        model = load_logreg()
        preds = predict_logreg(clean_texts, model)

    for text, pred in zip(texts, preds):
        print(f"{pred}: {text}")

if __name__ == "__main__":
    main()
