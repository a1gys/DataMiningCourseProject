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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor


def get_partisan_prompt(text: str) -> str:
    return f"""<start_of_turn>user
Classify the following tweet into one of two political parties: 'Democrat' or 'Republican'. 
Base your decision on the sentiment, policy focus, and terminology used.

Tweet: "{text}"

Respond with ONLY the word 'Democrat' or 'Republican'. No explanations.<end_of_turn>
<start_of_turn>model
"""

def build_gemma_zero_shot(config: Box):
    # model_hf_name = "google/gemma-2-27b-it"
    model_hf_name = "google/gemma-4-26B-A4B-it"
    config.model_hf_name = model_hf_name
    
    return None

def inference_gemma(config: Box,
                    dataset: pd.DataFrame):
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    processor = AutoProcessor.from_pretrained(config.model_hf_name)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_hf_name,
        dtype="auto",
        device_map="auto"
    )

    model.eval()
    predictions = []
    if config.dataset == "twitter_partisan":
        ground_truth = dataset["party"].tolist()
    elif config.dataset == "mbib":
        ground_truth = dataset["label"].tolist()

    start_time = time.perf_counter()

    print(f"Starting Zero-Shot Inference on {len(dataset)} samples")

    contents = {
        "twitter_partisan": "Classify the following tweet into one of two political parties: 'Democrat' or 'Republican'. Base your decision on the sentiment, policy focus, and terminology used. Respond with ONLY the word 'Democrat' or 'Republican'. No explanations.",
        "mbib": "Classifty the following tweet on whether it contains political bias '1' or no '0'. Respond only with '1' or '0'. No explanations."
    }

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        messages = [
            {"role": "system", "content": contents[config.dataset]},
            {"role": "user", "content": row["text"]}
        ]

        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=False)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = model.generate(**inputs, max_new_tokens=1024)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        processor.parse_response(response)

        if config.dataset == "twitter_partisan":
            if "democrat" in response.lower():
                predictions.append("D")
            elif "republican" in response.lower():
                predictions.append("R")
            else:
                predictions.append("U")
        elif config.dataset == "mbib":
            if "0" in response.lower():
                predictions.append(0)
            elif "1" in response.lower():
                predictions.append(1)
            else:
                predictions.append(2)
    
    end_time = time.perf_counter()

    acc = accuracy_score(ground_truth, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="macro"
    )

    run_data = {
        "metadata": {
            "model_type": "Gemma-ZeroShot",
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
        "predictions": predictions # Storing raw outputs for error analysis
    }

    log_dir = os.path.join(config.log_dir, "zero_shot")
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"gemma_results_{datetime.now().strftime('%H%M%S')}.json"
    
    with open(os.path.join(log_dir, log_name), "w") as f:
        json.dump(run_data, f, indent=4)

    print(f"\nInference Complete. F1: {f1:.4f} | Acc: {acc:.4f}")