import yaml
from itertools import product
from pathlib import Path

# --- 1. Define Search Space ---
# Common BERT fine-tuning ranges:
# Learning Rate: 2e-5, 3e-5, 5e-5
# Batch Size: 16, 32
# Epochs: 3, 4
search_space = {
    "model": {
        "model_name": ["vinai/bertweet-base"],

        "learning_rate": [2e-5, 3e-5, 5e-5],
        "lr_scheduler_type": ["linear", "cosine"],
        "warmup_ratio": [0.0, 0.1],

        # training setup
        "batch_size": [128, 256],
        "epochs": [3, 5],

        # regularization
        "weight_decay": [0.0, 0.01],
        "dropout": [0.1, 0.3],
        "label_smoothing": [0.0, 0.1],

        # input
        "max_length": [128, 256],

    }
}

output_dir = Path("/home/gpuhead-2/data_mining/DataMiningCourseProject/configs/bertweet")
output_dir.mkdir(parents=True, exist_ok=True)

def generate_combinations(space):
    keys = []
    values = []
    for category, params in space.items():
        for param, choices in params.items():
            keys.append((category, param))
            values.append(choices)
    
    for combo in product(*values):
        config = {"model": {}}
        for (cat, par), val in zip(keys, combo):
            config[cat][par] = val
        yield config

count = 0
for cfg in generate_combinations(search_space):
    file_name = f"config_{count:03d}.yaml"
    
    with open(output_dir / file_name, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, indent=4)
    count += 1

print(f"Generated {count} BERTweet configs in '{output_dir}/'")