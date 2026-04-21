import yaml
from itertools import product
from pathlib import Path


def is_valid_config(model_params: dict, embed_params: dict) -> bool:
    # single-layer LSTM has no inter-layer dropout — dropout must be 0
    if model_params["num_layers"] == 1 and model_params["dropout"] > 0.0:
        return False

    # GloVe 50d / 100d only make sense with a matching hidden_dim ceiling
    # (avoid hidden_dim >> embed_dim, which wastes capacity)
    if embed_params["embed_dim"] == 50 and model_params["hidden_dim"] > 128:
        return False

    # freeze_embed=True with random init (no GloVe) is pointless —
    # caller controls whether GloVe is loaded, but flag still shouldn't
    # be True when embed_dim=50 (we only ship 200d Twitter GloVe by default)
    if embed_params["freeze_embed"] and embed_params["embed_dim"] != 200:
        return False

    return True


search_space = {
    "embed": {
        "embed_dim":    [200],          # matches glove.twitter.27B.200d
        "max_len":      [64],       # token budget per tweet
        "vocab_size":   [60_000],
        "freeze_embed": [False],  # fine-tune vs frozen GloVe
    },
    "model": {
        "hidden_dim": [64, 128, 256],
        "num_layers": [1, 2],
        "dropout":    [0.2, 0.3, 0.5],
        "num_classes": [2]
    },
    "training": {
        "batch_size": [128],
        "lr":         [1e-3, 5e-4],
        "patience":   [3],              # fixed — not worth sweeping
    },
}

output_dir = Path("configs/bilstm")
output_dir.mkdir(parents=True, exist_ok=True)


def generate_combinations(space: dict):
    keys, values = [], []
    for category, params in space.items():
        for param, choices in params.items():
            keys.append((category, param))
            values.append(choices)

    for combo in product(*values):
        config = {cat: {} for cat in space}
        for (cat, par), val in zip(keys, combo):
            config[cat][par] = val
        yield config


count = 0
for cfg in generate_combinations(search_space):
    if is_valid_config(cfg["model"], cfg["embed"]):
        file_name = f"config_{count:03d}.yaml"
        with open(output_dir / file_name, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, indent=4)
        count += 1

print(f"Generated {count} BiLSTM configs in '{output_dir}/'")