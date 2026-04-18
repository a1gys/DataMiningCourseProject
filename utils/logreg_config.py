import yaml
import numpy as np
from itertools import product
from pathlib import Path

def is_valid_config(model_params):
    solver = model_params['solver']
    l1_ratio = model_params['l1_ratio']
    C = model_params['C']

    if solver == 'liblinear' and C == float('inf'):
        return False

    if l1_ratio == 1.0 and solver not in ['liblinear', 'saga']:
        return False

    if 0.0 < l1_ratio < 1.0 and solver != 'saga':
        return False

    return True

search_space = {
    "tf_idf": {
        "max_features": [5000, 10000],
        "ngram_range": [[1, 1], [1, 2]],
        "min_df": [2],
        "sublinear_tf": [True]
    },
    "model": {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear", "saga"],
        "l1_ratio": [0.0, 0.5, 1.0]
    }
}

output_dir = Path("/home/gpuhead-2/data_mining/DataMiningCourseProject/configs/logreg")
output_dir.mkdir(exist_ok=True)

def generate_combinations(space):
    keys = []
    values = []
    for category, params in space.items():
        for param, choices in params.items():
            keys.append((category, param))
            values.append(choices)
    
    for combo in product(*values):
        config = {"tf_idf": {}, "model": {}}
        for (cat, par), val in zip(keys, combo):
            config[cat][par] = val
        yield config

count = 0
for cfg in generate_combinations(search_space):
    if is_valid_config(cfg['model']):
        file_name = f"config_{count:03d}.yaml"
        
        with open(output_dir / file_name, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, indent=4)
        count += 1

print(f"Generated {count} scikit-learn configs in '{output_dir}/'")