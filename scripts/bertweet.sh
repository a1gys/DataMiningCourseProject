CONFIG_DIR="/home/gpuhead-2/data_mining/DataMiningCourseProject/configs/bertweet"
LOG_BASE_DIR="/home/gpuhead-2/data_mining/DataMiningCourseProject/logs/bertweet_partisan"
TRAIN_DATASET_PATH="/home/gpuhead-2/data_mining/DataMiningCourseProject/data/twitter_partisan/tweets_balanced_trainval.csv"
TEST_DATASET_PATH="/home/gpuhead-2/data_mining/DataMiningCourseProject/data/twitter_partisan/tweets_balanced_test.csv"
MODEL_NAME="bertweet"
CHECKPOINT_DIR="/home/gpuhead-2/data_mining/DataMiningCourseProject/checkpoints/bertweet_partisan"

NUM_CONFIGS=20   # <-- how many random configs to run
SEED=42          # for reproducibility

echo "Starting BERTweet experiment runner..."
echo "--------------------------------"

# Sample configs (reproducible)
SELECTED_CONFIGS=$(ls "$CONFIG_DIR"/*.yaml | shuf --random-source=<(yes $SEED) | head -n $NUM_CONFIGS)
echo "$SELECTED_CONFIGS" > selected_configs.txt
for config_file in $SELECTED_CONFIGS; do
    config_name=$(basename "$config_file" .yaml)

    echo "Running Experiment: $config_name"

    python main.py \
        --dataset "twitter_partisan" \
        --train_dataset_path "$TRAIN_DATASET_PATH" \
        --test_dataset_path "$TEST_DATASET_PATH" \
        --model_name "$MODEL_NAME" \
        --config_path "$config_file" \
        --train \
        --log_dir "$LOG_BASE_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --device "cuda"

    echo "Finished $config_name"
    echo "--------------------------------"
done

echo "Selected $NUM_CONFIGS configs out of total."
echo "All experiments completed!"