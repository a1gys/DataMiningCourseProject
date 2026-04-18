CONFIG_DIR="/home/gpuhead-2/data_mining/DataMiningCourseProject/configs/logreg"
LOG_BASE_DIR="/home/gpuhead-2/data_mining/DataMiningCourseProject/logs/logreg"
TRAIN_DATASET_PATH="/home/gpuhead-2/data_mining/DataMiningCourseProject/data/twitter_partisan/tweets_balanced_trainval.csv"
TEST_DATASET_PATH="/home/gpuhead-2/data_mining/DataMiningCourseProject/data/twitter_partisan/tweets_balanced_trainval.csv"
MODEL_NAME="logreg"
CHECKPOINT_DIR="/home/gpuhead-2/data_mining/DataMiningCourseProject/checkpoints/logreg"

echo "Starting experiment runner..."
echo "--------------------------------"

for config_file in "$CONFIG_DIR"/*.yaml; do
    config_name=$(basename "$config_file" .yaml)

    echo "Running Experiment: $config_name"

    python main.py \
        --dataset "twitter_partisan" \
        --train_dataset_path "$TRAIN_DATASET_PATH" \
        --test_dataset_path "$TEST_DATASET_PATH" \
        --model_name "$MODEL_NAME" \
        --config_path "$config_file" \
        --train True \
        --test True \
        --log_dir "$LOG_BASE_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR"

    echo "Finished $config_name. Results saved to $current_log_dir"
    echo "--------------------------------"
done

echo "All experiments completed!"