import argparse
import yaml

from box import Box

from data import get_dataset
from model import get_model
from train import train, test

def load_model_config(config_path) -> Box:
    with open(config_path) as file:
        model_config = yaml.safe_load(file)
    return Box(model_config)

def main(args: Box):
    config = load_model_config(args.config_path)
    
    model = get_model(model_name=args.model_name,
                      model_config=config)

    config.dataset = args.dataset
    config.log_dir = args.log_dir
    config.config_path = args.config_path
    config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device
    
    if args.train and args.model_name not in ["gemma", "zeroshot"]:
        train_dataset = get_dataset(dataset=args.dataset,
                                    dataset_path=args.train_dataset_path)
        train(model=model,
              model_name=args.model_name,
              config=config,
              dataset=train_dataset)
    
    if args.test:
        test_dataset = get_dataset(dataset=args.dataset,
                                   dataset_path=args.test_dataset_path)
        test(model_name=args.model_name,
             config=config,
             dataset=test_dataset)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Political Party Classification")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="twitter_partisan", choices=["twitter_partisan", "mbib"])
    parser.add_argument("--train_dataset_path", type=str)
    parser.add_argument("--test_dataset_path", type=str)

    # Model arguments
    parser.add_argument("--model_name", type=str, choices=["bertweet", "logreg", "svm", "rnn", "gemma", "zeroshot"])
    parser.add_argument("--config_path", type=str)

    # Training arguments
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)

    args = Box(vars(parser.parse_args()))
    main(args=args)