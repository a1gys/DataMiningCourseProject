import pandas as pd

def get_dataset(dataset: str,
                dataset_path: str) -> pd.DataFrame:
    assert dataset in ["twitter_partisan", "mbib"]

    if dataset == "twitter_partisan":
        df = get_partisan_twitter_dataset(dataset_path)
    
    elif dataset == "mbib":
        df = get_mbib_dataset(dataset_path)
    
    return df

def get_mbib_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)

    columns = ["text", "label"]
    return df[columns]

def get_partisan_twitter_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = (df["party"].str.strip().str.upper() == "R").astype(int)

    columns = ["text", "party", "source", "label"]
    return df[columns]

if __name__ == "__main__":
    # dataset = "twitter_partisan"
    # dataset_path = "/home/gpuhead-2/data_mining/DataMiningCourseProject/data/twitter_partisan/tweets_balanced_trainval.csv"
    dataset = "mbib"
    dataset_path = "/home/gpuhead-2/data_mining/DataMiningCourseProject/data/mbib/mbib_balanced_test.csv"

    df = get_dataset(dataset=dataset,
                     dataset_path=dataset_path)

    print(df.head())