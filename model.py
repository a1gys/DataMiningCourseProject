from enum import Enum

from box import Box
from sklearn.pipeline import Pipeline

from models.logreg import build_logreg
from models.bertweet import build_bertweet
from models.gemma import build_gemma_zero_shot
from models.zeroshot import build_deberta_zero_shot
from models.bilstm import build_bilstm

class Model(str, Enum):
    LOGREG = "logreg"
    BERTWEET = "bertweet"
    SVM = "svm"
    RNN = "rnn"
    GEMMA = "gemma"
    ZEROSHOT = "zeroshot"

    @staticmethod
    def validate_model(model_name: str):
        for elem in Model:
            if model_name == elem.value:
                return True
        return False


def _init_logreg(config: Box) -> Pipeline:
    return build_logreg(config=config)

def _init_bertweet(config: Box):
    return build_bertweet(config=config)

def _init_svm(config: Box):
    ...

def _init_rnn(config: Box):
    return build_bilstm(config=config)

def _init_gemma(config: Box):
    return build_gemma_zero_shot(config=config)

def _init_zeroshot(config: Box):
    return build_deberta_zero_shot(config=config)


def get_model(model_name: str,
              model_config: Box):
    # if not Model.validate_model(model_name):
    #     raise ValueError(f"{model_name} is not available model")
    try:
        model_type = Model(model_name)
    except ValueError:
        raise ValueError(f"'{model_name}' is not a valid model. Choices: {[m.value for m in Model]}")

    dispatch_table = {
        Model.LOGREG: _init_logreg,
        Model.BERTWEET: _init_bertweet,
        Model.SVM: _init_svm,
        Model.RNN: _init_rnn,
        Model.GEMMA: _init_gemma,
        Model.ZEROSHOT: _init_zeroshot,
    }

    constructor = dispatch_table[model_type]
    return constructor(model_config)


if __name__ == "__main__":
    import yaml

    config_path = "/home/gpuhead-2/data_mining/DataMiningCourseProject/configs/logreg/config1.yaml"
    with open(config_path) as file:
        config = Box(yaml.safe_load(file))
    res = get_model(model_name="logreg",
                    model_config=config)
    print(res)