import pandas as pd

from box import Box

from model import Model
from models.logreg import train_logreg, test_logreg
from models.bertweet import train_bertweet, test_bertweet
from models.gemma import inference_gemma
from models.zeroshot import inference_deberta
from models.bilstm import train_bilstm, test_bilstm

def train(model,
          model_name: str,
          config: Box,
          dataset: pd.DataFrame):
    model_type = Model(model_name)

    dispatch_table = {
        Model.LOGREG: train_logreg,
        Model.BERTWEET: train_bertweet,
        # Model.SVM: train_svm,
        Model.RNN: train_bilstm,
    }

    constructor = dispatch_table[model_type]
    return constructor(model, config, dataset)

def test(model_name: str,
         config: Box,
         dataset: pd.DataFrame):
    model_type = Model(model_name)

    dispatch_table = {
        Model.LOGREG: test_logreg,
        Model.BERTWEET: test_bertweet,
        # Model.SVM: test_svm,
        Model.RNN: test_bilstm,
        Model.GEMMA: inference_gemma,
        Model.ZEROSHOT: inference_deberta
    }

    constructor = dispatch_table[model_type]
    return constructor(config, dataset)