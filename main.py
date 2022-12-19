from typing import Union

from fastapi import FastAPI
import sys
sys.path.append("..")
import os
from params import *
from dataset.vocab import Vocab
from models.corrector import Corrector
from models.model import ModelWrapper
from models.util import load_weights
import torch.nn.functional as F
import torch
import numpy as np


model_name = "tfmwtr"
dataset = "binhvq"
vocab_path = f'data/{dataset}/{dataset}.vocab.pkl'
weight_path = f'data/checkpoints/tfmwtr/{dataset}.weights.pth'
vocab = Vocab("vi")
vocab.load_vocab_dict(vocab_path)
model_wrapper = ModelWrapper(f"{model_name}", vocab)
corrector = Corrector(model_wrapper)
load_weights(corrector.model, weight_path)

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/spelling/")
def read_string(string: Union[str, None] = None):
    out = corrector.correct_transfomer_with_tr(string, num_beams=1)
    return {"string": out}