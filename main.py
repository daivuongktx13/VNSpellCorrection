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
import re
from dataset.noise import SynthesizeData


model_name = "tfmwtr"
dataset = "binhvq"
vocab_path = f'data/{dataset}/{dataset}.vocab.pkl'
weight_path = f'data/checkpoints/tfmwtr/{dataset}.weights.pth'
vocab = Vocab("vi")
vocab.load_vocab_dict(vocab_path)
noiser = SynthesizeData(vocab)
model_wrapper = ModelWrapper(f"{model_name}", vocab)
corrector = Corrector(model_wrapper)
load_weights(corrector.model, weight_path)

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/spelling/")
def correct(string: Union[str, None] = None):
    out = corrector.correct_transfomer_with_tr(string, num_beams=1)
    t = re.sub(r"(\s*)([.,:?!;]{1})(\s*)", r"\2\3", out['predict_text'][0])
    t = re.sub(r"((?P<parenthesis>\()\s)", r"\g<parenthesis>", t)
    t = re.sub(r"(\s(?P<parenthesis>\)))", r"\g<parenthesis>", t)
    t = re.sub(r"((?P<bracket>\[)\s)", r"\g<bracket>", t)
    t = re.sub(r"(\s(?P<bracket>\]))", r"\g<bracket>", t)
    t = re.sub(r"([\'\"])\s(.*)\s([\'\"])", r"\1\2\3", t)
    out['predict_text']= re.sub(r"\s(%)", "%", t)
    return out
    
@app.get("/splme_noise")
def split_merge_noise(string: Union[str, None] = None):
    text = " ".join(re.findall("\w+|[^\w\s]{1}", string))
    noised_text, onehot_label = noiser.add_split_merge_noise(text, percent_err=0.3, percent_normal_err=0.3)
    return noised_text

@app.get("/norm_noise")
def normal_noise(string: Union[str, None] = None):
    text = " ".join(re.findall("\w+|[^\w\s]{1}", string))
    noised_text, onehot_label = noiser.add_normal_noise(text, percent_err=0.3)
    return noised_text