import numpy as np
import sys
sys.path.append("../..")
from dataset.vocab import Vocab
import torch
import os

def batch_accuracy_func(batch_predictions: np.ndarray,
                            batch_targets: np.ndarray,
                            batch_lengths: list):
    """
    given the predicted word idxs, this method computes the accuracy 
    by matching all values from 0 index to batch_lengths_ index along each 
    batch example
    """
    assert len(batch_predictions) == len(
        batch_targets) == len(batch_lengths)
    count_ = 0
    total_ = 0
    for pred, targ, len_ in zip(batch_predictions, batch_targets, batch_lengths):
        count_ += (pred[:len_] == targ[:len_]).sum()
        total_ += len_
    return count_, total_


def load_weights(model, filename, neuspell=False):
    if not os.path.exists(filename):
        print("pt_til.py - Cannot find weights path !!!")
        print(f'Path: {filename}')
        return

    state_dict = torch.load(filename, map_location=torch.device('cpu'))

    if not neuspell:
        state_dict = state_dict["state_dict"]

    for name, param in model.named_parameters():
        if name not in state_dict:
            print('{} not found'.format(name))
        elif state_dict[name].shape != param.shape:
            print(
                '{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
            del state_dict[name]

    model.load_state_dict(state_dict, strict=False)