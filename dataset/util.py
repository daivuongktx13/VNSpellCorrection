import os
import numpy as np
from tqdm import tqdm

def load_vsec_dataset(base_path, corr_file, incorr_file):
    # load files
    if base_path:
        assert os.path.exists(base_path) == True
    incorr_data = []
    opfile1 = open(os.path.join(base_path, incorr_file), "r", encoding="utf-8")
    for line in opfile1:
        if line.strip() != "":
            incorr_data.append(line.strip())
    opfile1.close()
    corr_data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r", encoding="utf-8")
    for line in opfile2:
        if line.strip() != "":
            corr_data.append(line.strip())
    opfile2.close()

    assert len(incorr_data) == len(corr_data)

    data = []
    for x, y in zip(incorr_data, corr_data):
        data.append((x, y))

    print(f"loaded tuples of (incorr, corr) examples from {base_path}")
    return data

def load_dataset(base_path, corr_file, incorr_file, length_file = None):
    # load files
    if base_path:
        assert os.path.exists(base_path) == True
    incorr_data = []
    opfile1 = open(os.path.join(base_path, incorr_file), "r", encoding="utf-8")
    for line in tqdm(opfile1):
        if line.strip() != "":
            incorr_data.append(line.strip())
    opfile1.close()
    corr_data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r", encoding="utf-8")
    for line in tqdm(opfile2):
        if line.strip() != "":
            corr_data.append(line.strip())
            corr_data.append(line.strip())
    opfile2.close()

    length_data = []
    opfile4 = open(os.path.join(base_path, length_file), "r", encoding="utf-8")
    for line in tqdm(opfile4):
        if line.strip() != "":
            length_data.append(int(line))
    opfile4.close()

    assert len(incorr_data) == len(corr_data) == len(length_data)
 
    data = []
    for x, y, z in zip(incorr_data, corr_data, length_data):
        data.append((x, y, z))

    print(f"loaded tuples of (incorr, corr, length) examples from {base_path}")
    return data

def train_validation_split(data, train_ratio, seed):
    np.random.seed(seed)
    len_ = len(data)
    train_len_ = int(np.ceil(train_ratio * len_))
    inds_shuffled = np.arange(len_)
    np.random.shuffle(inds_shuffled)
    train_data = []
    for ind in inds_shuffled[:train_len_]:
        train_data.append(data[ind])
    validation_data = []
    for ind in inds_shuffled[train_len_:]:
        validation_data.append(data[ind])
    return train_data, validation_data



