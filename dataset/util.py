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
    
    data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r", encoding="utf-8")
    for line in tqdm(opfile2):
        if line.strip() != "":
            data.append([line.strip()])
            data.append([line.strip()])
    opfile2.close()

    opfile1 = open(os.path.join(base_path, incorr_file), "r", encoding="utf-8")
    for i, line in tqdm(enumerate(opfile1)):
        if line.strip() != "":
            data[i].append(line.strip())
    opfile1.close()

    opfile4 = open(os.path.join(base_path, length_file), "r", encoding="utf-8")
    for i, line in tqdm(enumerate(opfile4)):
        if line.strip() != "":
            data[i].append(int(line))
    opfile4.close()

    print(f"loaded tuples of (incorr, corr, length) examples from {base_path}")
    return data

def load_epoch_dataset(base_path, corr_file, incorr_file, length_file, epoch: int, num_epoch: int):
    # load files
    if base_path:
        assert os.path.exists(base_path) == True
    assert num_epoch >= 1
    assert epoch >= 1 and epoch <= num_epoch

    ## Count number of data
    opfile = open(os.path.join(base_path, length_file), "r", encoding="utf-8")
    count = 0
    for i, line in tqdm(enumerate(opfile)):
        count +=1
    opfile.close()
    print(f"Number of training datas: {count} examples!")
    
    epochdataset_examples = 1 / num_epoch * count
    start_index = epochdataset_examples * (epoch - 1)
    end_index = start_index + epochdataset_examples

    data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r", encoding="utf-8")
    traverse_count = 0
    for i, line in tqdm(enumerate(opfile2)):

        if line.strip() != "":
            if traverse_count >= start_index and traverse_count < end_index :
                data.append([line.strip()])
            elif traverse_count >= end_index:
                break
            else:
                traverse_count += 1

            if traverse_count >= start_index and traverse_count < end_index :
                data.append([line.strip()])
            elif traverse_count >= end_index:
                break
            else:
                traverse_count += 1

    opfile2.close()
    opfile1 = open(os.path.join(base_path, incorr_file), "r", encoding="utf-8")
    traverse_count = 0
    for i, line in tqdm(enumerate(opfile1)):
        if line.strip() != "":
            if traverse_count >= start_index and traverse_count < end_index :
                data[i].append(line.strip())
            elif traverse_count >= end_index:
                break
        traverse_count += 1
    opfile1.close()
    traverse_count = 0
    opfile4 = open(os.path.join(base_path, length_file), "r", encoding="utf-8")
    for i, line in tqdm(enumerate(opfile4)):
        if line.strip() != "":
            if traverse_count >= start_index and traverse_count < end_index :
                data[i].append(int(line))
            elif traverse_count >= end_index:
                break
        traverse_count += 1
    opfile4.close()

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



