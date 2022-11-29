from vocab import Vocab
from datetime import datetime as dt
from tqdm import tqdm
import re
from viet_text_tools import normalize_diacritics
from util import train_validation_split, check_violated_cases
from utils.logger import get_logger
import time
from params import *
import ray
import os
from dataset.noise import SynthesizeData
import numpy as np
from transformers import AutoTokenizer
PHOBERT_MAX_SEQ_LEN = 256
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

class DatasetGenerator:

    def __init__(self, id, lang, noiser, data_root="../data", corpus="vi_wiki"):

        self.id = id
        self.lang = lang
        self.noiser = noiser
        self.data_root, self.lang, self.corpus = data_root, lang, corpus
        self.data_dir = f'{data_root}/{corpus}'
        self.vocab = Vocab(self.lang)

    def open_files(self):
        self.train_noise_file_name = f'{self.corpus}.train.noise'
        self.train_file_name =  f'{self.corpus}.train'
        self.train_onehot_file_name = f'{self.corpus}.onehot.train'
        self.train_length_file_name = f'{self.corpus}.length.train'
        self.train_file_path = self.data_dir + '/' + self.train_file_name
        self.train_noise_file_path = self.data_dir + '/' + self.train_noise_file_name
        self.train_onehot_file_path = self.data_dir + '/' + self.train_onehot_file_name
        self.train_length_file_path = self.data_dir + '/' + self.train_length_file_name
        self.train_file = open(self.train_file_path, 'w', encoding='utf-8')
        self.train_noise_file = open(self.train_noise_file_path, 'w', encoding='utf-8')
        self.train_onehot_file = open(self.train_onehot_file_path, 'w', encoding='utf-8')
        self.train_length_file = open(self.train_length_file_path, 'w', encoding='utf-8')

        self.test_file_name =  f'{self.corpus}.test'
        self.test_noise_file_name =  f'{self.corpus}.test.noise'
        self.test_onehot_file_name = f'{self.corpus}.onehot.test'
        self.test_length_file_name = f'{self.corpus}.length.test'
        self.test_file_path = self.data_dir + '/' + self.test_file_name
        self.test_noise_file_path = self.data_dir + '/' + self.test_noise_file_name
        self.test_onehot_file_path = self.data_dir + '/' + self.test_onehot_file_name
        self.test_length_file_path = self.data_dir + '/' + self.test_length_file_name
        self.test_file = open(self.test_file_path, 'w', encoding='utf-8')
        self.test_noise_file = open(self.test_noise_file_path, 'w', encoding='utf-8')
        self.test_onehot_file = open(self.test_onehot_file_path, 'w', encoding='utf-8')
        self.test_length_file = open(self.test_length_file_path, 'w', encoding='utf-8')

        self.vocab_pickle_name =  f'{self.corpus}.vocab.pkl'
        self.vocab_pickle_path = self.data_dir + '/' + self.vocab_pickle_name
        self.vocab_dict_name =  f'{self.corpus}.dict.txt'
        self.vocab_dict_path = self.data_dir + '/' + self.vocab_dict_name

    def close_files(self):
        if self.train_file:
            self.train_file.close()
        if self.test_file:
            self.test_file.close()
        if self.train_noise_file:
            self.train_noise_file.close()
        if self.test_noise_file:
            self.test_noise_file.close()
        if self.train_onehot_file:
            self.train_onehot_file.close()
        if self.test_onehot_file:
            self.test_onehot_file.close()

    def build_data(self):

        with open(self.in_file_path, 'r', encoding='utf-8') as ifile:
            dataset = ifile.readlines()

        self.open_files()

        if not os.path.exists(self.in_file_path):
            print(f"{dt.now()} PrepareDataset.build_data(): Cannot find input file!!!")
            print(f'File path: {self.in_file_path}')
            return

        print(f"{dt.now()} PrepareDataset.build_vocab_and_subwords()")
        subword_sents = []
        logger = get_logger(f"log/prepare_data_worker.log")

        for line in tqdm(dataset):
            line = line.strip("\n")
            words = line.split(" ")
            ###
            if len(words) > 200:
                splited_lines = re.split("[.;]+", line)
                for splited_line in splited_lines:
                    words = splited_line.split(" ")
                    if len(words) < 10 or len(words) > 200:
                        continue
                    words = [normalize_diacritics(word) for word in words]
                    splited_line = " ".join(words)
                    self.vocab.update_subword_freq(words)
                    subword_sents.append(splited_line)
                continue
            ###
            if len(words) < 10:
                continue
            words = [normalize_diacritics(word) for word in words]
            line = " ".join(words)
            self.vocab.update_subword_freq(words)
            subword_sents.append(line)

        print(f"{dt.now()} PrepareDataset.build_data(): Building Vocabulary...")
        self.vocab.build_vocab(topk=100000)
        print(f"{dt.now()} PrepareDataset.build_data(): Writing Vocabulary to text file...")
        self.vocab.save_dict_text(self.vocab_dict_path)
        print(f"{dt.now()} PrepareDataset.build_data(): Writing Vocabulary to pickle file...")
        self.vocab.save_vocab_dict(self.vocab_pickle_path)
        print(f"{dt.now()} PrepareDataset.build_data(): Make noise model...")
        self.noiser = SynthesizeData(self.vocab)

        np.random.shuffle(subword_sents)
        
        train_examples = 0
        max_train_examples = int(0.9 * len(subword_sents))

        for line in tqdm(subword_sents):
            train_examples += 1
            if train_examples < max_train_examples:
                for_train = True
            else:
                for_train = False
            normal_noise, normal_onehot = self.noiser.add_normal_noise(
                line, percent_err=PERCENT_NOISE)

            split_merge_noise, split_merge_onehot = self.noiser.add_split_merge_noise(
                line, percent_err=PERCENT_NOISE)

            tokens = tokenizer.tokenize(normal_noise)
            la = len(tokens)
            if len(tokens) > (PHOBERT_MAX_SEQ_LEN - 2):
                logger.log(f"INFO: BERT noised tokens larger than BERT's max limit (NORMAL NOISE).")
                logger.log(f"NOISED: {tokens}")
                logger.log(f"TEXT: {normal_noise}")
                continue
            
            if check_violated_cases(tokens, [int(x) for x in normal_onehot.split(" ")]):
                logger.log("INFO: BERT SUBWORD ERROR IN NORMAL NOISE!!!")
                logger.log(f"text: {normal_noise}")
                logger.log(f"noised tokens: {tokens}")
                logger.log(f"onehot: {normal_onehot}")
                continue
                
            tokens = tokenizer.tokenize(split_merge_noise)
            lb = len(tokens)

            if len(tokens) > (PHOBERT_MAX_SEQ_LEN - 2):
                logger.log(f"INFO: BERT noised tokens larger than BERT's max limit (SPLIT MERGE NOISE).")
                logger.log(f"NOISED: {tokens}")
                continue
            
            if check_violated_cases(tokens, [int(x) for x in split_merge_onehot.split(" ")]):
                logger.log("INFO: BERT SUBWORD ERROR IN SPLIT MERGE ERROR!!!")
                logger.log(f"text: {split_merge_noise}")
                logger.log(f"noised tokens: {tokens}")
                logger.log(f"onehot: {split_merge_onehot}")
                continue
            
            if for_train:
                self.train_noise_file.write(normal_noise + '\n')
                self.train_noise_file.write(split_merge_noise + '\n')
                self.train_onehot_file.write(normal_onehot + '\n')
                self.train_onehot_file.write(split_merge_onehot + '\n')
                self.train_file.write(line)
                self.train_length_file.write(str(la) + "\n")
                self.train_length_file.write(str(lb) + "\n")     
            else:
                self.test_noise_file.write(normal_noise + '\n')
                self.test_noise_file.write(split_merge_noise + '\n')
                self.test_onehot_file.write(normal_onehot + '\n')
                self.test_onehot_file.write(split_merge_onehot + '\n')
                self.test_file.write(line)
                self.test_length_file.write(str(la) + "\n")
                self.test_length_file.write(str(lb) + "\n")   

            self.close_files() 

    
if __name__ == "__main__":
    import argparse
    description = '''
        prepare_dataset_v3.py:

        Usage: python prepare_dataset.py --dataset vi_wiki --file vi_wiki.data.txt --test False
                    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--corpus', type=str, default="vi_wiki")
    parser.add_argument('--file', type=str, default='corpus-small.txt')
    args = parser.parse_args()
    creater = DatasetGenerator(data_root=args.data_root, corpus=args.corpus, file_name=args.file)
    start_time = time.time()
    creater.build_data()
    end_time = time.time()
    print(f"Time consumed for generate data: {end_time - start_time}")