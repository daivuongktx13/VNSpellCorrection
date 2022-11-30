from util import train_validation_split, check_violated_cases
from vocab import Vocab
from noise import SynthesizeData
import os
import sys
import ray
import re
import time
from datetime import datetime as dt
sys.path.append("..")
import numpy as np
from params import PERCENT_NOISE, NUM_CPUS, NUM_PROCESSES
from utils.logger import get_logger
from viet_text_tools import normalize_diacritics

from transformers import AutoTokenizer
CHAR_TRANSFORMER_MAX_SEQ_LEN = 768
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word-base", use_fast=False)
logger = get_logger("./log/prepare_data.log")

@ray.remote
class PrepareActor(object):
    def __init__(self, id, lang, data_root='../data', corpus="binhbq") -> None:
        self.data_root, self.lang, self.corpus = data_root, lang, corpus
        self.id = id
        self.data_dir = f'{data_root}/{corpus}'

    def open_files(self):
        self.train_noise_file_name = f'{self.corpus}.train.noise'  + str(self.id)
        self.train_file_name =  f'{self.corpus}.train'  + str(self.id)
        self.train_onehot_file_name = f'{self.corpus}.onehot.train'  + str(self.id)
        self.train_length_file_name = f'{self.corpus}.length.train' + str(self.id)
        self.train_file_path = self.data_dir + '/' + self.train_file_name
        self.train_noise_file_path = self.data_dir + '/' + self.train_noise_file_name
        self.train_onehot_file_path = self.data_dir + '/' + self.train_onehot_file_name
        self.train_length_file_path = self.data_dir + '/' + self.train_length_file_name
        self.train_file = open(self.train_file_path, 'w', encoding='utf-8')
        self.train_noise_file = open(self.train_noise_file_path, 'w', encoding='utf-8')
        self.train_onehot_file = open(self.train_onehot_file_path, 'w', encoding='utf-8')
        self.train_length_file = open(self.train_length_file_path, 'w', encoding='utf-8')

        self.test_file_name =  f'{self.corpus}.test'  + str(self.id)
        self.test_noise_file_name =  f'{self.corpus}.test.noise'  + str(self.id)
        self.test_onehot_file_name = f'{self.corpus}.onehot.test'  + str(self.id)
        self.test_length_file_name = f'{self.corpus}.length.test'  + str(self.id)
        self.test_file_path = self.data_dir + '/' + self.test_file_name
        self.test_noise_file_path = self.data_dir + '/' + self.test_noise_file_name
        self.test_onehot_file_path = self.data_dir + '/' + self.test_onehot_file_name
        self.test_length_file_path = self.data_dir + '/' + self.test_length_file_name
        self.test_file = open(self.test_file_path, 'w', encoding='utf-8')
        self.test_noise_file = open(self.test_noise_file_path, 'w', encoding='utf-8')
        self.test_onehot_file = open(self.test_onehot_file_path, 'w', encoding='utf-8')
        self.test_length_file = open(self.test_length_file_path, 'w', encoding='utf-8')

    def close_files(self):
        if self.train_noise_file:
            self.train_noise_file.close()
        if self.test_noise_file:
            self.test_noise_file.close()
        if self.train_onehot_file:
            self.train_onehot_file.close()
        if self.test_onehot_file:
            self.test_onehot_file.close()


    def prepare_subword_sents_and_vocab(self, lines: ray.data.Dataset):

        vocab = Vocab(self.lang)
        self.subword_sents = []
        
        print(f"{dt.now()} PrepareActor[{self.id}].prepare_sublist_and_vocab() BEGIN...")

        for line in lines.iter_rows():
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
                    vocab.update_subword_freq(words)
                    splited_line = " ".join(words)
                    self.subword_sents.append(splited_line)
                continue
            ###
            if len(words) < 10:
                continue
            words = [normalize_diacritics(word) for word in words]
            line = " ".join(words)
            vocab.update_subword_freq(words)
            self.subword_sents.append(line)

        print(f"{dt.now()} PrepareActor[{self.id}].prepare_sublist_and_vocab() COMPLETED...")
        
        return vocab


    def gen_noised_and_onehot(self, noiser:SynthesizeData = None):
        print(f"{dt.now()} PrepareActor[{self.id}].gen_training_data() BEGIN...")
        self.open_files()
        logger = get_logger(f"log/prepare_data_worker{self.id}.log")
        assert noiser != None

        self.noiser = noiser
        np.random.seed(2001)
        np.random.shuffle(self.subword_sents)
        
        train_examples = 0
        max_train_examples = int(0.9 * len(self.subword_sents))

        for line in self.subword_sents:
            train_examples += 1

            if train_examples < max_train_examples:
                for_train = True
            else:
                for_train = False

            normal_noise, normal_onehot = self.noiser.add_normal_noise(
                line, percent_err=PERCENT_NOISE)

            split_merge_noise, split_merge_onehot = self.noiser.add_split_merge_noise(
                line, percent_err=PERCENT_NOISE, percent_normal_err=PERCENT_NOISE)

            la = len(normal_noise)
            lb = len(split_merge_noise)

            if la > (CHAR_TRANSFORMER_MAX_SEQ_LEN - 2):
                logger.log(f"INFO:  Noised longer than Transformer's max limit (NORMAL NOISE).")
                logger.log(f"TEXT: {normal_noise}")
                continue

            if lb > (CHAR_TRANSFORMER_MAX_SEQ_LEN - 2):
                logger.log(f"INFO:  Noised longer than Transformer's max limit (SPLIT MERGE NOISE).")
                logger.log(f"TEXT: {split_merge_noise}")
                continue

            if for_train:
                self.train_noise_file.write(normal_noise + '\n')
                self.train_noise_file.write(split_merge_noise + '\n')
                self.train_onehot_file.write(normal_onehot + '\n')
                self.train_onehot_file.write(split_merge_onehot + '\n')
                self.train_file.write(line + "\n")
                self.train_length_file.write(str(la) + "\n")
                self.train_length_file.write(str(lb) + "\n")   
            else:
                self.test_noise_file.write(normal_noise + '\n')
                self.test_noise_file.write(split_merge_noise + '\n')
                self.test_onehot_file.write(normal_onehot + '\n')
                self.test_onehot_file.write(split_merge_onehot + '\n')
                self.test_file.write(line + "\n")
                self.test_length_file.write(str(la) + "\n")
                self.test_length_file.write(str(lb) + "\n")   

        print(f"{dt.now()} PrepareActor[{self.id}].gen_training_data() COMPLETED...")
        self.close_files()


class PrepareDataset:

    def __init__(self, data_root='../data', lang='vi', corpus='binhvq'):
        self.data_root, self.lang, self.corpus = data_root, lang, corpus
        self.data_dir = f'{data_root}/{corpus}'

        self.vocab = Vocab(self.lang)
        
        # Number of CPUS
        self.MAX_CPUS = 12
        self.NUM_CPUS = NUM_CPUS if NUM_CPUS < self.MAX_CPUS else self.MAX_CPUS

        ray.init(num_cpus=NUM_CPUS)

        print(f"{dt.now()} PrepareDataset: Initiating {NUM_PROCESSES} PrepareActor")
        self.actors = [PrepareActor.remote(i, lang, self.data_root, self.corpus) for i in range(NUM_PROCESSES)]

        self.vocab_pickle_name = f'{self.corpus}.vocab.pkl'
        self.vocab_pickle_path = self.data_dir + '/' + self.vocab_pickle_name
        self.vocab_dict_name = f'{self.corpus}.dict.txt'
        self.vocab_dict_path = self.data_dir + '/' + self.vocab_dict_name     

    def build_vocab_and_subwords(self, ray_ds: ray.data.Dataset):

        print(f"{dt.now()} PrepareDataset.build_vocab_and_subwords()")

        shards = ray_ds.split(n = NUM_PROCESSES)

        subword_and_vocab_refs = [actor.prepare_subword_sents_and_vocab.remote(
            shard) for actor, shard in zip(self.actors, shards)]
        subwords_and_vocabs = ray.get(subword_and_vocab_refs)
        # Return results is vocab

        for i in range(NUM_PROCESSES):
            self.vocab.merge_sub_vocabs(subwords_and_vocabs[i]) 
            

    def build_noised_and_onehot(self):

        print(f"{dt.now()} PrepareDataset.build_noised_and_onehot.remote() BEGIN...")

        noiser = SynthesizeData(self.vocab)

        noised_and_onehot_refs = [actor.gen_noised_and_onehot.remote(noiser) \
            for actor in self.actors]
        
        _ = ray.get(noised_and_onehot_refs)

        print(f"{dt.now()} PrepareDataset.build_noised_and_onehot.remote() COMPLETE !!!")

        print(f"{dt.now()} PrepareDataset.build_noised_and_onehot(): Writing to noised and onehot files!!!")


    def prepare_data(self, in_file_name='vi_wiki.data.txt'):

        print(f"{dt.now()} PrepareDataset.prepare_data(): open_files()")

        self.in_file_path = self.data_dir + '/' + in_file_name

        if not os.path.exists(self.in_file_path):
            print(f"{dt.now()} PrepareDataset.prepare_data(): Cannot find input file!!!")
            print(f'File path: {self.in_file_path}')
            return

        print(f"{dt.now()} PrepareDataset.prepare_data(): Processing file part by part ...")

        with open(self.in_file_path, 'r', encoding='utf-8') as ifile:
            lines = ifile.readlines()
        
        ray_ds = ray.data.from_items(lines)
        del lines
        print(f"{dt.now()} PrepareDataset.prepare_data(): Building Vocabulary...")
        self.build_vocab_and_subwords(ray_ds)
        self.vocab.build_vocab(topk=100000)
        print(f"{dt.now()} PrepareDataset.prepare_data(): Writing Vocabulary to text file...")
        self.vocab.save_dict_text(self.vocab_dict_path)
        print(f"{dt.now()} PrepareDataset.prepare_data(): Writing Vocabulary to pickle file...")
        self.vocab.save_vocab_dict(self.vocab_pickle_path)
        print(f"{dt.now()} PrepareDataset.prepare_data(): Gen train noised and onehot...")
        self.build_noised_and_onehot()
        print(f"{dt.now()} PrepareDataset - Complete preparing dataset!!!")


if __name__ == "__main__":
    import argparse
    description = '''
        prepare_dataset.py:

        Usage: python prepare_dataset.py --dataset vi_wiki --file vi_wiki.data.txt --test False
                    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--file', type=str, default='corpus-small.txt')
    parser.add_argument('--corpus', type=str, default='binhvq')
    parser.add_argument('--data_root', type=str, default="../data")
    args = parser.parse_args()
    creater = PrepareDataset(data_root = args.data_root, corpus=args.corpus)
    start_time = time.time()
    creater.prepare_data(args.file)
    end_time = time.time()
    print(f"Time consumed for generate data: {end_time - start_time}")
