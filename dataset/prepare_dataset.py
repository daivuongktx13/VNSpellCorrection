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
from params import PERCENT_NOISE, NUM_CPUS, NUM_PROCESSES
from utils.logger import get_logger
from viet_text_tools import normalize_diacritics


from transformers import AutoTokenizer
PHOBERT_MAX_SEQ_LEN = 256
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
logger = get_logger("./log/prepare_data.log")

@ray.remote
class PrepareActor(object):
    def __init__(self, id, lang) -> None:
        self.id = id
        self.lang = lang
        self.noiser = None
        self.train_list = None
        self.test_list = None
        self.vocab = None

    def prepare_subword_sents_and_vocab(self, lines: ray.data.Dataset):

        vocab = Vocab(self.lang)
        subword_sents = []
        
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
                    subword_sents.append(splited_line)
                continue
            ###
            if len(words) < 10:
                continue
            words = [normalize_diacritics(word) for word in words]
            vocab.update_subword_freq(words)
            subword_sents.append(line)

        
        print(f"{dt.now()} PrepareActor[{self.id}].prepare_sublist_and_vocab() COMPLETED...")

        train_list, test_list \
            = train_validation_split(subword_sents, 0.9, seed=11690)
        
        return train_list, test_list, vocab


    def gen_noised_and_onehot(self, lines: ray.data.Dataset, vocab: Vocab):
        print(f"{dt.now()} PrepareActor[{self.id}].gen_training_data() BEGIN...")

        logger = get_logger(f"log/prepare_data_worker{self.id}.log")

        if self.vocab == None:
            self.vocab = vocab
            self.noiser = SynthesizeData(vocab)

        normal_list = []
        noised_list = []
        onehot_list = []
        length_list = []

        for line in lines.iter_rows():
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


            ## Should perform check_violated_cases in tokenizer here
            normal_list.append(line)
            noised_list.extend([normal_noise, split_merge_noise])
            onehot_list.extend([normal_onehot, split_merge_onehot])
            length_list.extend([la, lb])

        print(f"{dt.now()} PrepareActor[{self.id}].gen_training_data() COMPLETED...")

        return noised_list, onehot_list, normal_list, length_list

class PrepareDataset:

    def __init__(self, data_root='../data', lang='vi', corpus='vi_wiki'):
        self.data_root, self.lang, self.corpus = data_root, lang, corpus
        self.data_dir = f'{data_root}/{corpus}'

        self.vocab = Vocab(self.lang)
        
        # Number of CPUS
        self.MAX_CPUS = 8
        self.NUM_CPUS = NUM_CPUS if NUM_CPUS < self.MAX_CPUS else self.MAX_CPUS

        ray.init(num_cpus=NUM_CPUS)

        print(f"{dt.now()} PrepareDataset: Initiating {NUM_PROCESSES} PrepareActor")
        self.actors = [PrepareActor.remote(i, lang) for i in range(NUM_PROCESSES)]

    def open_files(self, test=False):
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

        self.vocab_pickle_name = f'{self.corpus}.vocab.test.pkl' if test else f'{self.corpus}.vocab.pkl'
        self.vocab_pickle_path = self.data_dir + '/' + self.vocab_pickle_name
        self.vocab_dict_name = f'{self.corpus}.dict.test.txt' if test else f'{self.corpus}.dict.txt'
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

    def build_vocab_and_subwords(self, ray_ds: ray.data.Dataset):

        print(f"{dt.now()} PrepareDataset.build_vocab_and_subwords()")

        shards = ray_ds.split(n = NUM_PROCESSES)

        subword_and_vocab_refs = [actor.prepare_subword_sents_and_vocab.remote(
            shard) for actor, shard in zip(self.actors, shards)]
        subwords_and_vocabs = ray.get(subword_and_vocab_refs)
        # Return results is tuple of train, test, vocab corresponding to 0, 1, 2

        for i in range(NUM_PROCESSES):
            self.vocab.merge_sub_vocabs(subwords_and_vocabs[i][2]) 
            
            for sent in subwords_and_vocabs[i][0]:
                self.train_file.write(sent + '\n')

            for sent in subwords_and_vocabs[i][1]:
                self.test_file.write(sent + '\n')

    def build_noised_and_onehot(self, ray_ds: ray.data.Dataset, train=True):

        print(f"{dt.now()} PrepareDataset.build_noised_and_onehot.remote() BEGIN...")

        shards = ray_ds.split(n = NUM_PROCESSES)

        noised_and_onehot_refs = [actor.gen_noised_and_onehot.remote(shard, self.vocab) \
            for actor, shard in zip(self.actors, shards)]

        noised_and_onehot = ray.get(noised_and_onehot_refs)

        print(f"{dt.now()} PrepareDataset.build_noised_and_onehot.remote() COMPLETE !!!")

        print(f"{dt.now()} PrepareDataset.build_noised_and_onehot(): Writing to noised and onehot files!!!")

        if train:
            self.train_file.truncate()
        else:
            self.test_file.truncate()

        for i in range(len(noised_and_onehot)):

            print(f"{dt.now()} PrepareDataset.prepare_data(): training_data[{i}] Writing to files!!! BEGIN... !!!")

            for sent in noised_and_onehot[i][0]:
                if train:
                    self.train_noise_file.write(sent + '\n')
                else:
                    self.test_noise_file.write(sent + '\n')

            for sent in noised_and_onehot[i][1]:
                if train:
                    self.train_onehot_file.write(sent + '\n')
                else:
                    self.test_onehot_file.write(sent + '\n')

            for sent in noised_and_onehot[i][2]:
                if train:
                    self.train_file.write(sent)
                else:
                    self.test_file.write(sent)
            
            for sent in noised_and_onehot[i][3]:
                if train:
                    self.train_length_file.write(str(sent) + "\n")
                else:
                    self.test_length_file.write(str(sent) + "\n")

            print(f"{dt.now()} PrepareDataset.prepare_data(): training_data[{i}] Writing to files!!! COMPLETE !!!")

        del noised_and_onehot
        del noised_and_onehot_refs

    def prepare_data(self, in_file_name='vi_wiki.data.txt', test=False):

        print(f"{dt.now()} PrepareDataset.prepare_data(): open_files()")

        self.open_files(test=test)

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
        self.build_vocab_and_subwords(ray_ds)

        
        print(f"{dt.now()} PrepareDataset.prepare_data(): Building Vocabulary...")
        self.vocab.build_vocab(topk=100000)
        print(f"{dt.now()} PrepareDataset.prepare_data(): Writing Vocabulary to text file...")
        self.vocab.save_dict_text(self.vocab_dict_path)
        print(f"{dt.now()} PrepareDataset.prepare_data(): Writing Vocabulary to pickle file...")
        self.vocab.save_vocab_dict(self.vocab_pickle_path)

        self.train_file.close()
        
        print(f"{dt.now()} PrepareDataset.prepare_data(): Gen train noised and onehot...")
        self.train_file = open(self.train_file_path, 'r', encoding = "utf-8")
        lines = self.train_file.readlines()
        self.train_file.close()

        self.train_file = open(self.train_file_path, 'w+', encoding = 'utf-8')
        ray_ds = ray.data.from_items(lines)
        del lines
        self.build_noised_and_onehot(ray_ds)

        self.test_file.close()
        print(f"{dt.now()} PrepareDataset.prepare_data(): Gen test noised and onehot...")
        self.test_file = open(self.test_file_path, 'r', encoding = "utf-8")
        lines = self.test_file.readlines()
        self.test_file.close()

        self.test_file = open(self.test_file_path, 'w+', encoding = "utf-8")
        ray_ds = ray.data.from_items(lines)
        del lines
        self.build_noised_and_onehot(ray_ds, train = False)


        print(f"{dt.now()} PrepareDataset.prepare_data(): close_file()")
        self.close_files()
        print(f"{dt.now()} PrepareDataset - Complete preparing dataset!!!")


if __name__ == "__main__":
    import argparse
    description = '''
        prepare_dataset.py:

        Usage: python prepare_dataset.py --dataset vi_wiki --file vi_wiki.data.txt --test False
                    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--file', type=str, default='corpus-small.txt')
    parser.add_argument('--dataset', type=str, default='vi_wiki')
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()
    creater = PrepareDataset(corpus=args.dataset)
    start_time = time.time()
    creater.prepare_data(args.file, args.test)
    end_time = time.time()
    print(f"Time consumed for generate data: {end_time - start_time}")
