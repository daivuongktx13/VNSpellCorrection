from __future__ import annotations
import pickle
import re
import os
import sys


sys.path.append("..")

from params import *

class Vocab():
    def __init__(self, lang='vi'):
        self.not_alphabet_regex = '''[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ ]'''
        self.lang = lang
        self.token_freq_list = []
        self.token_freq, self.token2idx, self.idx2token = {}, {}, {}
        self.pad_token = "<<PAD>>"
        self.unk_token = "<<UNK>>"
        self.sub_token = "<<SUB>>"
        self.eos_token = "<<EOS>>"

        self.chartoken2idx, self.idx2chartoken = {}, {}
        self.char_unk_token, self.char_pad_token, self.char_start_token, self.char_end_token = \
            "<<CHAR_UNK>>", "<<CHAR_PAD>>", "<<CHAR_START>>", "<<CHAR_END>>"
        self.char_space_token = "<<CHAR_SPACE>>"

    def set_lang(self, lang):
        self.lang = lang

    def exist(self, word):

        return word in self.token2idx

    def update_subword_freq(self, subwords: list):
        for subword in subwords:
            if not subword.isdigit():
                if re.search(self.not_alphabet_regex, subword):
                    continue
                if subword not in self.token_freq:
                    self.token_freq[subword] = 0
                self.token_freq[subword] += 1

    def merge_sub_vocabs(self, vocab: Vocab):
        for subword in vocab.token_freq:
            if subword not in self.token_freq:
                self.token_freq[subword] = 0
            self.token_freq[subword] += vocab.token_freq[subword]

    def insert_special_tokens(self):
        # add <<PAD>> special token
        self.pad_token_idx = len(self.token2idx)
        self.token2idx[self.pad_token] = self.pad_token_idx
        self.idx2token[self.pad_token_idx] = self.pad_token

        # add <<SUB>> special token
        self.sub_token_idx = len(self.token2idx)
        self.token2idx[self.sub_token] = self.sub_token_idx
        self.idx2token[self.sub_token_idx] = self.sub_token

        # add <<UNK>> special token
        self.unk_token_idx = len(self.token2idx)
        self.token2idx[self.unk_token] = self.unk_token_idx
        self.idx2token[self.unk_token_idx] = self.unk_token

        # add <<EOS>> special token
        self.eos_token_idx = len(self.token2idx)
        self.token2idx[self.eos_token] = self.eos_token_idx
        self.idx2token[self.eos_token_idx] = self.eos_token

    def insert_dicts(self, build_char_vocab=True):

        for (token, _) in self.token_freq_list:
            idx = len(self.token2idx)
            self.idx2token[idx] = token
            self.token2idx[token] = idx

        self.insert_special_tokens()


        print(f"Total Vocab's size: {len(self.token2idx)}")

        self.vocab_dict = {"token2idx": self.token2idx,
                           "idx2token": self.idx2token}

        # load_char_tokens
        if build_char_vocab:
            print("loading character tokens")
            self.get_char_tokens()

    def build_vocab(self,  topk=100000, build_char_vocab=True):
        # retain only topk tokens
        if topk is not None:
            sorted_ = sorted(self.token_freq.items(),
                             key=lambda item: item[1], reverse=True)

            self.token_freq_list = sorted_[:topk]

            print(f"Total tokens retained: {len(self.token_freq_list)}")

        self.insert_dicts(build_char_vocab)

    def build_vocab_from_text(self, path_: str, build_char_vocab=True):
        if not os.path.exists(path_):
            print(f"Vocab: Cannot find dict file: {path_}")
        else:
            print("Building vocab from vocab dict file!")
            with open(path_, 'r') as dict_file:
                for line in dict_file:
                    token_freq = line.split()
                    if token_freq[0] not in [self.pad_token, self.sub_token, self.unk_token, self.eos_token]:
                        try:
                            self.token_freq_list.append((token_freq[0], token_freq[1]))
                        except:
                            print(line)

            self.insert_dicts(build_char_vocab)

    def load_vocab_dict(self, path_: str):
        """
        path_: path where the vocab pickle file is saved
        """
        with open(path_, 'rb') as fp:
            self.vocab_dict = pickle.load(fp)
            self.token2idx = self.vocab_dict['token2idx']
            self.idx2token = self.vocab_dict['idx2token']

            self.chartoken2idx = self.vocab_dict['chartoken2idx']

            self.idx2chartoken = self.vocab_dict['idx2chartoken']

            self.pad_token_idx = self.token2idx[self.pad_token]
            self.sub_token_idx = self.token2idx[self.sub_token]
            self.unk_token_idx = self.token2idx[self.unk_token]

            self.char_unk_token_idx = self.chartoken2idx[self.char_unk_token]

    def save_vocab_dict(self, path_: str):
        """
        path_: path where the vocab pickle file to be saved
        vocab_: the dict data
        """
        with open(path_, 'wb') as fp:
            pickle.dump(self.vocab_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def save_dict_text(self, path_):

        with open(path_, 'w', encoding='utf-8') as ofile:
            print("len(self.token_freq_list): ", len(self.token_freq_list))
            for (subword, fre) in self.token_freq_list:
                ofile.write(f'{subword} {fre}\n')

            ofile.write(f'{self.pad_token} -1\n')
            ofile.write(f'{self.sub_token} -1\n')
            ofile.write(f'{self.unk_token} -1\n')
            ofile.write(f'{self.eos_token} -1\n')

    def get_char_tokens(self):
        special_tokens = [self.char_pad_token, self.char_start_token,
                            self.char_end_token, self.char_unk_token, 
                            self.char_space_token]

        for char in special_tokens:
            idx = len(self.chartoken2idx)
            self.chartoken2idx[char] = idx
            self.idx2chartoken[idx] = char

        if self.lang == 'vi':
            chars = list(
                '''aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[]^_`{|}~''')
        else:
            chars = list(
                '''aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ0123456789,;.!?:'"/\_@#$%^&*~`+-=<>()[]{|}''')

        for char in chars:
            if char not in self.chartoken2idx:
                idx = len(self.chartoken2idx)
                self.chartoken2idx[char] = idx
                self.idx2chartoken[idx] = char

        print(f"number of unique chars found: {len(self.chartoken2idx)}")

        self.vocab_dict["chartoken2idx"] = self.chartoken2idx
        self.vocab_dict["idx2chartoken"] = self.idx2chartoken

        