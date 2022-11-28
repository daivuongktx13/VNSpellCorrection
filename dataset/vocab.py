from __future__ import annotations
import pickle
import re
import os
import sys
import numpy as np
from viet_text_tools import normalize_diacritics


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
        # print(self.chartoken2idx)

        self.vocab_dict["chartoken2idx"] = self.chartoken2idx
        self.vocab_dict["idx2chartoken"] = self.idx2chartoken

    def onehot_recover(self, onehots):
        new_onehots = []
        split = False
        for i in range(len(onehots)):
            value = onehots[i]
            if value > 0:
                if not split:
                    new_onehots.append(value)
                    split = True
                else:
                    continue
            elif value == 0 or value == 1:
                new_onehots.append(value)
            else:
                new_onehots.extend([value]*(abs(value)+1))
            split = False
        return new_onehots

    def onehot_extend(self, onehots):
        new_onehots = []
        for i in range(len(onehots)):
            value = onehots[i]
            if value >= 0:
                new_onehots.append(value)
            else:
                new_onehots.extend([value]*(abs(value)+1))

        return new_onehots

    def count_sub_word(self, tokens, idx):
        assert idx < len(tokens)
        cnt = 0
        while tokens[idx].endswith("@@"):
            if idx < len(tokens):
                cnt += 1
                idx += 1
            else:
                return cnt + 1
        return cnt + 1

    def insert_label(self, label_tokens: list, label_idx, label_ids: list, label_test_tokens: list):
        if self.exist(label_tokens[label_idx]):
            label_ids.append(self.token2idx[label_tokens[label_idx]])
            if DEBUG:
                label_test_tokens.append(label_tokens[label_idx])
        else:
            label_ids.append(self.unk_token_idx)
            if DEBUG:
                label_test_tokens.append(self.unk_token)

    def insert_sub(self, label_ids: list, label_test_tokens: list):
        label_ids.append(self.sub_token_idx)
        if DEBUG:
            label_test_tokens.append(self.sub_token)

    def get_detection_targets(self, batch_noised_tokens, batch_labels_ids):
        batch_labels_ids = batch_labels_ids.tolist()
        batch_detection_targets = []
        batch_label_tokens = []
        for index in range(len(batch_labels_ids)):
            batch_label_tokens.append(self.decode(batch_labels_ids[index]))

        for index in range(len(batch_noised_tokens)):
            sample_length = len(batch_noised_tokens[index])
            detection_targets = list(np.array(np.array(batch_noised_tokens[index] != np.array(batch_label_tokens[index][0:sample_length])), dtype = "int"))
            batch_detection_targets.append(detection_targets)
        
        return batch_detection_targets
    
    def _get_batch_splits(self, batch_noise_tokens):
        batch_splits = []
        for noised_tokens in batch_noise_tokens:
            idxs = np.array([-1] + [idx for idx, token in enumerate(noised_tokens) if not token.endswith("@@")])
            batch_splits.append((idxs[1:] - idxs[:-1]).tolist())
        return batch_splits

    def get_detection_targets_mdc(self, batch_noise_tokens, batch_onehots):
        batch_detection_targets = []
        batch_splits = self._get_batch_splits(batch_noise_tokens)
        
        def map_onehot_key(key):
            if key == 0 or key == 1:
                return key
            elif key > 1:
                return 2
            else:
                return 3
        
        for onehot in batch_onehots:
            batch_detection_targets.append([map_onehot_key(value) for value in onehot])

        return batch_detection_targets, batch_splits
        

    def labelize_with_phobert_subword(self, batch_label_texts, batch_noised_tokens, batch_onehots):
        batch_size = len(batch_label_texts)
        assert batch_size == len(batch_noised_tokens) == len(batch_onehots)
        batch_label_ids = []
        batch_label_lens = []
        batch_label_tokens = []
        for i in range(batch_size):
            token_list_len = len(batch_noised_tokens[i])
            label_tokens = batch_label_texts[i].split()
            noised_tokens = batch_noised_tokens[i]
            assert token_list_len >= len(label_tokens)
            onehots = self.onehot_extend(batch_onehots[i])
            assert token_list_len >= len(onehots)

            label_ids = []
            label_test_tokens = []
            onehot_idx = 0
            label_idx = 0
            idx = 0
            split_cnt = 1
            while idx < token_list_len:
                if onehots[onehot_idx] == 0:
                    if label_tokens[label_idx] == noised_tokens[idx]:
                        self.insert_label(label_tokens, label_idx, label_ids, label_test_tokens)
                    elif noised_tokens[idx].endswith("@@"):
                        self.insert_sub(label_ids, label_test_tokens)
                        idx += 1
                        continue
                    else:
                        self.insert_label(label_tokens, label_idx, label_ids, label_test_tokens)
                elif onehots[onehot_idx] == 1:
                    if noised_tokens[idx].endswith("@@"):
                        self.insert_sub(label_ids, label_test_tokens)
                        idx += 1
                        continue
                    else:
                        self.insert_label(label_tokens, label_idx, label_ids, label_test_tokens)
                elif onehots[onehot_idx] > 1:
                    if noised_tokens[idx].endswith("@@"):
                        self.insert_sub(label_ids, label_test_tokens)
                        idx += 1
                        continue
                    else:
                        if split_cnt < onehots[onehot_idx]:
                            self.insert_sub(label_ids, label_test_tokens)
                            idx += 1
                            onehot_idx  += 1
                            split_cnt += 1
                            continue
                        else:
                            self.insert_label(label_tokens, label_idx, label_ids, label_test_tokens)
                            split_cnt = 1
                else:
                    label_merged_word_cnt = abs(onehots[onehot_idx]) + 1
                    subword_cnt = self.count_sub_word(noised_tokens, idx)
                    assert subword_cnt >= label_merged_word_cnt
                    if subword_cnt == label_merged_word_cnt:
                        for _ in range(subword_cnt):
                            self.insert_label(label_tokens, label_idx, label_ids, label_test_tokens)
                            idx += 1
                            onehot_idx += 1
                            label_idx += 1
                        continue
                    else:
                        
                        num_sub_tokens = subword_cnt - label_merged_word_cnt
                        for i in range(num_sub_tokens):
                            self.insert_sub(label_ids, label_test_tokens)
                        for i in range(label_merged_word_cnt):
                            self.insert_label(label_tokens, label_idx, label_ids, label_test_tokens)
                            label_idx += 1

                        onehot_idx += label_merged_word_cnt
                        idx += subword_cnt
                        continue

                # end of if-else
                idx += 1
                onehot_idx += 1
                label_idx += 1

            # end of while loop
            assert len(label_ids) == len(noised_tokens)
            # except:
            #     print(batch_label_texts[i], batch_noised_tokens[i], batch_onehots[i])
        
            batch_label_ids.append(label_ids)
            batch_label_lens.append(len(label_ids))

            if DEBUG:
                batch_label_tokens.append(label_test_tokens)

        # end of for loop

        if DEBUG:
            return batch_label_ids, batch_label_tokens, batch_label_lens
        else:
            return batch_label_ids, batch_label_lens


    def decode(self, ids):
        tokens = [self.idx2token[idx] if idx in self.idx2token else self.unk_token for idx in ids]
        return tokens


    def labelize(self, batch_labels):
        list_list = [[self.token2idx[token] if token in self.token2idx else self.unk_token_idx
                      for token in line.split()] for line in batch_labels]

        list_len = [len(x) for x in list_list]

        return list_list, list_len

if __name__ == "__main__":
    import argparse
    description='''
        vocab.py:

        Usage: python vocab.py --dataset vi_wiki --file vi_wiki.dict.txt --test False
                    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--file', type=str, default='vi_wiki.dict.txt')
    parser.add_argument('--dataset', type=str, default='vi_wiki')
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    if args.test:
        dict_text_path = '../data/vi/datasets/vi_wiki/vi_wiki.dict.test.txt'
        dict_pickle_path = '../data/vi/datasets/vi_wiki/vi_wiki.vocab.test.pkl'
    else:
        dict_text_path = '../data/vi/datasets/vi_wiki/vi_wiki.dict.txt'
        dict_pickle_path = '../data/vi/datasets/vi_wiki/vi_wiki.vocab.pkl'
    
    vocab = Vocab()
    vocab.load_vocab_dict(dict_pickle_path)

    batch_labels = [
        "Emiliella epapposa là một loài thực vật có hoa trong họ Cúc .",
        "Emiliella epapposa là một loài thực vật có hoa trong họ Cúc ."
    ]
    batch_noisy = [
        "Emiliella epapposa là một loài thực vật có hoa trong họ Cúc .",
        "Emiliella epapposa là một loàithựcvậtcó ho a t rong họ Cúc ."
    ]

    batch_onehots = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -3, 2, 2, 2, 2, 0, 0, 0] # [0, 0, 0, 0, -3, -3, -3, -3, 2, 2, 2, 2, 0, 0, 0]

    ]
    batch_tokens = [
        ['Emili@@', 'ella', 'ep@@', 'ap@@', 'p@@', 'osa', 'là', 'một', 'loài', 'thực', 'vật', 'có', 'hoa', 'trong', 'họ', 'Cúc', '.'],
        ['Emili@@', 'ella', 'ep@@', 'ap@@', 'p@@', 'osa', 'là', 'một', 'l@@', 'oà@@', 'i@@', 'thự@@', 'c@@', 'vậ@@', 't@@', 'có', 'ho', 'a', 't', 'rong', 'họ', 'Cúc', '.']
    ]

    batch_label_ids, batch_label_tokens, _ = vocab.labelize_with_phobert_subword(batch_labels, batch_tokens, batch_onehots)

    print("batch_label_ids:", batch_label_ids)
    print("batch_label_tokens:", batch_label_tokens)