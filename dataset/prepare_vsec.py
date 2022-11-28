import json
from tqdm import tqdm
from util import check_violated_cases
import sys
from viet_text_tools import normalize_diacritics
sys.path.append("..")
from utils.logger import get_logger
import re
vsec_path = "../data/vi/datasets/vsec/VSEC.jsonl"
test_file = open("../data/vi/datasets/vsec/vsec.test", "w+")
test_noise_file = open("../data/vi/datasets/vsec/vsec.test.noise", "w+")
test_onehot_file =  open("../data/vi/datasets/vsec/vsec.onehot.test", "w+")

logger = get_logger("log/violated_cases_prepare.log")

from transformers import AutoTokenizer
PHOBERT_MAX_SEQ_LEN = 256
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

with open(vsec_path, "r") as file:
    data = [json.loads(x[0:-1]) for x in file.readlines()]

# token_wo_delimiter = re.sub("[.,\":\';?]", "", token_wo_delimiter)

### VSEC data do have delimiters at the start or the end of words for example:
### thanh"? | ,thanh 
vietnamese_chars = "àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ"
def strip_delimiter_for_true_word(word):
    if word == "":
        return True, [], -1
    if len(word) <= 1 or re.search(f"[0-9a-zA-Z{vietnamese_chars}]+", word) == None:
        return False, None, 0
    delimiter_end = []
    delimiter_start = []
    result_end = re.search("[^\w\s]+$", word)
    result_start = re.search("^[^\w\s]*", word)
    if result_end:
        delimiter_end = [*result_end.group()]
    if result_start:
        delimiter_start = [*result_start.group()]
    start_index = len(delimiter_start)
    end_index = -len(delimiter_end) if len(delimiter_end) > 0 else None
    return True, [*delimiter_start, word[start_index: end_index], *delimiter_end], len(delimiter_start) + len(delimiter_end)

def strip_delimiter_for_false_word(word, target):
    if word == "":
        return True, [], []
    if len(word) <= 1 or re.search(f"[0-9a-zA-Z{vietnamese_chars}]+", word) == None:
        return False, None, []
    delimiter_end = []
    delimiter_start = []
    result_end = re.search("[^\w\s]+$", word)
    result_start = re.search("^[^\w\s]*", word)
    if result_end:
        delimiter_end = [*result_end.group()]
    if result_start:
        delimiter_start = [*result_start.group()]
    start_index = len(delimiter_start)
    end_index = -len(delimiter_end) if len(delimiter_end) > 0 else None
    if is_error_with_missing_or_redundant_delimiters_at_start(word, target):
        delimiter_start = []
    if is_error_with_missing_or_redundant_delimiters_at_end(word, target):
        delimiter_end = []
    return True, [*delimiter_start, word[start_index: end_index], *delimiter_end],\
         [* "0" * len(delimiter_start), "1", * "0" * len(delimiter_end)]



def get_noise_text(sentence: dict):
    noised_tokens = []
    for word in sentence['annotations']:
        if word['is_correct'] == True:
            has_delimiter, word_list, num_delimiters = \
                strip_delimiter_for_true_word(word['current_syllable'])
            if has_delimiter:
                noised_tokens.extend(word_list)
            else:
                noised_tokens.append(word['current_syllable'])
        else:
            has_delimiter, word_list, gen_onehot = \
                strip_delimiter_for_false_word(word['current_syllable'], word['alternative_syllables'][0])
            if has_delimiter:
                noised_tokens.extend(word_list)
            else:
                noised_tokens.append(word['current_syllable'])
    noised_tokens = [normalize_diacritics(token) for token in noised_tokens]            
    return " ".join(noised_tokens)

def get_true_text(sentence: dict):
    true_tokens = []
    for word in sentence['annotations']:
        if word['is_correct'] == True:
            has_delimiter, word_list, num_delimiters = \
                strip_delimiter_for_true_word(word['current_syllable'])
            if has_delimiter:
                true_tokens.extend(word_list)
            else:
                true_tokens.append(word['current_syllable'])
        else:
            has_delimiter, word_list, gen_onehot = \
                strip_delimiter_for_false_word(word['alternative_syllables'][0], word['current_syllable'])
            if has_delimiter:
                true_tokens.extend(word_list)
            else:
                true_tokens.append(word['alternative_syllables'][0])
                
    return " ".join(true_tokens)

def is_error_with_missing_or_redundant_delimiters_at_end(word, target):
    result_end = re.search("[^\w\s]+$", word)
    result_end_t = re.search("[^\w\s]+$", target)
    wrapper = [result_end, result_end_t]
    result = [x.group() if x else "" for x in wrapper] 
    return result[0] != result[1]

def is_error_with_missing_or_redundant_delimiters_at_start(word, target):
    result_start = re.search("^[^\w\s]+", word)

    result_start_t = re.search("^[^\w\s]+", target)
    wrapper = [result_start, result_start_t]
    result = [x.group() if x else "" for x in wrapper] 
    return result[0] != result[1]

def get_onehot(sentence: dict):
    onehot = []
    ignore_list = []
    for index, word in enumerate(sentence['annotations']):
        annotations = sentence['annotations']

        if index in ignore_list:
            continue

        annotations = sentence['annotations']
        if word['is_correct'] == True:
            has_delimiter, word_list, num_delimiters = \
                strip_delimiter_for_true_word(word['current_syllable'])
            if has_delimiter:
                onehot.extend(["0"] * (num_delimiters + 1))
            else:
                onehot.append("0")
            continue

        if word['alternative_syllables'][0] not in ["", " "] :
            ## Error missing word "thong" -> "thong bao" or merge word thongbao -> thong bao
            if "- tác động" in word['alternative_syllables']:
                return None
            if len(word['alternative_syllables'][0].split(" ")) > 1:
                onehot.append(str(- len(word['alternative_syllables'][0].split(" ")) + 1))
                continue
                
            has_delimiter, word_list, gen_onehot = \
                strip_delimiter_for_false_word(word['alternative_syllables'][0], word['current_syllable'])
            if has_delimiter:
                onehot.extend(gen_onehot)
            else:
                onehot.append("1")
            continue
        
        ## Error redundant word "canh canh quan" -> "canh quan"
        if index + 1 < len(annotations) and annotations[index + 1]['is_correct'] == True:
            return None

        traverse = 0
        while True:
            traverse += 1
            ignore_list.append(traverse + index)
            if annotations[traverse + index]['alternative_syllables'][0] != "":
                break
        onehot.extend([f"{traverse + 1}"] * (traverse + 1))
    return " ".join(onehot)
number_violate_cases = 0
for sentence in tqdm(data):
    true_text = get_true_text(sentence)
    noised_text = get_noise_text(sentence)
    try:
        onehot = get_onehot(sentence)
    except IndexError:
        onehot = None
        print(true_text)
        print(noised_text)
        print(sentence)

    if onehot == None:
        number_violate_cases += 1
        print(true_text)
        print(noised_text)
        continue

    tokens = tokenizer.tokenize(noised_text)

    if len(tokens) > (PHOBERT_MAX_SEQ_LEN - 2):
        print(f"INFO: BERT noised tokens larger than BERT's max limit (SPLIT MERGE NOISE).")
        print(f"NOISED: {tokens}")
        number_violate_cases += 1
        continue

    if check_violated_cases(tokens, [int(x) for x in onehot.split(" ")]):
        print("INFO: BERT SUBWORD ERROR!!!")
        print(f"text: {noised_text}")
        print(f"noised tokens: {tokens}")
        print(f"onehot: {onehot}")
        number_violate_cases += 1
        continue

    test_file.write(true_text + "\n")
    test_noise_file.write(noised_text + "\n")
    test_onehot_file.write(onehot + "\n")

test_file.close()
test_noise_file.close()
test_onehot_file.close()

print(f"Number violated cases: {number_violate_cases}")
    

