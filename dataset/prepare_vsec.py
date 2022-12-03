import json
from tqdm import tqdm
import sys
from viet_text_tools import normalize_diacritics
sys.path.append("..")
from utils.logger import get_logger
import re
vsec_path = "../data/vi/datasets/vsec/VSEC.jsonl"
test_file = open("../data/vi/datasets/vsec/vsec.test", "w+")
test_noise_file = open("../data/vi/datasets/vsec/vsec.test.noise", "w+")
logger = get_logger("log/violated_cases_prepare.log")

with open(vsec_path, "r") as file:
    data = [json.loads(x[0:-1]) for x in file.readlines()]

def get_true_text(sentence: dict):
    true_tokens = []
    for word in sentence['annotations']:
        if word['is_correct'] == True:
            true_tokens.append(word['current_syllable'])
        else:
            true_tokens.append(word['alternative_syllables'][0])
    true_sentence =  " ".join(true_tokens)
    words = re.findall("\w+|[^\w\s]{1}", true_sentence)
    return " ".join(words)

def get_noise_text(sentence: dict):
    noised_tokens = []
    for word in sentence['annotations']:
        noised_tokens.append(word['current_syllable'])
    noised_sentence = " ".join(noised_tokens)
    words = re.findall("\w+|[^\w\s]{1}", noised_sentence)   
    noised_tokens = []
    for word in words:
        new_word = normalize_diacritics(word)
        noised_tokens.append(new_word)  
    return " ".join(noised_tokens)

for sentence in tqdm(data):
    true_text = get_true_text(sentence)
    noised_text = get_noise_text(sentence)

    test_file.write(true_text + "\n")
    test_noise_file.write(noised_text + "\n")

test_file.close()
test_noise_file.close()