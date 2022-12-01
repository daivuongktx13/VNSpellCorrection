from utils.logger import get_logger
import numpy as np
from rapidfuzz.distance.Levenshtein import normalized_distance
import utils.diff_match_patch as dmp_module


def _get_mned_metric_from_TruePredict(true_text, predict_text):
    return normalized_distance(predict_text, true_text)

def get_mned_metric_from_TruePredict(batch_true_text, batch_predict_text): 
    total_NMED = 0.0
    count = 0
    for true_text, predict_text in zip(batch_true_text, batch_predict_text):
        total_NMED += _get_mned_metric_from_TruePredict(true_text, predict_text)
        count += 1
    return total_NMED / count

def get_metric_for_tfm(batch_predicts, batch_targets, batch_length):
    num_correct, num_wrong = 0, 0
    for predict, target, length in zip(batch_predicts, batch_targets, batch_length):
        predict = predict[1:-1]
        target = target[1:-1]
        predict = np.array(predict[0:length])
        target = np.array(target[0:length])
        num_correct += np.sum(predict == target)
        num_wrong += np.sum(predict != target)
    return num_correct, num_wrong


def diff_wordMode(text1, text2):
  dmp = dmp_module.diff_match_patch()
  a = dmp.diff_linesToWords(text1, text2)
  lineText1 = a[0]
  lineText2 = a[1]
  lineArray = a[2]
  diffs = dmp.diff_main(lineText1, lineText2, False)
  dmp.diff_charsToLines(diffs, lineArray)
  return diffs

def get_misspelled(wrong_text, true_text):
    diff = diff_wordMode(wrong_text, true_text)
    num_words = 0
    misspelled_indies = set()
    misspelled_texts = list()
    for pos, entry in enumerate(diff):
        if entry[0] == -1:
            continue
        if entry[0] == 0:
            words = entry[1].strip(" ").split(" ")
            num_words += len(words)
        if entry[0] == 1:
            words = entry[1].strip(" ").split(" ")
            for i in range(len(words)):
                misspelled_indies.add(num_words + i)
                misspelled_texts.append(words[i])
            num_words += len(words)
    return misspelled_indies, misspelled_texts

def get_restored(predict_text, true_text):
    diff = diff_wordMode(predict_text, true_text)
    num_words = 0
    restored_indies = set()
    restored_texts = list()
    for pos, entry in enumerate(diff):
        if entry[0] == -1:
            continue
        if entry[0] == 1:
            words = entry[1].strip(" ").split(" ")
            num_words += len(words)
        if entry[0] == 0:
            words = entry[1].strip(" ").split(" ")
            for i in range(len(words)):
                restored_indies.add(num_words + i)
                restored_texts.append(words[i])
            num_words += len(words)
    return restored_indies, restored_texts

def get_changed(predict_text, wrong_text):
    diff = diff_wordMode(predict_text, wrong_text)
    num_words = 0
    changed_indies = set()
    changed_texts = list()
    for pos, entry in enumerate(diff):
        if entry[0] == -1:
            continue
        if entry[0] == 0:
            words = entry[1].strip(" ").split(" ")
            num_words += len(words)
        if entry[0] == 1:
            words = entry[1].strip(" ").split(" ")
            for i in range(len(words)):
                changed_indies.add(num_words + i)
                changed_texts.append(words[i])
            num_words += len(words)
    return changed_indies, changed_texts

def get_not_misspelled(wrong_text, true_text):
    diff = diff_wordMode(wrong_text, true_text)
    num_words = 0
    not_misspelled_indies = set()
    not_misspelled_texts = list()
    for pos, entry in enumerate(diff):
        if entry[0] == 1:
            continue
        if entry[0] == 0:
            words = entry[1].strip(" ").split(" ")
            for i in range(len(words)):
                not_misspelled_indies.add(num_words + i)
                not_misspelled_texts.append(words[i])
            num_words += len(words)
        if entry[0] == -1:
            words = entry[1].strip(" ").split(" ")
            num_words += len(words)
    return not_misspelled_indies, not_misspelled_texts

def _get_metric_from_TrueWrongPredictV2(true_text, wrong_text, predict_text):
    TP = get_misspelled(wrong_text, true_text)[0].intersection(get_restored(predict_text, true_text)[0])
    FP = get_not_misspelled(wrong_text, true_text)[0].intersection(get_changed(predict_text, wrong_text)[0])
    FN = get_misspelled(wrong_text, true_text)[0].difference(get_restored(predict_text, true_text)[0])
    return len(TP), len(FP), len(FN)

def get_metric_from_TrueWrongPredictV2(batch_true_text, batch_wrong_text, batch_predict_text): 
    TPs, FPs, FNs = 0, 0, 0
    for true_text, wrong_text, predict_text in zip(batch_true_text, batch_wrong_text, batch_predict_text): 
        TP, FP, FN = _get_metric_from_TrueWrongPredictV2(true_text, wrong_text, predict_text)
        TPs += TP
        FPs += FP
        FNs += FN
    return TPs, FPs, FNs