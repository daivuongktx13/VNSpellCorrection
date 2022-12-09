from utils.logger import get_logger
import numpy as np
from rapidfuzz.distance.Levenshtein import normalized_distance
import multiprocessing
import time
import utils.alignment as alignment

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


def allign_seq2trueseq(seq, true_seq, gap_symbol = "-"):
    prev_sep = None
    next_sep = None
    seq_list = []
    true_list = []
    accumulate_true_word = ""
    accumulate_pred_word = ""
    assert len(true_seq) == len(seq)
    for i in range(len(true_seq)):
        if true_seq[i] != " ":
            accumulate_true_word += true_seq[i]
            accumulate_pred_word += seq[i]
        else:
            if seq[i] == gap_symbol:
                next_sep = gap_symbol
                if prev_sep != None and prev_sep == gap_symbol:
                    accumulate_pred_word = "@@" + accumulate_pred_word
                if next_sep != None and next_sep == gap_symbol:
                    accumulate_pred_word = accumulate_pred_word + "@@"
            else:
                next_sep = " "
                if prev_sep != None and prev_sep == gap_symbol:
                    accumulate_pred_word = "@@" + accumulate_pred_word
                if next_sep != None and next_sep == gap_symbol:
                    accumulate_pred_word = accumulate_pred_word + "@@"
            true_list.append(accumulate_true_word.replace(gap_symbol, ""))
            seq_list.append(accumulate_pred_word)
            accumulate_pred_word = ""
            accumulate_true_word = ""
            prev_sep = next_sep
            next_sep = None   
    return seq_list, true_list

def align_2seq2trueseq(wrong_text, pred_text, true_text, gap_symbol = "-"):
    assert gap_symbol != None and len(gap_symbol) == 1
    seq1, true_seq = alignment.needle(wrong_text, true_text, gap_symbol)
    seq1_list, true_list = allign_seq2trueseq(seq1, true_seq, gap_symbol)
    seq2, true_seq = alignment.needle(pred_text, true_text, gap_symbol)
    seq2_list, _ = allign_seq2trueseq(seq2, true_seq, gap_symbol)
    return list(zip(seq1_list, seq2_list, true_list))

def _get_metric_from_TrueWrongPredictV3(true_text, wrong_text, predict_text, vocab = None):
    gap_symbol = None
    if vocab != None:
        all_symbols = set(list(vocab.chartoken2idx.keys())[4:])
        symbols = set(list(wrong_text + predict_text + true_text))
        usable_symbols = all_symbols.difference(symbols)
        assert len(usable_symbols) > 0
        if "-" not in usable_symbols:
            gap_symbol = usable_symbols.pop()
        else:
            gap_symbol = "-"
    gap_symbol = gap_symbol if gap_symbol != None else "-"

    alignment = align_2seq2trueseq(wrong_text, predict_text, true_text, gap_symbol)
    TP, FP, FN = 0, 0, 0
    for wrong, predict, true in alignment:
        if wrong == true:
            if predict[:-2] == true:
                pass
            elif predict != true:
                if len(predict.split(" ")) == len(true.split(" ")):
                    FP += 1
                else:
                    penalty = len(predict.split(" ")) - len(true.split(" "))
                    assert penalty > 0
                    FP += penalty
        else:
            if predict == true:
                TP += 1
            else:
                if len(predict.split(" ")) == len(true.split(" ")):
                    FN += 1
                else:
                    penalty = len(predict.split(" ")) - len(true.split(" "))
                    assert penalty > 0
                    FN += penalty

    return TP, FP, FN 

def worker_task(true_text, wrong_text, predict_text, vocab):
    _TP, _FP, _FN = _get_metric_from_TrueWrongPredictV3(true_text, wrong_text, predict_text, vocab)
    return (_TP, _FP, _FN)
twp_logger = get_logger(f"./log/true_wrong_predict{time.time()}.log")
from multiprocessing import Pool
def get_metric_from_TrueWrongPredictV3(batch_true_text, batch_wrong_text, batch_predict_text, vocab = None):
    assert vocab != None
    TPs, FPs, FNs = 0, 0, 0
    with Pool(multiprocessing.cpu_count()) as pool:
        data = [(true_text, wrong_text, pred_text, vocab) for true_text, wrong_text, pred_text in zip(batch_true_text, batch_wrong_text, batch_predict_text)]
        result = pool.starmap_async(worker_task, data)
        for i, result in enumerate(result.get()):
            TPs += result[0]
            FPs += result[1]
            FNs += result[2]
            twp_logger.log(batch_true_text[i], file_only=True)
            twp_logger.log(batch_wrong_text[i], file_only=True)
            twp_logger.log(batch_predict_text[i], file_only=True)
            twp_logger.log(f"{result[0]} - {result[1]} - {result[2]}", file_only=True)
    return TPs, FPs, FNs