import torch
import time

from torch.utils.data import DataLoader
from datetime import datetime as dt
from params import *
from models.model import ModelWrapper
from utils.metrics import get_metric_for_tfm, get_mned_metric_from_TruePredict, get_metric_from_TrueWrongPredictV2
from utils.logger import get_logger
from models.sampler import RandomBatchSampler, BucketBatchSampler
from termcolor import colored


class Corrector:
    def __init__(self, model_wrapper: ModelWrapper):
        self.model_name = model_wrapper.model_name
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.logger = get_logger("./log/test.log")

        self.device = DEVICE

        self.model.to(self.device)
        self.logger.log(f"Device: {self.device}")
        self.logger.log("Loaded model")
    
    def correct_transfomer_with_tr(self, batch, num_beams = 2):
        correction = dict()
        with torch.no_grad():
            self.model.eval()
            batch_infer_start = time.time()
            batch = self.model_wrapper.collator.collate([[batch, None,None, None]], type = "correct")
            
            result = self.model.inference(batch['batch_src'], num_beams = num_beams,
                 tokenAligner=self.model_wrapper.collator.tokenAligner)
            correction['predict_text'] = result
            correction['noised_text'] = batch['noised_texts']

            total_infer_time = time.time() - batch_infer_start
            correction['time'] = total_infer_time

        return correction
    
    def _get_transfomer_with_tr_generations(self, batch, num_beams = 2):
        correction = dict()
        with torch.no_grad():
            self.model.eval()
            batch_infer_start = time.time()
            
            result = self.model.inference(batch['batch_src'], num_beams = num_beams,
                 tokenAligner=self.model_wrapper.collator.tokenAligner)

            correction['predict_text'] = result
            correction['noised_text'] = batch['noised_texts']

            total_infer_time = time.time() - batch_infer_start
            correction['time'] = total_infer_time

        return correction

    def step(self, batch, training = True, num_beams = 2):
        if training == True:
            outputs = self.model(batch['batch_src'], batch['batch_tgt'])
        else:
            outputs= self._get_transfomer_with_tr_generations(batch, num_beams)

        batch_predictions = outputs['preds'] if training == True else outputs['predict_text']
        batch_token_lens = batch['lengths']
        batch_label_ids = batch['batch_tgt'].cpu().detach().numpy()
        batch_label_texts = batch['label_texts']
        batch_noised_texts = batch['noised_texts']

        return batch_predictions, batch_token_lens, batch_label_ids, batch_noised_texts, batch_label_texts


    def _evaluation_loop_teacher_forcing(self, data_loader):
        num_wrong, num_correct = 0, 0
        TP, FP, FN = 0, 0, 0
        O_MNED = 0.0
        MNED = 0.0
        total_infer_time = 0.0

        with torch.no_grad():
            self.model.eval()

            for step, batch in enumerate(data_loader):

                batch_infer_start = time.time()

                batch_predict_ids, batch_subword_lens, batch_label_ids,\
                    batch_noised_texts, batch_label_texts = self.step(batch)

                batch_infer_time = time.time() - batch_infer_start

                batch_predicts_wo_padding = [predict[0:length + 1] for predict, length in zip(batch_predict_ids, batch_subword_lens) ]

                predict_texts = self.model_wrapper.tokenAligner.tokenizer.batch_decode(batch_predicts_wo_padding, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces = False)
            
                
                _TP, _FP, _FN = get_metric_from_TrueWrongPredictV2(batch_label_texts, batch_noised_texts, predict_texts)

                _num_correct, _num_wrong = get_metric_for_tfm(batch_predict_ids, batch_label_ids, batch_subword_lens)

                num_wrong += _num_wrong
                num_correct += _num_correct
                TP += _TP
                FP += _FP
                FN += _FN

                _MNED = get_mned_metric_from_TruePredict(batch_label_texts, predict_texts)
                MNED += _MNED

                _O_MNED = get_mned_metric_from_TruePredict(batch_label_texts, batch_noised_texts)
                O_MNED += _O_MNED

                info = '{} - Evaluate - iter: {:08d}/{:08d} - correct: {} - wrong: {} - _TP: {} - _FP: {} - _FN: {} - _MNED: {:.5f} - _O_MNED: {:.5f} -  {} time: {:.2f}s'.format(
                    dt.now(),
                    step,
                    self.test_iters,
                    _num_correct,
                    _num_wrong,
                    _TP,
                    _FP,
                    _FN,
                    _MNED,
                    _O_MNED,
                    self.device,
                    batch_infer_time)


                self.logger.log(info)

                torch.cuda.empty_cache()
                total_infer_time += time.time() - batch_infer_start

        return total_infer_time, num_correct, num_wrong, TP, FP, FN, MNED / len(data_loader), O_MNED / len(data_loader)

    def _evaluation_loop_autoregressive(self, data_loader, num_beams = 2):
        TP, FP, FN = 0, 0, 0
        MNED = 0.0
        O_MNED = 0.0
        total_infer_time = 0.0
        with torch.no_grad():

            self.model.eval()

            for step, batch in enumerate(data_loader):

                batch_infer_start = time.time()

                batch_predictions, batch_subword_lens, batch_label_ids,\
                    batch_noised_texts, batch_label_texts = \
                        self.step(batch, training = False, num_beams = num_beams)

                batch_infer_time = time.time() - batch_infer_start

                _TP, _FP, _FN = get_metric_from_TrueWrongPredictV2(batch_label_texts, batch_noised_texts, batch_predictions)

                TP += _TP
                FP += _FP
                FN += _FN

                _MNED = get_mned_metric_from_TruePredict(batch_label_texts, batch_predictions)
                MNED += _MNED

                _O_MNED = get_mned_metric_from_TruePredict(batch_label_texts, batch_noised_texts)
                O_MNED += _O_MNED

                info = '{} - Evaluate - iter: {:08d}/{:08d} - TP: {} - FP: {} - FN: {} - _MNED: {:.5f} - _O_MNED: {:.5f} - {} time: {:.2f}s'.format(
                    dt.now(),
                    step,
                    self.test_iters,
                    _TP,
                    _FP,
                    _FN,
                    _MNED,
                    _O_MNED,
                    self.device,
                    batch_infer_time)

                self.logger.log(info)

                torch.cuda.empty_cache()
                total_infer_time += time.time() - batch_infer_start
        return total_infer_time, TP, FP, FN, MNED / len(data_loader), O_MNED / len(data_loader)
        

    """
        evaluate with metrics
        metrics:
            - auto-regressive: metrics when using beam search for different length seq
            - teacher-forcing: metrics when teacher forcing with decoder
    """
    def evaluate(self, dataset, metrics = "teacher-forcing", beams: int = None):

        def test_collate_wrapper(batch):
            return self.model_wrapper.collator.collate(batch, type = "test")

        if not BUCKET_SAMPLING:
            self.test_sampler = RandomBatchSampler(dataset, VALID_BATCH_SIZE, shuffle = False)
        else:
            self.test_sampler = BucketBatchSampler(dataset, shuffle = True)

        data_loader = DataLoader(dataset=dataset,batch_sampler= self.test_sampler,\
            collate_fn=test_collate_wrapper)
            
        self.test_iters = len(data_loader)

        assert metrics in ["teacher-forcing", "auto-regressive"]

        if metrics == "teacher-forcing":
            total_infer_time, num_correct, num_wrong, TP, FP, FN, MNED, O_MNED = self._evaluation_loop_teacher_forcing(data_loader)

        if metrics == "auto-regressive":
            assert beams != None
            total_infer_time, TP, FP, FN, MNED, O_MNED = self._evaluation_loop_autoregressive(data_loader, num_beams = beams)
            

        self.logger.log("Total inference time for this data is: {:4f} secs".format(total_infer_time))
        self.logger.log("###############################################")

        assert metrics in ["teacher-forcing", "auto-regressive"], "Metrics should be teacher-forcing or auto-regressive"

        if metrics == "teacher-forcing":
            info = "Metrics for Teacher-Forcing"
            self.logger.log(colored(info, "green"))
            info = f"Num_correct tokens: {num_correct}. Num_wrong tokens: {num_wrong}"
            self.logger.log(info)

            info = f"Accuracy: {num_correct / (num_correct + num_wrong)}"
            self.logger.log(info)

            dc_TP = TP
            dc_FP = FP
            dc_FN = FN

            dc_precision = dc_TP / (dc_TP + dc_FP)
            dc_recall = dc_TP / (dc_TP + dc_FN)
            dc_F1 = 2. * dc_precision * dc_recall/ ((dc_precision + dc_recall) + 1e-8)
            self.logger.log(f"TP: {TP}. FP: {FP}. FN: {FN}")

            self.logger.log(f"Precision: {dc_precision}")
            self.logger.log(f"Recall: {dc_recall}")
            self.logger.log(f"F1: {dc_F1}")
            self.logger.log(f"MNED: {MNED}")
            self.logger.log(f"O_MNED: {O_MNED}")

        if metrics == "auto-regressive":
            info = f"Metrics for Auto-Regressive with Beam Search number {beams}"
            self.logger.log(colored(info, "green"))

            dc_TP = TP
            dc_FP = FP
            dc_FN = FN

            dc_precision = dc_TP / (dc_TP + dc_FP)
            dc_recall = dc_TP / (dc_TP + dc_FN)
            dc_F1 = 2. * dc_precision * dc_recall/ ((dc_precision + dc_recall) + 1e-8)

            self.logger.log(f"TP: {TP}. FP: {FP}. FN: {FN}")

            self.logger.log(f"Precision: {dc_precision}")
            self.logger.log(f"Recall: {dc_recall}")
            self.logger.log(f"F1: {dc_F1}")
            self.logger.log(f"MNED: {MNED}")
            self.logger.log(f"O_MNED: {O_MNED}")

        return
