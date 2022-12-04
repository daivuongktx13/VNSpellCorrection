import torch
import time

from torch.utils.data import DataLoader
from datetime import datetime as dt
from params import *
from models.model import ModelWrapper
from utils.metrics import get_mned_metric_from_TruePredict, get_metric_from_TrueWrongPredictV2
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
            batch = self.model_wrapper.collator.collate([[None, batch,None, None]], type = "correct")
            
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

    def step(self, batch, num_beams = 2):
        outputs= self._get_transfomer_with_tr_generations(batch, num_beams)
        batch_predictions = outputs['predict_text']
        batch_label_texts = batch['label_texts']
        batch_noised_texts = batch['noised_texts']

        return batch_predictions, batch_noised_texts, batch_label_texts

    def _evaluation_loop_autoregressive(self, data_loader, num_beams = 2):
        TP, FP, FN = 0, 0, 0
        MNED = 0.0
        O_MNED = 0.0
        total_infer_time = 0.0
        with torch.no_grad():

            self.model.eval()

            for step, batch in enumerate(data_loader):

                batch_infer_start = time.time()

                batch_predictions, batch_noised_texts, batch_label_texts = \
                        self.step(batch, num_beams = num_beams)

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
        
    def evaluate(self, dataset, beams: int = None):

        def test_collate_wrapper(batch):
            return self.model_wrapper.collator.collate(batch, type = "test")

        if not BUCKET_SAMPLING:
            self.test_sampler = RandomBatchSampler(dataset, VALID_BATCH_SIZE, shuffle = False)
        else:
            self.test_sampler = BucketBatchSampler(dataset, shuffle = True)

        data_loader = DataLoader(dataset=dataset,batch_sampler= self.test_sampler,\
            collate_fn=test_collate_wrapper)
            
        self.test_iters = len(data_loader)

        assert beams != None
        total_infer_time, TP, FP, FN, MNED, O_MNED = self._evaluation_loop_autoregressive(data_loader, num_beams = beams)
            

        self.logger.log("Total inference time for this data is: {:4f} secs".format(total_infer_time))
        self.logger.log("###############################################")

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
