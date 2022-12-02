import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from termcolor import colored
from transformers.optimization import AdamW
from itertools import chain
import sys
sys.path.append("..")

from transformers.optimization import get_linear_schedule_with_warmup
import os
import math
import time
from datetime import datetime as dt
from torch.utils.data import DataLoader
from params import *
from utils.logger import get_logger
from models.model import ModelWrapper
from models.sampler import RandomBatchSampler, BucketBatchSampler
from utils.metrics import get_metric_for_tfm
from accelerate import Accelerator
from dataset.autocorrect_dataset import SpellCorrectDataset
from dataset.util import load_epoch_dataset


class Trainer():
    def __init__(self, model_wrapper: ModelWrapper, data_path, dataset_name, valid_dataset: Dataset):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.model_name = model_wrapper.model_name
        self.data_path = data_path
        self.incorrect_file = f'{dataset_name}.train.noise'
        self.correct_file = f'{dataset_name}.train'
        self.length_file = f'{dataset_name}.length.train'
        train_dataset = load_epoch_dataset(data_path, self.correct_file, \
            self.incorrect_file, self.length_file, 1, EPOCHS)
        train_dataset = SpellCorrectDataset(dataset=train_dataset)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        if not BUCKET_SAMPLING:
            self.train_sampler = RandomBatchSampler(train_dataset, TRAIN_BATCH_SIZE)
            self.valid_sampler = RandomBatchSampler(valid_dataset, VALID_BATCH_SIZE, shuffle = False)
        else:
            self.train_sampler = BucketBatchSampler(train_dataset)
            self.valid_sampler = BucketBatchSampler(valid_dataset, shuffle = False)

        self.train_data = DataLoader(dataset=train_dataset, batch_sampler=self.train_sampler,
                                      collate_fn=model_wrapper.collator.collate, num_workers=2, pin_memory=True)

        self.valid_data = DataLoader(dataset=valid_dataset, batch_sampler=self.valid_sampler,
                                      collate_fn=model_wrapper.collator.collate, num_workers=2, pin_memory=True)

        self.total_training_steps = int(len(self.train_dataset) * EPOCHS)

        self.checkpoint_cycle = math.ceil((len(self.train_data) * EPOCHS / CHECKPOINT_FREQ) / PRINT_PER_ITER) * PRINT_PER_ITER


        self.print_every = PRINT_PER_ITER

        self.iter = 0
        self.scratch_iter = 0
        self.start_epoch = 1
        self.best_F1 = -1
        self.current_epoch = 1
        self.progress_epoch = None

        self.max_epochs = EPOCHS
        self.learning_rate = MAX_LR

        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=0.01,
                               correct_bias=False)

        self.num_warmup_steps = WARMUP_PERCENT * self.total_training_steps

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.total_training_steps)
        
        self.train_losses = []

        self.accelerator = Accelerator(cpu= True if DEVICE == "cpu" else False)
        self.device = self.accelerator.device
        self.total_fw_time = 0

        log_path = LOG + \
            f'/pytorch.{self.model_name}.lr.{self.learning_rate}.train.log'

        if log_path:
            self.logger = get_logger(log_path)

        self.logger.log(f'DEVICE is: {self.device}')
        self.logger.log(
            f"============TOTAL TRAINING STEPS===========\n{self.total_training_steps}")
        self.logger.log(f"CHECKPOINT CYCLE: {self.checkpoint_cycle} ITER")

    def load_lazy_dataset(self, epoch):
        self.train_dataset = load_epoch_dataset(self.data_path, self.correct_file,\
                self.incorrect_file, self.length_file, epoch, EPOCHS)
            
        if not BUCKET_SAMPLING:
            self.train_sampler = RandomBatchSampler(self.train_dataset, TRAIN_BATCH_SIZE)
        else:
            self.train_sampler = BucketBatchSampler(self.train_dataset)

        self.train_data = DataLoader(dataset=self.train_dataset, batch_sampler=self.train_sampler,
                                    collate_fn=self.model_wrapper.collator.collate,\
                                    num_workers=2, pin_memory=True)   
        

    def step(self, batch, training=True):

        if training:
            self.model.train()
            start = time.time()
            outputs = self.model(batch['batch_src'], batch['batch_tgt']) # outputs.logits , outputs.loss
            self.total_fw_time += time.time() - start
            loss = outputs['loss']
            batch_loss = outputs['loss'].cpu().detach().numpy()
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(self.iter)
            return batch_loss
        else:
            self.model.eval()
            outputs = self.model(batch['batch_src'], batch['batch_tgt'])
            return outputs['loss'], outputs['preds'], \
                batch['batch_tgt'].cpu().detach().numpy(), batch['lengths']

    def train(self):
        self.logger.log("Loading model to device")

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare( 
            self.model, self.optimizer, self.scheduler)

        self.logger.log(f"Begin training from epoch: {self.start_epoch}")

        total_time = 0
        total_loss = 0
        overall_loss, overall_iter = 0, 0
        patience = 0

        for epoch_id in range(self.start_epoch, self.max_epochs):
            self.current_epoch = epoch_id
            if self.progress_epoch and self.progress_epoch != epoch_id:
                self.load_lazy_dataset(epoch_id)
                self.logger.log(f"Loaded lazy dataset {epoch_id} / {self.max_epochs}")
            else:
                self.progress_epoch = None
            self.logger.log(f"START OF EPOCH {epoch_id}")
            for step, batch in enumerate(self.train_data):
                start = time.time()
                self.iter += batch['batch_tgt'].size(0)
                self.scratch_iter += batch['batch_tgt'].size(0)
                overall_iter += batch['batch_tgt'].size(0)
                batch_loss = self.step(batch)
                total_time += time.time() - start
                total_loss += batch_loss
                overall_loss += batch_loss
                if step % self.print_every == 0:
                    info = '{} - epoch: {} - step: {} - iter: {:08d}/{:08d} - train loss: {:.5f} - lr: {:.5e} - {} time: {:.2f}s'.format(
                        colored(str(dt.now()),"green"),
                        epoch_id,
                        step,
                        self.iter,
                        self.total_training_steps,
                        total_loss / self.print_every,
                        self.optimizer.param_groups[0]['lr'],
                        self.device,
                        total_time)

                    total_loss = 0
                    total_time = 0

                    self.logger.log(info)
                
                if step % self.checkpoint_cycle == 0:
                    torch.cuda.empty_cache()    
                    if step == 0:
                        continue
                    # <---- validate ----->
                    val_loss, val_accu, val_mean_time = self.validate()

                    info = '{} - epoch: {} - valid loss: {:.5f} - valid accuracy: {:.4f}'.format(
                        colored(str(dt.now()),"green"), epoch_id, val_loss, val_accu)

                    self.logger.log(info)
                    self.logger.log(f"Overall trainning loss between two checkpoints: {overall_loss / overall_iter}")
                    overall_loss, overall_iter = 0, 0
                    if val_accu > self.best_F1:
                        self.best_F1 = val_accu
                        info = 'Saving weights to disk......'
                        self.logger.log(info)
                        self.save_weights(self.checkpoint_dir, epoch_id, self.best_F1)
                        info = 'Saving checkpoint to disk......'
                        self.logger.log(info)
                        self.save_checkpoint(
                            self.checkpoint_dir, epoch_id, self.best_F1)
                        patience = 0
                    else:
                        patience += 1

                    self.logger.log("Mean forward time: {:.5f}".format(
                        self.total_fw_time / VALID_BATCH_SIZE))
                        
                    self.total_fw_time = 0

                    if patience >= PATIENCE:
                        break

                    torch.cuda.empty_cache()

            ## Validation before next epoch
            torch.cuda.empty_cache()
            val_loss, val_accu, val_mean_time = self.validate()

            info = '{} - epoch: {} - valid loss: {:.5f} - valid accuracy: {:.4f}'.format(
                        colored(str(dt.now()),"green"), epoch_id, val_loss, val_accu)  
            self.logger.log(info)
            self.logger.log(f"Overall trainning loss between two checkpoints: {overall_loss / overall_iter}")
            overall_loss, overall_iter = 0, 0
            if val_accu > self.best_F1:
                self.best_F1 = val_accu
                info = 'Saving weights to disk......'
                self.logger.log(info)
                self.save_weights(self.checkpoint_dir, epoch_id, self.best_F1)
                info = 'Saving checkpoint to disk......'
                self.logger.log(info)
                self.save_checkpoint(
                    self.checkpoint_dir, epoch_id, self.best_F1)
                patience = 0
            else:
                patience += 1

            self.logger.log("Mean forward time: {:.5f}".format(
                self.total_fw_time / VALID_BATCH_SIZE))
                
            self.total_fw_time = 0

            if patience >= PATIENCE:
                break

            torch.cuda.empty_cache()
            self.iter = 0
            self.scratch_iter = 0

            self.logger.log(f"END OF EPOCH {epoch_id}")

        self.logger.log("Train complete!")            

    def validate(self):
        total_loss = 0
        valid_loss = 0
        valid_time = 0
        total_time = 0
        total_examples = 0

        num_correct, num_wrong = 0, 0
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_data):

                start = time.time()

                total_examples += batch['batch_tgt'].size(0)

                batch_loss, batch_predictions, \
                    batch_label_ids, batch_lengths = self.step(
                        batch, training=False)
                valid_time += time.time() - start

                batch_token_lens = batch['lengths']
                batch_label_ids = batch['batch_tgt'].cpu().detach().numpy()
                
                _num_correct, _num_wrong = get_metric_for_tfm(batch_predictions, batch_label_ids, batch_token_lens)
                
                
                num_correct += _num_correct
                num_wrong += _num_wrong
                valid_loss += batch_loss
                total_loss += batch_loss

                if step % self.print_every == 0:
                    info = '{} Validation - iter: {:08d}/{:08d} - valid loss: {:.5f} - {} time: {:.2f}s'.format(
                        colored(str(dt.now()),"green"),
                        step,
                        len(self.valid_data),
                        valid_loss / self.print_every,
                        self.device,
                        valid_time / self.print_every)

                    valid_loss = 0
                    total_time += valid_time
                    valid_time = 0

                    self.logger.log(info)

                del batch_loss
        avg_loss = total_loss / len(self.valid_data)
        avg_accu = num_correct / (num_correct + num_wrong)
        avg_time = total_time / total_examples

        return avg_loss, avg_accu, avg_time

    def load_checkpoint(self, checkpoint_dir, dataset_name, start_epoch=0):
        self.checkpoint_dir = checkpoint_dir
        self.dataset_name = dataset_name
        
        checkpoint_path = checkpoint_dir + \
                f'/{dataset_name}.model.epoch_{start_epoch - 1}.pth'

        if start_epoch > 0 and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device('cpu'))

            assert EPOCHS == checkpoint['num_epochs']

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.base_lrs = [MAX_LR]
            self.scheduler.base_lrs = [MAX_LR]
            self.model.load_state_dict(checkpoint['state_dict'])
            
            self.iter = checkpoint['iter']
            self.remained_indies = checkpoint['remained_indies']

            self.start_epoch = checkpoint['epoch']
            self.progress_epoch = self.start_epoch
            self.scratch_iter = checkpoint['scratch_iter']

            train_dataset = load_epoch_dataset(self.data_path, self.correct_file,\
                 self.incorrect_file, self.length_file, self.start_epoch, EPOCHS)
            self.train_dataset = SpellCorrectDataset(dataset=train_dataset)
            if not BUCKET_SAMPLING:
                assert checkpoint['strategy'] == "random_sampling"
                self.train_sampler = RandomBatchSampler(self.train_dataset, TRAIN_BATCH_SIZE)
                self.train_sampler.load_checkpoints(self.scratch_iter)
            else:
                assert checkpoint['strategy'] == "bucket_sampling"
                self.train_sampler = BucketBatchSampler(self.train_dataset)
                self.train_sampler.load_checkpoints(self.remained_indies)

            self.train_data = DataLoader(dataset=self.train_dataset, batch_sampler=self.train_sampler,
                                        collate_fn=self.model_wrapper.collator.collate,\
                                        num_workers=2, pin_memory=True)                        

            
            self.best_F1 = checkpoint['best_F1']

    def save_checkpoint(self, checkpoint_dir, epoch, best_F1):
        checkpoint_path = checkpoint_dir + \
            f'/{self.dataset_name}.model.epoch_{epoch}.pth'
        flatten_iterator_indies = list(chain.from_iterable(self.train_sampler.seq))
        remained_indies = flatten_iterator_indies[self.scratch_iter:None]
        self.logger.log(f"Traversed iter from beginning: {self.scratch_iter}")
        state = {
            'epoch': epoch,
            'iter': self.iter, 'state_dict': self.model.state_dict(), 'scratch_iter': self.scratch_iterz,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_F1': best_F1,
            'remained_indies': remained_indies,
            'strategy': 'bucket_sampling' if BUCKET_SAMPLING else 'random_sampling',
            'num_epochs': EPOCHS
        }

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        info = f'Saving model checkpoint to: {checkpoint_path}'
        self.logger.log(info)

        torch.save(state, checkpoint_path)

    def save_weights(self, checkpoint_dir, epoch, best_F1):
        weight_path = checkpoint_dir + \
                f'/{self.dataset_name}.weights.pth'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_F1': best_F1
        }

        info = f'Saving model weights to: {weight_path}'
        self.logger.log(info)

        torch.save(state, weight_path)
