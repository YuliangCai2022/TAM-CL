import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import math
import shutil
import pickle as pkl
import copy
import pdb
from tqdm import tqdm
from typing import List, Dict

sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.nn import functional as F

from data.visionlanguage_datasets.vcr_dataset import build_vcr_dataloader
from train.task_trainer import TaskTrainer
from WandB import wandb_logger

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


class VCRTrainer(TaskTrainer):

    def __init__(self, 
                 args: argparse.Namespace, 
                 task_configs: Dict, 
                 model_config: Dict, 
                 device: torch.device,
                 teacher_model: torch.nn.Module,
                 ft: bool,
                 num_task: int):
        '''
        Initializes a Trainer that handles training of a model on the VCR task

        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        '''

        super().__init__()

        self.args = args
        self.device = device
        self.finetune = ft
        self.vcr_config = task_configs['vcr']
        self.data_dir = os.path.join(args.climb_data_dir, self.vcr_config['data_dir'])
        self.task_type = self.vcr_config['task_type']
        self.num_task = num_task
        # Model-specific stuff
        self.visual_input_type = model_config['visual_input_type']
        self.batch2inputs_converter = model_config['batch2inputs_converter']

        # Create dataloaders for training and validation
        self.vcr_train_dataloader = build_vcr_dataloader(args=args,
                                                data_dir=self.data_dir,
                                                split='train',
                                                task_type=self.task_type,
                                                visual_input_type=self.visual_input_type)
    
        self.vcr_val_dataloader = build_vcr_dataloader(args=args,
                                                data_dir=self.data_dir,
                                                split='val',
                                                task_type=self.task_type,
                                                visual_input_type=self.visual_input_type)

        # Training hyperparameters
        self.num_epochs = self.vcr_config['num_epochs']
        self.lr = self.vcr_config['lr']
        self.adam_epsilon = self.vcr_config['adam_epsilon']
        self.weight_decay = self.vcr_config['weight_decay']
        self.loss_criterion = nn.CrossEntropyLoss()
        self.max_steps = len(self.vcr_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1 # TODO remove hard code

    def get_train_dataloader(self):
        return self.vcr_train_dataloader

    def get_collate_fn(self):
        return self.vcr_train_dataloader.collate_fn

    def forward_pass(self, model, batch: Dict, do_eval: bool = False) -> tuple:
        '''
        Forward pass of batch inputs through model
        output: tuple containing (encoder_pooled_output, output_logits)
        '''

        inputs = self.batch2inputs_converter(batch)
        if do_eval is True:
            with torch.no_grad():
                output = model(task_key='vcr', **inputs)
        else:
            output = model(task_key='vcr', **inputs)
        return output


    def train_step(self, model, batch: Dict, optimizer=None, scheduler=None, ewc=None):

        '''
        A single training step, including forward pass and backpropagation of loss

        Args:
        model
        batch: Dictionary containing model inputs
        optimizer
        scheduler
        ewc: Instance of EWC class for computing EWC loss

        Returns:
        loss
        output: output tuple from forward_pass
        ewc_task: string indicating which previous task's weights to compare against
        ewc_loss
        ''' 

        output = self.forward_pass(model, batch)
        if self.args.dytox == 0:
            '''
            if (len(output[1].shape) == 3):
                logits = torch.max(output[1],-1)[0]
            else:
                logits = output[1]
            '''
            logits = output[1]
        else:
            logits = output['logits']
            
        target = batch['labels'].to(self.device)
        #loss = self.loss_criterion(logits[:,3132:], target)

        #loss = self.loss_criterion(logits[:,3132:], target)
        if target.shape[0] == logits.shape[0]:
            loss = self.loss_criterion(torch.max(logits,-1)[0].squeeze(), target)
        #    #loss = self.loss_criterion(logits[:,:,-1], target)
        #    #logger.info("logits shape is " + str(logits.shape))
        #    loss = self.loss_criterion(logits[:,3132:], target)
        else:
            return 0,0,0,0

        logger.info("loss is " + str(loss))
       
        
        if self.args.dytox != 0:
            if model.teacher_model != None and self.args.task_attention:
                # get the output from the model of previous task
                
                kd_loss = 0
                tau = 5
                old_inputs = self.batch2inputs_converter(batch)
                output_old = model.teacher_model(task_key='snli-ve', teacher_key = 'vcr', **old_inputs)
                output_old = output_old['test']
                #logger.info("output old shape is " + str(output_old.shape))
                logits_kd = output['test'][:,:output_old.shape[1]]
                #logits_kd = logits[0,:,:output_old.shape[2]]
                #logits_kd = logits[:,:output_old.shape[1]]
                #logger.info("logits kd shape is " + str(logits_kd.shape))
                #output_old = output_old.reshape(output_old.shape[0]*output_old.shape[1],-1)
                #logits_kd = logits_kd.reshape(logits_kd.shape[0]*logits_kd.shape[1],-1)
                kd_loss = 0
                _kd_loss = F.kl_div(
                        F.log_softmax(logits_kd / tau, dim=1),
                        F.log_softmax(output_old / tau, dim=1),
                        reduction='mean',
                        log_target=True
                ) * (tau ** 2)
                kd_loss += 0.5 * _kd_loss
                logger.info("kd_loss is " + str(kd_loss))
                loss = kd_loss * 20000 + 0.5 * loss

                
                # creating KD loss for vilt intermediate output
                inputs = self.batch2inputs_converter(batch)

                _,_,_,curr_vilt_output,_ = model.forward_features(task_key='vcr', **inputs)
                _,_,_,old_vilt_output,_ = model.teacher_model.forward_features(task_key='snli-ve', teacher_key = 'vcr', **inputs)
                #curr_vilt_output = curr_vilt_output.reshape(curr_vilt_output.shape[0]*curr_vilt_output.shape[1],-1)
                #old_vilt_output = old_vilt_output.reshape(old_vilt_output.shape[0]*old_vilt_output.shape[1],-1)
                kd_loss_vilt = 0
                tau = 1
                _kd_loss_vilt = F.kl_div(
                        F.log_softmax(curr_vilt_output / tau, dim=2),
                        F.log_softmax(old_vilt_output / tau, dim=2),
                        reduction='mean',
                        log_target=True
                ) * (tau ** 2)
                kd_loss_vilt += 0.5 * _kd_loss_vilt
                logger.info('vKD loss is ' + str(kd_loss_vilt))
                loss = kd_loss_vilt * 5000 + loss
                loss -= 0.5 * self.loss_criterion(model.task_tokens[1],model.task_tokens[2])
                

        if ewc is not None and ewc.do_ewc() is True:
            ewc_task, ewc_loss = ewc.compute_ewc_loss(model)
            total_loss = loss + ewc_loss
            total_loss.backward()
        else:
            ewc_task = None
            ewc_loss = None
            loss.backward()

        if optimizer is not None:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        return loss, output, ewc_task, ewc_loss

    def create_optimizer(self, model):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon, betas=(0.9, 0.98))
        return optimizer

    def train(self, model, replay_memory=None, ewc=None) -> (float, Dict):
        '''
        Trains model on VCR task
        Args:
        model
        replay_memory: If experience replay is to be performed
        ewc: If EWC regularization loss is to be added

        Returns:
        best_score: Best validation VCR score
        best_model: Model checkpoint of best validation epoch
        '''

        model.to(self.device)
        if self.args.cl_algorithm == 'adapter':
            model.set_active_adapters("vcr")
        elif self.args.cl_algorithm == 'experience_replay':
            assert replay_memory is not None
            do_replay = replay_memory.do_replay()
        elif self.args.cl_algorithm == 'ewc':
            assert ewc is not None
            do_ewc = ewc.do_ewc()

        # Create optimizer
        optimizer = self.create_optimizer(model)
        # Create Scheduler
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.max_steps * self.warmup_ratio),
            num_training_steps=self.max_steps,
            lr_end=0,
            power=1,
        )

        best_score = 0
        best_model = {
            'epoch': 0,
            'model': copy.deepcopy(model), #model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        model.zero_grad()
        for epoch in range(self.num_epochs):
            # Training loop for epoch

            model.train()
            for step, batch in enumerate(tqdm(self.vcr_train_dataloader, desc='Training epoch {}'.format(epoch+1))):

                loss, output, ewc_task, ewc_loss = self.train_step(model, batch, optimizer, scheduler, ewc)

                if self.args.cl_algorithm == 'experience_replay' and do_replay is True:
                    if (step + 1) % self.args.replay_frequency == 0:
                        sampled_replay_task = replay_memory.sample_replay_task()
                        replay_loss = replay_memory.run_replay_step(task_key=sampled_replay_task, model=model)

                if (step + 1) % wandb_logger.get_log_freq() == 0:
                    log_dict = {'vcr': {'loss': loss.item()}}
                    if ewc is not None and do_ewc is True:
                        log_dict[ewc_task] = {'ewc_loss': ewc_loss.item()}
                    wandb_logger.log(log_dict)

            # Do evaluation after epoch
            eval_score = self.eval(model)
            logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
            wandb_logger.log({'vcr': {'val_score': eval_score}})
            if eval_score > best_score:
                logger.info("New best evaluation score: {:.2f}".format(eval_score))
                best_score = eval_score
                best_model['epoch'] = epoch
                model.teacher_model = None

                best_model['model'] = copy.deepcopy(model)

        return best_score, best_model

    def eval(self, model) -> float:

        '''
        Evaluates model on VCR validation set
        Returns validation VCR accuracy
        '''

        model.eval()
        eval_score = 0

        for step, batch in enumerate(tqdm(self.vcr_val_dataloader, desc='Evaluating on VCR val set')):
            output = self.forward_pass(model, batch, do_eval=True)

            if self.args.dytox:
                logits = output['logits']
            else:
                logits = output[1]

            #logger.info("logits shape is " + str(logits.shape))
            logger.info("logtis.argmax is " + str(logits[:,3132:].argmax(-1).cpu()))
            logger.info("labels are " + str(batch['labels']))
            if batch['labels'].shape[0] != 4:
                logger.info("continue")
                continue
            #batch_scores = (torch.max(logits,-1)[0].squeeze().argmax(-1).cpu() == batch['labels'])
            batch_scores = (logits[:,:,-1].argmax(-1).cpu() == batch['labels'])
            #batch_scores = (logits[:,3132:].argmax(-1).cpu() == batch['labels'])
            eval_score += batch_scores.sum().item()

        eval_score = eval_score/len(self.vcr_val_dataloader.dataset)*100.0

        model.train()
        return eval_score

    def eval_forgetting(self, model, model_path: str) -> float:

        '''
        Evaluates forgetting by loading model weights from model_path, 
        which has encoder weights of later task and classifier weights from VCR
        Returns VCR evaluation accuracy of post-VCR model checkpoint
        '''

        model.to(self.device)
        if self.args.cl_algorithm == 'adapter':
            model.set_active_adapters("vcr")

        # Load model with encoder weights from encoder_path, and classifier weights from model_path
        model.load_state_dict(torch.load(model_path))
        logger.info("Loaded model checkpoint from {}".format(model_path))

        return self.eval(model)

class LowShotVCRTrainer(VCRTrainer):

    def __init__(self,
                 args: argparse.Namespace, 
                 task_configs: Dict, 
                 model_config: Dict, 
                 device: torch.device, 
                 low_shot_config: Dict = None):

        '''
        Creates instance of low-shot VCR trainer according to low_shot_config
        
        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        low_shot_config: dictionary containing low-shot configuration parameters
        '''

        super(LowShotVCRTrainer, self).__init__(args, task_configs, model_config, tokenizer, device)
        self.low_shot_config = low_shot_config
        self.eval_epochs = [x-1 for x in low_shot_config['eval_epochs']]

        self.vcr_train_dataloader.dataset.convert_to_low_shot(low_shot_percentage=low_shot_config['percentage'])
        self.max_steps = len(self.vcr_train_dataloader) * self.num_epochs

    def train(self, model) -> (float, Dict):
        '''
        Trains model on VCR task
        Args:
        model

        Returns:
        best_score: Best validation VCR score
        best_model: Model checkpoint of best validation epoch
        '''

        model.to(self.device)

        # Create optimizer
        optimizer = self.create_optimizer(model)
        # Create Scheduler
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.max_steps * self.warmup_ratio),
            num_training_steps=self.max_steps,
            lr_end=0,
            power=1,
        )

        best_score = 0
        best_model = {
            'epoch': 0,
            'model': copy.deepcopy(model), #model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        model.zero_grad()
        for epoch in range(self.num_epochs):
            # Training loop for epoch

            model.train()
            for step, batch in enumerate(tqdm(self.vcr_train_dataloader, desc='Training epoch {}'.format(epoch+1))):

                loss, output, _, _ = self.train_step(model, batch, optimizer, scheduler)

            if epoch in self.eval_epochs:
                # Do evaluation after epoch
                eval_score = self.eval(model)
                logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
                wandb_logger.log({'vcr': {'val_score': eval_score}})
                if eval_score > best_score:
                    logger.info("New best evaluation score: {:.2f}".format(eval_score))
                    best_score = eval_score
                    best_model['epoch'] = epoch
                    best_model['model'] = copy.deepcopy(model)

        return best_score, best_model
