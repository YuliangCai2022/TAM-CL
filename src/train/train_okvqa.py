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
from typing import List, Dict, Tuple

sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.nn import functional as F

from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from data.visionlanguage_datasets.okvqa_dataset import build_okvqa_dataloader
from train.task_trainer import TaskTrainer
from WandB import wandb_logger

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class OKVQATrainer(TaskTrainer):

    def __init__(self, 
                 args: argparse.Namespace, 
                 task_configs: Dict, 
                 model_config: Dict, 
                 device: torch.device,
                 teacher_model: torch.nn.Module,
                 ft: bool,
                 num_task: int):

        '''
        Initializes a Trainer that handles training of a model on the OKVQA task

        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        '''

        super().__init__()

        self.args = args
        self.device = device
        self.finetune = ft
        self.okvqa_config = task_configs['okvqa']
        self.data_dir = os.path.join(args.climb_data_dir, self.okvqa_config['data_dir'])

        # Model-specific stuff
        self.visual_input_type = model_config['visual_input_type']
        self.batch2inputs_converter = model_config['batch2inputs_converter']
        self.num_task = num_task

        # Load COCO Images dataset for image data backbone
        images_source = self.okvqa_config['images_source']
        mscoco_config = task_configs[images_source]
        self.images_dataset = MSCOCOImagesDataset(coco_dir=os.path.join(args.climb_data_dir, mscoco_config['data_dir']),
                                                  visual_input_type=args.visual_input_type)

        # Create dataloaders for training and validation
        self.okvqa_train_dataloader = build_okvqa_dataloader(args=args,
                                                    data_dir=self.data_dir,
                                                    images_dataset=self.images_dataset,
                                                    split='train',
                                                    visual_input_type=self.visual_input_type)

        self.okvqa_val_dataloader = build_okvqa_dataloader(args=args,
                                                  data_dir=self.data_dir,
                                                  images_dataset=self.images_dataset,
                                                  split='val',
                                                  visual_input_type=self.visual_input_type)

        # Training hyperparameters
        self.num_epochs = self.okvqa_config['num_epochs']
        self.lr = self.okvqa_config['lr']
        self.adam_epsilon = self.okvqa_config['adam_epsilon']
        self.weight_decay = self.okvqa_config['weight_decay']
        self.hparams = {
                        'lr': self.lr,
                        'weight_decay': self.weight_decay,
                        'adam_epsilon': self.adam_epsilon,
        }

        self.loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.max_steps = len(self.okvqa_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1 # TODO remove hard code

    def compute_score_with_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        Given logits for each answer in OKVQA classification, selects answer with max logit and returns OKVQA-score for that answer
        logits: logits for each answer - size=(batch_size, num_answers)
        labels: label for each answer in {0, 0.3, 0.6, 1} (batch_size, num_answers)
        
        Returns:
        scores: score of predicted answer (batch_size, num_answers)
        '''

        logits = torch.max(logits, 1)[1].data # argmax
        one_hots = torch.zeros(*labels.size()).to(self.device)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * labels)
        return scores
    
    def get_train_dataloader(self):
        return self.okvqa_train_dataloader

    def get_collate_fn(self):
        return self.okvqa_train_dataloader.collate_fn

    def forward_pass(self, model, batch: Dict, do_eval: bool = False) -> Tuple:
        '''
        Forward pass of batch inputs through model
        output: tuple containing (encoder_pooled_output, output_logits)
        '''

        inputs = self.batch2inputs_converter(batch)
        if do_eval is True:
            with torch.no_grad():
                output = model(task_key='okvqa', **inputs)
        else:
            output = model(task_key='okvqa', **inputs)
        return output, inputs

    def create_optimizer(self, model):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon, betas=(0.9, 0.98))
        return optimizer


    def train_step(self, model, batch: Dict, optimizer=None, scheduler=None, ewc=None, replay=None):

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

        output,batch_inputs = self.forward_pass(model, batch)
        logits = None
        div_output = None
        if self.args.dytox == 0:
            logits = output[1]
        else:
            logits = output['logits']
            div_output = output['div']
        target = batch['target_scores'].to(self.device)
        if logits.shape[1] != target.shape[1]:
            #target = torch.tensor(np.concatenate((np.zeros((logits.shape[0],logits.shape[1]-target.shape[1])),np.asarray(target.cpu())),1)).to(self.device)
            target = torch.tensor(np.concatenate((np.asarray(target.cpu()),np.zeros((logits.shape[0],logits.shape[1]-target.shape[1]))),1)).to(self.device)

        #loss = self.loss_criterion(logits, target) #* target.shape[1]
        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(logits, target)

        
        if self.args.dytox != 0 and replay != 1:
            if self.args.parallel != 0:
                   pass # deleted
            else:
                if model.teacher_model != None and self.args.task_attention:
                    # get the output from the model of previous task
                    kd_loss = 0
                    tau = 5
                    old_inputs = batch_inputs
                    output_old_origin = model.teacher_model(task_key=self.args.ordered_cl_tasks[self.num_task-2],teacher_key = 'okvqa', **old_inputs)
                    
                    '''
                    # the inner KD 
                    curr_intermediate = output['mid_features']
                    old_intermediate = output_old_origin['mid_features']

                    inner_kd_loss = 0

                    for key in curr_intermediate:
                        _kd_loss = F.kl_div(
                            F.log_softmax(curr_intermediate[key] / tau, dim=1),
                            F.log_softmax(old_intermediate[key] / tau, dim=1),
                            reduction='mean',
                            log_target=True
                        ) * (tau ** 2)
                        inner_kd_loss += (self.num_task-1)/(self.num_task) * _kd_loss

                    loss += inner_kd_loss * 5000'''

                    
                    # output KD
                    output_old = output_old_origin['logits']
                    logits_kd = logits[:,:output_old.shape[1]]
                    kd_loss = 0
                    _kd_loss = F.kl_div(
                            F.log_softmax(logits_kd / tau, dim=1),
                            F.log_softmax(output_old / tau, dim=1),
                            reduction='mean',
                            log_target=True
                    ) * (tau ** 2)
                    kd_loss += (self.num_task-1)/(self.num_task) * _kd_loss
                    loss += 5000 * kd_loss
                    
                    '''
                    curr_vilt_output = output['v_output'] # ikd is 'v_output / tokens [-1]
                    old_vilt_output = output_old_origin['v_output'] #ikd is 'v_output
                    kd_loss_vilt = 0
                    tau = 1
                    _kd_loss_vilt = F.kl_div(
                            F.log_softmax(curr_vilt_output / tau, dim=1),
                            F.log_softmax(old_vilt_output / tau, dim=1),
                            reduction='mean',
                            log_target=True
                    ) * (tau ** 2)
                    kd_loss_vilt += (self.num_task-1)/(self.num_task) * _kd_loss_vilt
                    loss = kd_loss_vilt * 5000 + loss  # used to be 10000'''

                    
                    for i in range(self.num_task-1): #self.loss_criterion()
                        loss -= max(0.1 * self.loss_criterion(model.task_tokens[i],model.task_tokens[-1]),1/(self.num_task-1)*0.05*loss)
                   

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

    def train(self, model, replay_memory=None, ewc=None) -> Tuple[float, Dict]:
        '''
        Trains model on OKVQA task
        Args:
        model
        replay_memory: If experience replay is to be performed
        ewc: If EWC regularization loss is to be added

        Returns:
        best_score: Best validation OKVQA score
        best_model: Model checkpoint of best validation epoch
        '''
        model.to(self.device)
        if self.args.replay== 1:
            assert replay_memory is not None
            do_replay = replay_memory.do_replay()
        elif self.args.ewc == 1:
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
            for step, batch in enumerate(tqdm(self.okvqa_train_dataloader, desc='Training epoch {}'.format(epoch+1))):

                loss, output, ewc_task, ewc_loss = self.train_step(model, batch, optimizer, scheduler, ewc)

                if self.args.cl_algorithm == 'experience_replay' and do_replay is True:
                    if (step + 1) % self.args.replay_frequency == 0:
                        sampled_replay_task = replay_memory.sample_replay_task()
                        replay_loss = replay_memory.run_replay_step(task_key=sampled_replay_task, model=model)

                if (step + 1) % wandb_logger.get_log_freq() == 0:
                    log_dict = {'okvqa': {'loss': loss.item()}}
                    if ewc is not None and do_ewc is True:
                        log_dict[ewc_task] = {'ewc_loss': ewc_loss.item()}
                    wandb_logger.log(log_dict)

            # Do evaluation after epoch
            eval_score = self.eval(model)
            logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
            wandb_logger.log({'okvqa': {'val_score': eval_score}})
            if eval_score > best_score:# and epoch == self.num_epochs -1:
                logger.info("New best evaluation score: {:.2f}".format(eval_score))
                best_score = eval_score
                best_model['epoch'] = epoch
                best_model['model'] = copy.deepcopy(model)

        return best_score, best_model

    def eval(self, model) -> float:

        '''
        Evaluates model on OKVQA validation set
        Returns validation OKVQA score
        '''
        model.eval()
        eval_score = 0

        for step, batch in enumerate(tqdm(self.okvqa_val_dataloader, desc='Evaluating on OKVQA val set')):
            output,_ = self.forward_pass(model, batch, do_eval=True)
            if self.args.dytox:
                logits = output['logits']
            else:
                logits = output[1]
            target = batch['target_scores'].to(self.device)
            answer_scores = self.compute_score_with_logits(logits, target)
            batch_scores = torch.sum(answer_scores, 1)
            eval_score += batch_scores.sum().item()

        eval_score = eval_score/len(self.okvqa_val_dataloader.dataset)*100.0

        model.train()
        return eval_score

    def eval_forgetting(self, model, model_path: str) -> float:

        '''
        Evaluates forgetting by loading model weights from model_path, 
        which has encoder weights of later task and classifier weights from OKVQA
        Returns OKVQA evaluation score of post-OKVQA model checkpoint
        '''

        model.to(self.device)

        # Load model with encoder weights from encoder_path, and classifier weights from model_path
        #model.load_state_dict(torch.load(model_path))
        logger.info("Loaded model checkpoint from {}".format(model_path))

        return self.eval(model)

class LowShotOKVQATrainer(OKVQATrainer):

    def __init__(self,
                 args: argparse.Namespace, 
                 task_configs: Dict, 
                 model_config: Dict, 
                 low_shot_config: Dict = None):

        '''
        Creates instance of low-shot OKVQA trainer according to low_shot_config
        
        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        low_shot_config: dictionary containing low-shot configuration parameters
        '''

        super(LowShotOKVQATrainer, self).__init__(args, task_configs, model_config)
        self.low_shot_config = low_shot_config
        self.eval_epochs = [x-1 for x in low_shot_config['eval_epochs']]

        self.okvqa_train_dataloader.dataset.convert_to_low_shot(low_shot_percentage=low_shot_config['percentage'])
        self.max_steps = len(self.okvqa_train_dataloader) * self.num_epochs

    def train(self, model):
        '''
        Trains model on low-shot OKVQA task
        Args:
        model

        Returns:
        best_score: Best validation OKVQA score
        best_model: Model checkpoint of best validation epoch
        '''

        model.to(self.device)

        # Create optimizer
        optimizer = model.create_optimizer(self.hparams)
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
            for step, batch in enumerate(tqdm(self.okvqa_train_dataloader, desc='Training epoch {}'.format(epoch+1))):

                loss, output, _, _ = self.train_step(model, batch, optimizer, scheduler)

            if epoch in self.eval_epochs:
                # Do evaluation after epoch
                eval_score = self.eval(model)
                logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
                wandb_logger.log({'okvqa': {'val_score': eval_score}})
                if eval_score > best_score:
                    logger.info("New best evaluation score: {:.2f}".format(eval_score))
                    best_score = eval_score
                    best_model['epoch'] = epoch
                    best_model['model'] = copy.deepcopy(model)
        return best_score, best_model