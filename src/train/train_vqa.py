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
from torch import functional as F
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from data.visionlanguage_datasets.vqa_dataset import build_vqa_dataloader
from train.task_trainer import TaskTrainer
from WandB import wandb_logger

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class VQATrainer(TaskTrainer):

    def __init__(self, 
                 args: argparse.Namespace, 
                 task_configs: Dict, 
                 model_config: Dict, 
                 device: torch.device):

        '''
        Initializes a Trainer that handles training of a model on the VQA task

        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        '''

        super().__init__()

        self.args = args
        self.device = device

        self.vqa_config = task_configs['vqa']
        self.data_dir = os.path.join(args.climb_data_dir, self.vqa_config['data_dir'])

        # Model-specific stuff
        self.visual_input_type = model_config['visual_input_type']
        self.batch2inputs_converter = model_config['batch2inputs_converter']

        # Load COCO Images dataset for image data backbone
        images_source = self.vqa_config['images_source']
        mscoco_config = task_configs[images_source]
        self.images_dataset = MSCOCOImagesDataset(coco_dir=os.path.join(args.climb_data_dir, mscoco_config['data_dir']),
                                                  visual_input_type=args.visual_input_type)

        # Create dataloaders for training and validation
        self.vqa_train_dataloader = build_vqa_dataloader(args=args,
                                                    data_dir=self.data_dir,
                                                    images_dataset=self.images_dataset,
                                                    split='train',
                                                    visual_input_type=self.visual_input_type)

        self.vqa_val_dataloader = build_vqa_dataloader(args=args,
                                                  data_dir=self.data_dir,
                                                  images_dataset=self.images_dataset,
                                                  split='val',
                                                  visual_input_type=self.visual_input_type)

        # Training hyperparameters
        self.num_epochs = self.vqa_config['num_epochs']
        self.lr = self.vqa_config['lr']
        self.adam_epsilon = self.vqa_config['adam_epsilon']
        self.weight_decay = self.vqa_config['weight_decay']
        self.loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.max_steps = len(self.vqa_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1 # TODO remove hard code

    def compute_score_with_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        Given logits for each answer in VQA classification, selects answer with max logit and returns VQA-score for that answer
        logits: logits for each answer - size=(batch_size, num_answers)
        labels: label for each answer in {0, 0.3, 0.6, 1} (batch_size, num_answers)
        
        Returns:
        scores: score of predicted answer (batch_size, num_answers)
        '''

        logits = torch.max(logits, 1)[1].data # argmax
        one_hots = torch.zeros(*labels.size()).to(self.device)

        #logger.info(logits)
        #print("logits shape is " + str(logits.shape))
        #print("one_hots shape is " + str(one_hots.shape))
        # modified by Yuliang, previous logits.view(-1,1)
        if self.args.task_attention:
            one_hots.scatter_(1, logits.view(-1,1), 1)
        else:
            one_hots.scatter_(1, logits.view(-1,1), 1)
        scores = (one_hots * labels)
        return scores
    
    def get_train_dataloader(self):
        return self.vqa_train_dataloader

    def get_collate_fn(self):
        return self.vqa_train_dataloader.collate_fn

    def forward_pass(self, model, batch: Dict, do_eval: bool = False) -> tuple:
        '''
        Forward pass of batch inputs through model
        output: tuple containing (encoder_pooled_output, output_logits)
        '''

        inputs = self.batch2inputs_converter(batch)
        if do_eval is True:
            with torch.no_grad():
                output = model(task_key='vqa', **inputs)
        else:
            output = model(task_key='vqa', **inputs)
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
        logits = None
        #modiefied by Yuliang, ignore the task token in this loss
        if self.args.task_attention:
            logits = output[1].reshape(-1,3129)
            logits = logits[1:,:] # logtis size 32,3129
        else:
            logits = output[1]


        target = batch['target_scores'].to(self.device)
        loss = self.loss_criterion(logits, target) * target.shape[1]

        #add by Yuliang call this only when have multitask
        # teacher model = model of previous task
        #kd_loss = 0
        #_kd_loss = F.kl_div(
        #            F.log_softmax(logits / tau, dim=1),
        #            F.log_softmax(main_output_old / tau, dim=1),
        #            reduction='mean',
        #            log_target=True
        #    ) * (tau ** 2)
        #kd_loss += args.kd * _kd_loss
        # add Dytox loss here

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
        Trains model on VQA task
        Args:
        model
        replay_memory: If experience replay is to be performed
        ewc: If EWC regularization loss is to be added

        Returns:
        best_score: Best validation VQA score
        best_model: Model checkpoint of best validation epoch
        '''
        model.to(self.device)
        if self.args.cl_algorithm == 'adapter':
            model.set_active_adapters("vqa")
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
            for step, batch in enumerate(tqdm(self.vqa_train_dataloader, desc='Training epoch {}'.format(epoch+1))):
                #print("batch size is " + str(batch.shape))
                loss, output, ewc_task, ewc_loss = self.train_step(model, batch, optimizer, scheduler, ewc)
              
                if self.args.cl_algorithm == 'experience_replay' and do_replay is True:
                    if (step + 1) % self.args.replay_frequency == 0:
                        sampled_replay_task = replay_memory.sample_replay_task()
                        replay_loss = replay_memory.run_replay_step(task_key=sampled_replay_task, model=model)

                if (step + 1) % wandb_logger.get_log_freq() == 0:
                    log_dict = {'vqa': {'loss': loss.item()}}
                    if ewc is not None and do_ewc is True:
                        log_dict[ewc_task] = {'ewc_loss': ewc_loss.item()}
                    wandb_logger.log(log_dict)

            #print(" the Q is " + str(model.TAB.attn.q.weight[0]))
            #print(" the token is " + str(model.task_token.data))
            # Do evaluation after epoch
            eval_score = self.eval(model)
            logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
            wandb_logger.log({'vqa': {'val_score': eval_score}})
            if eval_score > best_score:
                logger.info("New best evaluation score: {:.2f}".format(eval_score))
                best_score = eval_score
                best_model['epoch'] = epoch
                best_model['model'] = copy.deepcopy(model)

        return best_score, best_model

    def eval(self, model) -> float:

        '''
        Evaluates model on VQA validation set
        Returns validation VQA score
        '''
        model.eval()
        eval_score = 0
        for step, batch in enumerate(tqdm(self.vqa_val_dataloader, desc='Evaluating on VQA val set')):
            output = self.forward_pass(model, batch, do_eval=True)
            logits = output[1]
            # add by Yuliang
            if self.args.task_attention:
                logits = logits.reshape(-1,3129)
                logits = logits[1:,:]
            target = batch['target_scores'].to(self.device)
            answer_scores = self.compute_score_with_logits(logits, target)
            batch_scores = torch.sum(answer_scores, 1)
            eval_score += batch_scores.sum().item()
        eval_score = eval_score/len(self.vqa_val_dataloader.dataset)*100.0

        model.train()
        return eval_score

    def eval_forgetting(self, model, model_path: str) -> float:

        '''
        Evaluates forgetting by loading model weights from model_path, 
        which has encoder weights of later task and classifier weights from VQA
        Returns VQA evaluation score of post-VQA model checkpoint
        '''

        model.to(self.device)
        if self.args.cl_algorithm == 'adapter':
            model.set_active_adapters("vqa")

        # Load model with encoder weights from encoder_path, and classifier weights from model_path
        model.load_state_dict(torch.load(model_path))
        logger.info("Loaded model checkpoint from {}".format(model_path))

        return self.eval(model)

class LowShotVQATrainer(VQATrainer):

    def __init__(self,
                 args: argparse.Namespace, 
                 task_configs: Dict, 
                 model_config: Dict, 
                 device: torch.device, 
                 low_shot_config: Dict = None):

        '''
        Creates instance of low-shot VQA trainer according to low_shot_config
        
        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        low_shot_config: dictionary containing low-shot configuration parameters
        '''

        super(LowShotVQATrainer, self).__init__(args, task_configs, model_config, device)
        self.low_shot_config = low_shot_config
        self.eval_epochs = [x-1 for x in low_shot_config['eval_epochs']]

        self.vqa_train_dataloader.dataset.convert_to_low_shot(low_shot_percentage=low_shot_config['percentage'])
        self.max_steps = len(self.vqa_train_dataloader) * self.num_epochs

    def train(self, model):
        '''
        Trains model on low-shot VQA task
        Args:
        model

        Returns:
        best_score: Best validation VQA score
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
            for step, batch in enumerate(tqdm(self.vqa_train_dataloader, desc='Training epoch {}'.format(epoch+1))):

                loss, output, _, _ = self.train_step(model, batch, optimizer, scheduler)

            if epoch in self.eval_epochs:
                # Do evaluation after epoch
                eval_score = self.eval(model)
                logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
                wandb_logger.log({'vqa': {'val_score': eval_score}})
                if eval_score > best_score:
                    logger.info("New best evaluation score: {:.2f}".format(eval_score))
                    best_score = eval_score
                    best_model['epoch'] = epoch
                    best_model['model'] = copy.deepcopy(model)
        return best_score, best_model