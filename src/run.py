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
import yaml
import pdb

#sys.path.insert(0, '.')
sys.path.insert(0, '/project/rostamim_919/caiyulia/Multi-Dytox/src/')

import numpy as np
import torch
from tqdm import tqdm

#from transformers.adapters import AdapterConfig

#from modeling import load_encoder_map, create_continual_learner_map

#from cl_algorithms import ExperienceReplayMemory, EWC
from evaluate_cl_algorithm import upstream_knowledge_transfer_eval, catastrophic_forgetting_eval
from vilt import create_vilt_continual_learner_model
from model_configs import model_configs, ALLOWED_CL_ENCODERS
from task_configs import task_configs, SUPPORTED_VL_TASKS
#from configs.adapter_configs import ADAPTER_MAP
from wandb_config import wandb_config

from seed_utils import set_seed
from WandB import wandb_logger

logger = logging.getLogger(__name__)

device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")


def main():


    '''**************************************************************** Input parse *****************************************************************'''
    parser = argparse.ArgumentParser()

    ## Required parameters
    # encoder is always vilt
    parser.add_argument("--encoder_name", default=None, type=str, required=True, choices=ALLOWED_CL_ENCODERS,
                        help="The name of the base pretrained encoder.")

    # should be the pre-trained model weights
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True,
                        help="Name of pretrained model weights to load.")

    # can be either single task or multimask 
    parser.add_argument("--ordered_cl_tasks", type=str, required=True,
                        help="Ordered list of VL task keys for continual learning, seprated by commas.")

    # Dytox by defult, other choices will be added after finish Dytox, during test stage 
    parser.add_argument("--cl_algorithm", type=str, required=True, choices=['singletask_ft',
                                                                            'sequential_ft',
                                                                            'experience_replay',
                                                                            'ewc',
                                                                            'adapter',
                                                                            'freeze_encoder',
                                                                            'freeze_bottom_k_layers'],
                        help="Name of Continual Learning algorithm used.")

    # dir where datasets are stored
    parser.add_argument("--climb_data_dir", type=str, required=True, default='/data/datasets/MCL/',
                        help="Directory where all the MCL data is stored")

    # flag to do train
    parser.add_argument("--do_train", action='store_true',
                        help="If True, train the model on these tasks")

    parser.add_argument("--do_eval", action='store_true',
                        help="If True, evaluate the model on these tasks.")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Name of output directory, where all experiment results and checkpoints are saved.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")

    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for dataloader")

    parser.add_argument("--task_attention", type=int, default=1,
                        help="whether train with original vilt ,0, or with vilt + dytox , 1.")

    parser.add_argument("--do_wandb_logging", action='store_true',
                        help="Log experiments in W&B.")

    args = parser.parse_args()
    args.ordered_cl_tasks = args.ordered_cl_tasks.split(',')


    '''**************************************************** check if dataset load true ***********************************************'''
    if args.cl_algorithm == 'singletask_ft':
        assert len(args.ordered_cl_tasks) == 1

    for task_key in args.ordered_cl_tasks:
        assert task_key in SUPPORTED_VL_TASKS

    '''**************************************************** load CL vilt model ****************************************************'''
    model_config = model_configs[args.encoder_name]
    experiment_name = '{}-{}'.format(args.encoder_name, args.cl_algorithm)
    model = create_vilt_continual_learner_model(model_name_or_path=args.pretrained_model_name,
                                ordered_cl_tasks=args.ordered_cl_tasks, 
                                model_config=model_config, 
                                task_configs=task_configs,
                                device=device,
                                use_TAB=args.task_attention)
    # free the bottom layers
    model.vilt_encoder.freeze_bottom_k_layers(9)
    args.visual_input_type = model_config['visual_input_type']
    output_dir = os.path.join(args.output_dir, experiment_name)
    results_file = os.path.join(output_dir, 'results.json')
    '''*************************************************** train the model **************************************************************'''



    if args.do_train:

        # Create W&B experiment
        if args.do_wandb_logging:
            logger.info('W&B project: {}, experiment: {}'.format(wandb_config['project_name'], experiment_name))
            wandb_logger.initialize(wandb_config=wandb_config,
                                    experiment_name=experiment_name)

        # initialize teacher model
        teacher_model = None

        # if on-going experiment exists, load the old result
        results = []
        if os.path.isfile(results_file):
            results = json.load(open(results_file))
            logger.info("-"*100)
            logger.info("Cached results:")
            for i, r in enumerate(results):
                task_key = r['task_key']
                best_score = r['best_score']
                logger.info("Task #{}: {} - best score = {:.2f}".format(i+1, task_configs[task_key]['task_name'], best_score))
        task_trainers = {}

        # Begin training on VL tasks sequentially
        logger.info("-"*100)
        logger.info("Training models on Vision-Language continual learning tasks...")

        num_task = 0

        for task_num, task_key in enumerate(args.ordered_cl_tasks):
            num_task += 1
            logger.info("-"*100)
            task_name = task_configs[task_key]['task_name']
            task_output_dir = os.path.join(output_dir, 'checkpoints', 'task{}_{}'.format(task_num, task_key))

            # freeze the task_token for other tasks
            for key in model.token_dict:
                if key != task_key:
                    model.token_dict[key].requires_grad=False
                else:
                    model.token_dict[key].requires_grad=True
                    logger.info("********************** found the task token with same task key! *****************************")

            if os.path.isfile(os.path.join(task_output_dir, 'model')):

                # If we find model checkpoint for this task, load the checkpoint and move onto next CL task
                logger.info("Found checkpoint for task {}!".format(task_name))
                try:
                    model.load_state_dict(torch.load(os.path.join(task_output_dir, 'model')))
                except Exception as e:
                    ckpt_state_dict = torch.load(os.path.join(task_output_dir, 'model'))
                    initialized = {k: False for k in model.state_dict().keys()}
                    for k in ckpt_state_dict.keys():
                        model.state_dict()[k].copy_(ckpt_state_dict[k])
                        initialized[k] = True
                    logger.info("Uninitialized keys: {}".format(','.join([k for k in initialized.keys() if initialized[k] is False])))
                    torch.save(model.state_dict(), os.path.join(task_output_dir, 'model'))
                    logger.info("Saved model with uninitialized keys as new checkpoint")
                logger.info("Loaded model checkpoint from task {}! Moving on to next task...".format(task_name))
                model.teacher_model = copy.deepcopy(model)

                task_trainer_class = task_configs[task_key]['task_trainer']
                task_trainer = task_trainer_class(args, task_configs, model_config, device, teacher_model, num_task)

            else:

                #Create the Trainer method for the current CL task, and call the train method
                logger.info("Training {} model on task #{}: {}".format(args.encoder_name, task_num+1, task_name))
                task_trainer_class = task_configs[task_key]['task_trainer']
                task_trainer = task_trainer_class(args, task_configs, model_config, device, teacher_model, num_task)
                best_eval_score, best_model = task_trainer.train(model,
                                                                replay_memory=None,
                                                                ewc=None)
                logger.info("Best {} evaluation score = {:.2f}, after epoch {}".format(task_name, best_eval_score, best_model['epoch']+1))

                # Save best model checkpoint, and separately save the models' Encoder object
                logger.info("Saving best model and encoder checkpoint after {} training".format(task_name))
                if not os.path.isdir(task_output_dir):
                    os.makedirs(task_output_dir)
                best_task_model = best_model['model']
                torch.save(best_task_model.state_dict(), os.path.join(task_output_dir, 'model'))
                torch.save(best_task_model.get_encoder().state_dict(), os.path.join(task_output_dir, 'encoder'))
                logger.info("Saved checkpoint!")


                # Save CL results so far
                task_results = {
                    'task_num': task_num,
                    'task_key': task_key,
                    'best_score': best_eval_score,
                    'best_epoch': best_model['epoch']
                }
                results.append(task_results)
                json.dump(results, open(results_file, 'w'))
                logger.info("Saved continual learning results so far!")

                teacher_model = copy.deepcopy(model)
                teacher_model.vilt_encoder.freeze_all_weights()
                model.teacher_model = teacher_model
                #for param in teacher_model.TAB:
                #    param.requires_grad = False
            task_trainers[task_key] = task_trainer


    '''****************************************************************** do evaluation ******************************************************************'''
    if args.do_eval:

        logger.info("-"*100)

        # Forward transfer from continual learning, by comparing to single-task finetuning score
        logger.info("Evaluating FORWARD TRANSFER of {} model on {}".format(args.encoder_name, ' -> '.join(args.ordered_cl_tasks)))
        upstream_knowledge_dict = upstream_knowledge_transfer_eval(args, results_file)
        average_relative_gain = sum(list([x['relative_gain'] for x in upstream_knowledge_dict.values()]))/len(upstream_knowledge_dict)
        logger.info("Average forward transfer gain = {:.2f}%".format(average_relative_gain))

        logger.info("-"*100)

        # Forgetting evaluation
        if not args.do_train:
            logger.info("Creating task trainers for forgetting evaluation...")
            task_trainers = {}
            for task_num, task_key in enumerate(args.ordered_cl_tasks):
                task_trainer_class = task_configs[task_key]['task_trainer']
                task_trainer = task_trainer_class(args, task_configs, model_config, device)
                task_trainers[task_key] = task_trainer
        else:
            for task_num, task_key in enumerate(args.ordered_cl_tasks):
                assert task_key in task_trainers.keys()

        # Run the forgetting evaluation, and save results to file
        logger.info("Evaluating CATASTROPHIC FORGETTING of {} model on {}".format(args.encoder_name, ' -> '.join(args.ordered_cl_tasks)))
        catastrophic_forgetting_dict = catastrophic_forgetting_eval(args, results_file, model, task_trainers)

        eval_results_file = os.path.join(output_dir, 'eval_results.json')
        eval_results = {'upstream_knowledge_transfer': upstream_knowledge_dict,
                        'forgetting': catastrophic_forgetting_dict}
        json.dump(eval_results, open(eval_results_file, 'w'))

        logger.info("-"*100)

if __name__ == '__main__':
    main()