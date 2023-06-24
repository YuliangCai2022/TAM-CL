
import sys
import os
from os import listdir
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl
import pdb
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from PIL import Image
from image_utils import resize_image
from gqa_utils import create_gqa_labels, target_tensor

from data.image_datasets.gqaimages_dataset import GQAImagesDataset
from data.image_collation import image_collate

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class GQADataset(Dataset):

    def __init__(self, 
                 data_dir: str, 
                 images_dataset: GQAImagesDataset, 
                 split: str,
                 **kwargs):

        """
        Initiates the GQAdataset - loads all the questions and answers (including converting each to a numeric label, and a score based on occurence from annotators)
        Every item in self.data corresponds to a single QA pair, with a corresponding image

        Args:
        data_dir : path containing GQA questions and annotations. 
        images_dataset : instance of GQAImagesDataset, that is used to retrieve the image for each question from GQA Dataset
        split: either train/val split

        Returns:
        Loads all annotations into self.data, where each item is a single GQA pair
        """

        self.images_dataset = images_dataset
        self.data_dir = data_dir #set in task_configs: 'gqa/questions'
        self.split = split
        self.tokenizer = kwargs['tokenizer'] if 'tokenizer' in kwargs else None

        # if split == 'val':
        #     self.question_path = self.data_dir
        #     self.questions_file_list = [os.path.join(data_dir, 'val_all_questions.json')]
        # else: # train -- folder contain 10 json files
        #     self.question_path = os.path.join(data_dir, 'train_all_questions')                
        #     if not small and not superSmall:
        #         self.questions_file_list = listdir(os.path.join(data_dir, 'train_all_questions'))
        #     else:
        #         self.questions_file_list = [os.path.join('train_all_questions_0.json')]
        self.question_file_path = os.path.join(data_dir, '{}_balanced_questions.json'.format(split))
        
        # path for cached data
        self.cached_data_file = os.path.join(data_dir, 'cached_gqa_data', 'gqa_{}_data.pkl'.format(split))
        self.cached_label_file = os.path.join(data_dir, 'cached_gqa_data', 'gqa_answer2label.pkl')

        # Load cached answer2label if exists else create
        if not os.path.exists(self.cached_label_file):
            create_gqa_labels(self.data_dir)
        with open(self.cached_label_file, 'rb') as f:
            self.answer2label = pkl.load(f)
        self.num_labels = len(self.answer2label)
        logger.info("Loaded GQA answer2label, with {} labels".format(self.num_labels))
        
        if os.path.exists(self.cached_data_file):
            # Load cached data
            with open(self.cached_data_file, 'rb') as f:
                self.data = pkl.load(f)
        else:
            # Create all data
            self.data = []
            # Load each question JSON file
            # TODO: change path and no iteration then
            # for question_file in self.questions_file_list:
            curr_question_file = json.load(open(self.question_file_path))
            i = 0
            for question_id, question_object in tqdm(curr_question_file.items()):
                if split == "train":
                    if i == 80000:
                        break
                #if split == "val":
                #    if i == 10000:
                #        break
                i += 1
                image_id = question_object['imageId']
                question = question_object['question']
                # tokenize question input
                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenize(question)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = []
                    input_ids = []
                answer = question_object['answer']
                labels = self.answer2label[answer]
                full_answer = question_object['fullAnswer']
                # need to convert answer to label (int) after getting all possible answers
                # Store pre-processed example
                example = {
                    'question_id':question_id,
                    'image_id':image_id,
                    'question':question,
                    'question_input_ids': input_ids,
                    'answer':answer,
                    'labels':labels,
                    'full_answer:':full_answer
                }
                self.data.append(example)

            # save data and answer2label to file
            pkl.dump(self.data, open(self.cached_data_file, 'wb'))
            

        self.n_examples = len(self.data)
        logger.info("Loaded GQA {} dataset, with {} examples".format(self.split, self.n_examples))
        logger.info("Loaded GQA answer2label dict, with {} answers".format(len(self.answer2label)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do GQA

        """

        example = self.data[index]
        question_id = example['question_id']

        # Tokenize the input question 
        question = example['question']
        input_ids = example['question_input_ids']

        # Get the image tensor from ImageDataset
        image_id = example['image_id']
        image = self.images_dataset.get_image_data(image_id)

        # Get labels for question
        labels = example['labels']
        target_scores = target_tensor(self.num_labels, labels)

        return {'question': question, 
                'input_ids': input_ids, 
                'image': image, 
                'labels': labels, 
                'target_scores': target_scores, 
                'question_id': question_id
                }

    def convert_to_low_shot(self, low_shot_percentage: float):
        """
        Args:
        low_shot_percentage: float between 0 and 1, telling what % of full data to retain for low-shot setting
        """

        assert self.split == 'train'
        logger.info("Converting GQA train split into low-shot dataset, with {:.2f}% training samples...".format(low_shot_percentage*100.0))
        n_low_shot_examples = int(low_shot_percentage*self.n_examples)

        new_data = random.sample(self.data, n_low_shot_examples)
        self.data = new_data
        self.n_examples = len(self.data)

        logger.info("Converted into low-shot dataset, with {} examples".format(self.n_examples))

def gqa_batch_collate(batch: List[Dict], 
                      visual_input_type: str):

    """
    Collates each model input for all batch items into a single model input (e.g. converts a list of input_ids into a matrix of size (batch_size, max_len))

    Args:
    batch - list of batch items, each item being a dictionary returned by Dataset's __getitem__ method
    visual_input_type: string which specifies the type of visual input

    Returns:
    Dictionary containing batched inputs and outputs
    """

    pad_token = 0   # tokenizer.pad_token_id

    # Pad the text inputs
    questions = [x['question'] for x in batch]
    input_ids = [x['input_ids'] for x in batch]
    max_len = max([len(x) for x in input_ids])
    input_ids_padded = []
    attn_masks = []
    for i in range(len(input_ids)):
        ids_padded = input_ids[i] + [pad_token]*(max_len - len(input_ids[i]))
        attn_mask = [1]*len(input_ids[i]) + [0]*(max_len - len(input_ids[i]))

        input_ids_padded.append(ids_padded)
        attn_masks.append(attn_mask)
    input_ids = torch.tensor(input_ids_padded, dtype=torch.long)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long)

    # Stack the target tensors
    batch_labels = [x['labels'] for x in batch]
    batch_scores = [x['target_scores'] for x in batch]
    batch_scores = torch.stack(batch_scores, dim=0)

    # Depending on the visual_input_type variable, process the images accordingly
    images = [x['image'] for x in batch]
    images = image_collate(images, visual_input_type)

    return {'raw_texts': questions,
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'images': images,
            'target_scores': batch_scores,
            'labels': batch_labels}

def build_gqa_dataloader(args, 
                         data_dir: str, 
                         images_dataset: GQAImagesDataset, 
                         split: str, 
                         visual_input_type: str,
                         **kwargs) -> torch.utils.data.DataLoader:

    """
    Creates the GQA Dataloader, which gives batches of GQA inputs and outputs

    Args:
    data_dir : path containing GQA questions and annotations.
    images_dataset : instance of GQAImagesDataset, that is used to retrieve the images for each question from GQA Dataset
    split: either train/val split
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """

    batch_size = args.batch_size
    shuffle = True if split == 'train' else False

    logger.info("Creating GQAv2 {} dataloader with batch size of {}".format(split, batch_size))

    dataset = GQADataset(data_dir, images_dataset, split, **kwargs)
    num_labels = dataset.num_labels
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: gqa_batch_collate(x, visual_input_type))
    return dataloader

if __name__ == '__main__':
    data_dir = '/home/shared/MCL/gqa/questions'
    #dataset = VQADataset(data_dir, None, 'train', None)
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.shuffle = True
            self.num_workers = 2
            self.visual_input_type = 'pil-image'
    args = Args()

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    images_dataset = GQAImagesDataset('/home/shared/MCL/gqa/images', args.visual_input_type)
    gqa_dataloader = build_gqa_dataloader(args, data_dir, images_dataset, 'val', args.visual_input_type, tokenizer=tokenizer)

    for batch in gqa_dataloader:
        pdb.set_trace() 
