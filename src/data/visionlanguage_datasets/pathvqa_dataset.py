
import sys
import os
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

from data.image_collation import image_collate

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class PathVQADataset(Dataset):

    def __init__(self, 
                 data_dir: str, 
                 split: str, 
                 **kwargs):

        """
        Initiates the PathVQADataset - loads all the questions (and converts to input IDs using the tokenizer, if provided) 
        and answers (including converting each to a numeric label, and a score based on occurence from annotators)
        Every item in self.data corresponds to a single QA pair, with a corresponding image

        Args:
        data_dir : path containing PathVQA questions and annotations. Also contains mapping from each answer in set of possible answers to a numerical label
        split: either train/val split

        Returns:
        Loads all annotations into self.data, where each item is a single PathVQA pair
        """

        self.data_dir = data_dir
        self.split = split
        self.tokenizer = kwargs['tokenizer'] if 'tokenizer' in kwargs else None
        self.images_dir = os.path.join(data_dir, 'split', 'images', split)

        self.annotations_file = os.path.join(data_dir, 'split/qas/{}_vqa.pkl'.format(split))
        self.ans2label_file = os.path.join(data_dir, 'split/qas/ans2label.pkl'.format(split))

        # Load mapping from answers to labels
        self.ans2label = pkl.load(open(self.ans2label_file, 'rb'))
        self.label2ans = {v: k for k, v in self.ans2label.items()}
        self.num_labels = len(self.label2ans)
        self.num_answers = len(self.ans2label)

        self.cached_data_file = os.path.join(data_dir, 'cached_pathvqa_data', 'pathvqa_{}.pkl'.format(split))
        if False: #os.path.exists(self.cached_data_file):
            # Load cached data
            self.data = pkl.load(open(self.cached_data_file, 'rb'))

        else:
            # Create data for each annotation
            annotations = pkl.load(open(self.annotations_file, 'rb'))
            self.data = []
            for anno in annotations:

                #prob = random.randint(1,100)
                #if prob > 5:
                #    continue

                qid = anno['question_id']
                image_id = anno['img_id']

                # Retrieve the question for this annotation
                question = anno['sent']
                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenize(question)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = []
                    input_ids = []

                # Map from each crowdsourced answer to occurrences in annotation
                answer = list(anno['label'].keys())[0]
                label = self.ans2label[answer]
                img_filename = os.path.join(self.images_dir, '{}.jpg'.format(image_id))

                # Store pre-processed example
                example = {'question_id': qid,
                            'image_id': image_id,
                            'question': question,
                            'question_input_ids': input_ids,
                            'label': label,
                            'answer': answer,
                            'img_filename': img_filename,
                            }
                self.data.append(example)

            pkl.dump(self.data, open(self.cached_data_file, 'wb'))

        self.n_examples = len(self.data)
        self.pil_transform = T.Resize(size=384, max_size=640)


        logger.info("Loaded PathVQA {} dataset, with {} examples".format(self.split, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do PathVQA

        """

        example = self.data[index]
        question_id = example['question_id']

        # Tokenize the input question 
        question = example['question']
        input_ids = example['question_input_ids']

        # Get the image tensor from ImageDataset
        image_id = example['image_id']
        image = Image.open(example['img_filename'])
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = self.pil_transform(image)

        label = example['label']

        return {'question': question, 
                'input_ids': input_ids, 
                'image': image, 
                'label': label, 
                'question_id': question_id
                }

    def convert_to_low_shot(self, low_shot_percentage: float):
        """
        Args:
        low_shot_percentage: float between 0 and 1, telling what % of full data to retain for low-shot setting
        """

        assert self.split == 'train'
        logger.info("Converting PathVQA train split into low-shot dataset, with {:.2f}% training samples...".format(low_shot_percentage*100.0))
        n_low_shot_examples = int(low_shot_percentage*self.n_examples)

        new_data = random.sample(self.data, n_low_shot_examples)
        self.data = new_data
        self.n_examples = len(self.data)

        logger.info("Converted into low-shot dataset, with {} examples".format(self.n_examples))

def pathvqa_batch_collate(batch: List[Dict], 
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
    batch_labels = torch.LongTensor([x['label'] for x in batch])

    # Depending on the visual_input_type variable, process the images accordingly
    images = [x['image'] for x in batch]
    images = image_collate(images, visual_input_type)

    return {'raw_texts': questions,
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'images': images,
            'labels': batch_labels}

def build_pathvqa_dataloader(args, 
                         data_dir: str, 
                         split: str, 
                         visual_input_type: str,
                         **kwargs) -> torch.utils.data.DataLoader:

    """
    Creates the PathVQA Dataloader, which gives batches of PathVQA inputs and outputs

    Args:
    data_dir : path containing PathVQA questions and annotations.
    split: either train/val split
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """

    batch_size = args.batch_size
    shuffle = True if split == 'train' else False

    logger.info("Creating PathVQA {} dataloader with batch size of {}".format(split, batch_size))

    dataset = PathVQADataset(data_dir, split, **kwargs)
    num_labels = dataset.num_labels
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: pathvqa_batch_collate(x, visual_input_type))
    return dataloader

if __name__ == '__main__':
    data_dir = '/data/datasets/MCL/pathvqa/'
    #dataset = PathVQADataset(data_dir, None, 'train', None)
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.shuffle = True
            self.num_workers = 2
            self.visual_input_type = 'pil-image'
    args = Args()

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    pathvqa_dataloader = build_pathvqa_dataloader(args, data_dir, 'val', args.visual_input_type, tokenizer=tokenizer)

    for batch in pathvqa_dataloader:
        pdb.set_trace() 
