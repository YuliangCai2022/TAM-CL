import os
import sys
import logging
import itertools
import pdb
import time
from PIL import Image
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, BertModel
from transformers import ViltConfig, ViltProcessor, ViltModel
from transformers import BertTokenizerFast
from transformers import logging as transformers_logging
from TAB import ClassAttention, Block

#from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from continual_learner import EncoderWrapper, ContinualLearner

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
transformers_logging.set_verbosity_error()

class ViltEncoderWrapper(EncoderWrapper):

    def __init__(self, 
                 processor: ViltProcessor, 
                 vilt: ViltModel, 
                 device: torch.device):
        '''
        Wrapper around Vilt model from huggingface library
        this is the class that gets saved during checkpointing for continual learning
        args:
        processor - instance of ViltProcessor
        vilt - instance of ViltModel class
        device - gpu/cuda
        '''

        super().__init__()
        self.processor = processor
        self.vilt = vilt
        self.device = device
        self.processor.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.max_text_length = self.vilt.config.max_position_embeddings
        self.encoder_dim = self.vilt.config.hidden_size

    def reset_processor(self, max_text_length: int, img_size: tuple):
        self.max_text_length = max_text_length
        self.processor.feature_extractor.size = img_size

    def reallocate_text_image(self, pretrained_pos_emb: torch.Tensor, max_len: int, img_size: int):
        '''
        Re-allocate some of the image token slots to the language position embeddings

        Args:
        pretrained_pos_emb: Pretrained position embeddings which are extended for creating extra language token slots
        max_len: new maximum length of language inputs (original is 40 for ViLT)
        img_size: new size of images, which requires fewer slots so that extra image slots can be allocated to extending language slots
        '''
        vilt_config = self.vilt.config
        assert max_len % vilt_config.max_position_embeddings == 0

        self.reset_processor(max_len, img_size)

        # copy the pretrained positional embeddings to support texts with longer max_len 
        extended_pos_emb = torch.cat([pretrained_pos_emb \
            for _ in range(0, max_len, vilt_config.max_position_embeddings)], 0)
        # extend & re-init Embedding
        self.vilt.embeddings.text_embeddings.position_embeddings = \
            nn.Embedding(max_len, vilt_config.hidden_size).from_pretrained(extended_pos_emb, freeze=False)

        # extend self.position_ids
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vilt/modeling_vilt.py#L274
        self.vilt.embeddings.text_embeddings.\
            register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def process_inputs(self, images: List, texts: List[str]) -> Dict:
        '''
        Returns encodings that can be inputted to the ViLT transformer

        Args:
        images - list of PIL Image objects
        texts - list of text strings

        Returns:
        encodings - dictionary, where each key corresponds to a different argument of the vilt's forward method
        '''
        encodings = self.processor(images=images, text=texts, max_length=self.max_text_length,
            padding=True, truncation=True, return_tensors='pt').to(self.device)
        return encodings

    def expand_modality_type_embeddings(self, type_vocab_size=3):
        '''
        ViLT contains only 2 token type embeddings - some tasks (like NLVR) require three token type embeddings
        Method makes a copy second token type embedding into a new third embedding parameter
        '''
        self.vilt.config.modality_type_vocab_size = type_vocab_size
        #https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/vilt_module.py#L85
        emb_data = self.vilt.embeddings.token_type_embeddings.weight.data
        self.vilt.embeddings.token_type_embeddings = nn.Embedding(type_vocab_size, self.encoder_dim)
        self.vilt.embeddings.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
        self.vilt.embeddings.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
        self.vilt.embeddings.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

    def forward(self, **encodings: Dict) -> torch.FloatTensor:
        '''
        Does forward pass of input encodings through ViltModel
        https://huggingface.co/docs/transformers/model_doc/vilt#transformers.ViltModel.forward

        Args:
        encodings: Dictionary containing inputs ViltModel's forward pass

        Returns:
        pooler_output: torch.FloatTensor of size (batch_size, hidden_size)
        '''

        output = self.vilt(**encodings)
        return output.pooler_output

    def freeze_all_weights(self):
        '''
        Freeze all parameters in self.vilt
        '''

        for p in self.vilt.parameters():
            p.requires_grad = False

    def freeze_bottom_k_layers(self, k: int):
        '''
        Freeze embedding parameters and bottom K transformer layer parameters
        '''

        assert k < len(self.vilt.encoder.layer)
        for p in self.vilt.embeddings.parameters():
            p.requires_grad = False
        for i in range(k):
            for p in self.vilt.encoder.layer[i].parameters():
                p.requires_grad = False


class ViltContinualLearner(ContinualLearner):

    def __init__(self, 
                 ordered_cl_tasks: List[str], 
                 encoder: ViltEncoderWrapper, 
                 encoder_dim: int, 
                 task_configs: Dict,
                 use_TAB: int):

        '''
        The actual Continual Learning model, that consists of a vision-language encoder and task-specific heads on top

        arguments:
        ordered_cl_tasks - list of CL task keys that will be encountered by the ContinualLearner
        encoder - instance of ViltEncoderWrapper class
        encoder_dim - output dimension of vilt encoder
        task_configs - dictionary which contains task-specific configurations/hparams for each task in ordered_cl_tasks
        '''

        super().__init__()
        self.encoder_dim = encoder_dim
        self.vilt_encoder = encoder
        #print(self.vilt_encoder)
        self.ordered_cl_tasks = ordered_cl_tasks
        self.task_configs = task_configs
        self.use_TAB = use_TAB
        self.cumu_label = 0
        # crreate task token
        device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
        #self.task_token.requires_grad = True
        self.task_layer_dict = {}
        for task_key in ordered_cl_tasks:
            self.add_task_layer(task_key, task_configs[task_key])
        self.task_layer = nn.ModuleDict(self.task_layer_dict)

        if 'nlvr2' in ordered_cl_tasks:
            self.vilt_encoder.expand_modality_type_embeddings()

        self.mid_features = {}
        for i in range(7,11):
            self.vilt_encoder.vilt.encoder.layer[i].output.register_forward_hook(get_activation('TAB'+str(i),self.mid_features))
        #trunc_normal_(self.task_token, std=.02)
        
        # should add a block instead of single task attention layer

        #self.token_dict = nn.ParameterDict()
        if use_TAB:
            # initialize task attention block
            self.TAB = Block(
                            dim=self.encoder_dim, num_heads=12, qkv_bias=False, qk_scale=None,
                            drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                            attention_type=ClassAttention
                        )

            #if len(self.ordered_cl_tasks) == 1:

            #    task_token = nn.Parameter(torch.zeros(1,encoder_dim))
            #    nn.init.normal_(task_token, std=0.2)
            #    self.token_dict[ordered_cl_tasks[0]] = task_token
            #else:
                # in a CL learning scenario, create a dictionary with key = task_key, and value = task_token
                #self.TAB_dict = {}
                #for task_key in ordered_cl_tasks:
                #    task_token = nn.Parameter(torch.zeros(1,encoder_dim))
                #    nn.init.normal_(task_token, std=0.2)
                #    self.token_dict[task_key] = task_token

        


    def add_task_layer(self, task_key: str, task_config: Dict):

        '''
        for a new task, add task-specific head according to its task_config parameters
        task_key - string which indicates which task to do forward pass for
        task_config - dictionary which contains hparams/other config parameters for that task
        '''

        num_labels = task_config['num_labels']
        if not self.use_TAB:
            self.cumu_label = 0
        if task_config['model_type'] == 'classification':
            num_images = task_config['num_images']
            clf_layer = nn.Sequential(
                            nn.Linear(self.encoder_dim*1, self.encoder_dim*2),
                            nn.LayerNorm(self.encoder_dim*2),
                            nn.GELU(),
                            nn.Linear(self.encoder_dim*2, num_labels + self.cumu_label)
                        ) # used to be *num_images
            self.task_layer_dict[task_key] = clf_layer
            self.cumu_label += num_labels

        elif task_config['model_type'] == 'multi-choice':
            clf_layer = nn.Sequential(
                            nn.Dropout(0.1),
                            nn.Linear(self.encoder_dim, 4 + self.cumu_label)
                        )
            self.task_layer_dict[task_key] = clf_layer
            self.cumu_label += num_labels

    def forward(self, task_key: str, images: List, texts: List[str], teacher_key = None):
        '''
        Does forward pass of image and text inputs through model, 
        depending on if the task is multichoice or classification with one or more images

        Args:
        task_key - string which indicates which task to do forward pass for
        images - list of PIL Image objects
        texts - list of text strings

        Returns:
        https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling
        '''

        # for now only modify the single image function for VQAV2
        task_config = None
        if teacher_key == None:
            task_config = self.task_configs[task_key]
        else:
            task_config = self.task_configs[teacher_key]
        if task_config['model_type'] == 'multi-choice':
            return self.forward_multi_choice(task_key, images, texts, task_config['num_choices'])
        elif task_config['model_type'] == 'classification':
            if task_config['num_images'] == 1:
                return self.forward_single_image(task_key, images, texts)
            else:
                return self.forward_multi_images(task_key, images, texts, task_config['num_images'])

    def forward_single_image(self, task_key: str, images: List, texts: List[str]) -> (torch.FloatTensor, torch.FloatTensor):
        '''
        Does forward pass of image and text inputs through model, 
        where every input has one image and one text

        Args:
        task_key - string which indicates which task to do forward pass for
        images - list of PIL Image objects
        texts - list of text strings

        Returns:
        encoder_output: https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling
        output_logits: logits for each output class (batch_size, num_labels)
        '''
       
        # VQAv2 is trained here
        #if self.use_TAB:
        #    token = self.token_dict[task_key]
        self.mid_features = {}

        encodings = self.vilt_encoder.process_inputs(images, texts)

        # extract intermediate SAB features
        #for i in range(7,11):
        #    self.vilt_encoder.vilt.encoder.layer[i].output.register_forward_hook(get_activation('TAB'+str(i),mid_features))
        

        encoder_output = self.vilt_encoder(**encodings)

        if self.use_TAB != 0:
            return encoder_output, self.mid_features
        #print("encoder_output size is " + str(encoder_output.shape))
        #if self.use_TAB:
        #    TAB_output,_,_ = self.TAB(torch.cat((token, encoder_output), dim=0).reshape(-1,1,768))
        #    output_logits = self.task_layer[task_key](TAB_output)
        #    return TAB_output, output_logits
        output_logits = self.task_layer[task_key](encoder_output)
        return encoder_output, output_logits

    def forward_multi_images(self, task_key: str, images: List[List], texts: List[str], num_images=2) -> (torch.FloatTensor, torch.FloatTensor):

        '''
        Does forward pass of image and text inputs through model, 
        where every input has multiple images and one text
        For tasks like NLVR2, do multiple text-image passes with each image and aggregate results

        Args:
        task_key - string which indicates which task to do forward pass for
        images - batch_size-sized list of num_images-sized list of PIL Image objects
        texts - list of text strings

        Returns:
        pooled_output: pooled feature Tensor of size (batch_size, num_images*hidden_size)
        output_logits: logits for each output class (batch_size, num_labels)
        '''
        # store the intermediate features 
        self.mid_features = {}
        combine_features = {}
        #for i in range(7,11):
        #    self.vilt_encoder.vilt.encoder.layer[i].output.register_forward_hook(get_activation('TAB'+str(i),mid_features))
        

        flat_images_list = list(itertools.chain(*images))
        encodings = self.vilt_encoder.process_inputs(flat_images_list, texts)

        input_ids, attention_mask, token_type_ids = \
            encodings['input_ids'], encodings['attention_mask'], encodings['token_type_ids']
        # reshape
        bs = len(input_ids)
        pixel_values = encodings['pixel_values'].view(bs, num_images, *encodings["pixel_values"].shape[-3:])
        pixel_mask = encodings['pixel_mask'].view(bs, num_images, *encodings["pixel_mask"].shape[-2:])

        # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/vilt/modeling_vilt.py#L1351
        pooler_outputs = []
        for i in range(num_images):
            # forward every image through the model
            encodings = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'pixel_values': pixel_values[:, i, :, :, :],
                'pixel_mask': pixel_mask[:, i, :, :],
                'image_token_type_idx': i + 1,
            }
            pooled_out = self.vilt_encoder(**encodings)

            # for the first time in the loop, add the mid features into the combine feature, 
            # if the feature key already exist, combine new one with the already existed feature
            for key in self.mid_features:
                if key not in combine_features:
                    combine_features[key] = self.mid_features[key]
                else:
                    combine_features[key] = torch.cat((combine_features[key],self.mid_features[key]),dim=1)
          
            pooler_outputs.append(pooled_out)
        pooled_output = torch.cat(pooler_outputs, dim=-1) # [bs, 1536]
        pooled_output_odd = pooled_output[:,1::2]
        pooled_output_even = pooled_output[:,::2]
        pooled_output = torch.mean(torch.stack([pooled_output_even,pooled_output_odd],dim=0),dim=0)

        if self.use_TAB != 0:
            return pooled_output, combine_features

        output_logits = self.task_layer[task_key](pooled_output)
        return pooled_output, output_logits

    def forward_multi_choice(self, task_key: str, images: List, texts: List[List[str]], num_choices) -> (torch.FloatTensor, torch.FloatTensor):

        '''
        Does forward pass of image and text inputs through model, 
        where every input has one image and multiple text choices
        For tasks like VCR, do multiple text-image passes with each text, and select the text with highest score

        Args:
        task_key - string which indicates which task to do forward pass for
        images - batch_size-sized list of num_images-sized list of PIL Image objects
        texts - list of text strings

        Returns:
        pooled_output: pooled feature Tensor of size (batch_size, num_images*hidden_size)
        output_logits: logits for each output class (batch_size, num_labels)
        '''

        texts_list = list(itertools.chain(*texts))
        encodings = self.vilt_encoder.process_inputs(images, texts_list)
        bs = len(images)
        unflat_input_ids = encodings['input_ids'].view(bs, num_choices, -1)
        unflat_attention_mask = encodings['attention_mask'].view(bs, num_choices, -1)
        unflat_token_type_ids = encodings['token_type_ids'].view(bs, num_choices, -1)
        pixel_values, pixel_mask = encodings['pixel_values'], encodings['pixel_mask']

        pooler_outputs = []
        for i in range(num_choices):
            # Forward every choice through the model
            encodings = {
                'input_ids': unflat_input_ids[:, i, :],
                'attention_mask': unflat_attention_mask[:, i, :],
                'token_type_ids': unflat_token_type_ids[:, i, :],
                'pixel_values': pixel_values,
                'pixel_mask': pixel_mask
            }
            pooled_out = self.vilt_encoder(**encodings)
            pooler_outputs.append(pooled_out)
        #pooled_output = torch.cat(pooler_outputs, dim=-1) # [bs, 1536]
        pooled_output = torch.stack(pooler_outputs, dim=0).transpose(0, 1)

        if self.use_TAB != 0:
            return pooled_output


        output_logits = self.task_layer[task_key](pooled_output).squeeze()

        return pooled_output, output_logits

    def get_encoder(self):

        return self.vilt_encoder

    ############## Adapter-specific methods ##############
    def add_adapter(self, task_key: str, config: Dict):
        self.vilt_encoder.vilt.add_adapter(task_key, config)

    def train_adapter(self, task_key: str):
        self.vilt_encoder.vilt.train_adapter(task_key)

    def set_active_adapters(self, task_key: str):
        self.vilt_encoder.vilt.set_active_adapters(task_key)


def get_activation(name,holder):
    def hook(model, input, output):
        holder[name] = output.to("cuda:0")
    return hook

class ViltForImageClassification(nn.Module):

    def __init__(self, encoder: ViltEncoderWrapper, encoder_dim: int, num_labels: int):
        '''
        Modified ViLT model for image classification tasks
        args:
        encoder - instance of ViltEncoderWrapper class
        encoder_dim - output dimension of vilt encoder
        num_labels - number of output labels for image classification task
        '''

        super().__init__()
        self.encoder_dim = encoder_dim
        self.vilt_encoder = encoder
        self.clf_layer = nn.Sequential(
                            nn.Linear(encoder_dim, encoder_dim*2),
                            nn.LayerNorm(encoder_dim*2),
                            nn.GELU(),
                            nn.Linear(encoder_dim*2, num_labels)
                        )

    def forward(self, images: List, texts: List[str]) -> torch.FloatTensor:
        '''
        Does forward pass of image and text inputs through model, where texts are dummy texts

        Args:
        images - batch_size-sized list of num_images-sized list of PIL Image objects
        texts - list of dummy text strings
        '''
        encodings = self.vilt_encoder.process_inputs(images, texts)
        encoder_output = self.vilt_encoder(**encodings)

        output_logits = self.clf_layer(encoder_output)
        return output_logits


class ViltForSequenceClassification(nn.Module):

    def __init__(self, encoder: ViltEncoderWrapper, encoder_dim: int, num_labels: int):
        '''
        Modified ViLT model for text classification tasks

        Args:
        encoder - instance of ViltEncoderWrapper class
        encoder_dim - output dimension of vilt encoder
        num_labels - number of output labels for text classification task
        '''

        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder = encoder
        self.clf_layer = nn.Sequential(
                            nn.Linear(encoder_dim, encoder_dim*2),
                            nn.LayerNorm(encoder_dim*2),
                            nn.GELU(),
                            nn.Linear(encoder_dim*2, num_labels)
                        )

    def forward(self, images: List, texts: List[str]) -> torch.FloatTensor:
        '''
        Does forward pass of image and text inputs through model, where image is averaged image

        Args:
        images - batch_size-sized list of "average image"'sPIL Image objects
        texts - list of text strings
        '''

        encodings = self.encoder.process_inputs(images, texts)
        # expand to batch size
        bs = len(encodings['input_ids'])
        encodings['pixel_values'] = encodings['pixel_values'].expand([bs, *encodings['pixel_values'].shape[1:]])
        encodings['pixel_mask'] = encodings['pixel_mask'].expand([bs, *encodings['pixel_mask'].shape[1:]])
        encoder_output = self.encoder(**encodings)

        output_logits = self.clf_layer(encoder_output)
        return output_logits


class ViltForMultipleChoice(nn.Module):

    def __init__(self, encoder: ViltEncoderWrapper, encoder_dim: int, num_labels: int):
        '''
        Modified ViLT model for text multiple-choice tasks
        Args:
        encoder - instance of ViltEncoderWrapper class
        encoder_dim - output dimension of vilt encoder
        num_labels - number of choices for multi-choice task
        '''

        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_labels = num_labels
        self.encoder = encoder
        self.clf_layer = nn.Sequential(
                            nn.Dropout(0.1),
                            nn.Linear(encoder_dim, 1)
                        )

    def forward(self, images, texts):
        encodings = self.encoder.process_inputs(images, texts)
        # unflat_input_ids = encodings['input_ids'].view(self.num_labels, 32, -1).transpose(0, 1)
        bs = len(encodings['input_ids'])
        encodings['pixel_values'] = encodings['pixel_values'].expand([bs, *encodings['pixel_values'].shape[1:]])
        encodings['pixel_mask'] = encodings['pixel_mask'].expand([bs, *encodings['pixel_mask'].shape[1:]])
        encoder_output = self.encoder(**encodings)
        reshape_output = encoder_output.view(self.num_labels, -1, self.encoder_dim).transpose(0, 1).contiguous()

        output_logits = self.clf_layer(reshape_output).squeeze()
        return output_logits


def load_vilt_encoder(checkpoint_name: str, device: torch.device, pretrained_vilt_name: str) -> ViltEncoderWrapper:

    '''
    Method to load ViltEncoder wrapper, around specified pre-trained vilt 

    args:
    checkpoint_name: name of ViltEncoder checkpoint to load encoder from
    device: torch.device
    pretrained_vilt_name: pretrained vilt name for processor/config

    returns:
    vilt_encoder: ViltEncoderWrapper initialized with checkpoint
    '''

    logger.info("-"*100)
    logger.info("Loading ViLT encoder model: {}".format(checkpoint_name))
    vilt_processor = ViltProcessor.from_pretrained(pretrained_vilt_name)

    if checkpoint_name == pretrained_vilt_name: # load pretrained encoder
        vilt = ViltModel.from_pretrained(pretrained_vilt_name)
        vilt_encoder = ViltEncoderWrapper(vilt_processor, vilt, device)

    else: # load pre-finetuned encoder
        config = ViltConfig.from_pretrained(pretrained_vilt_name)
        vilt = ViltModel(config) # random init.
        vilt_encoder = ViltEncoderWrapper(vilt_processor, vilt, device)
        if 'nlvr2' in checkpoint_name:
            vilt_encoder.expand_modality_type_embeddings()

        ckpt = torch.load(checkpoint_name)
        vilt_encoder.load_state_dict(ckpt) # loaded

    logger.info("Successfully loaded pretrained ViLT encoder")
    return vilt_encoder

def create_vilt_continual_learner_model(model_name_or_path: str, 
                                       ordered_cl_tasks: List[str],
                                       model_config: Dict, 
                                       task_configs: Dict,
                                       device: torch.device,
                                       use_TAB: int):

    '''
    Creates an instance of ViltContinualLearner, with the encoder initialized from model_name_or_path

    Args:
    model_name_or_path: Name/path of model to load encoder checkpoint from
    ordered_cl_tasks: List of task_keys to do continual learning on
    model_config: Dictionary containing ViLT model configuration
    task_configs: Dictionary containing task-specific configurations for the CL tasks
    device: cpu/cuda

    Returns:
    cl_model: instance of ViltContinualLearner
    '''

    if use_TAB:
        print("******************************creating vilt moel with task attention block!*****************************")
    else:
        print("****************************** not using task attention block! **************************************")
    encoder = load_vilt_encoder(checkpoint_name=model_name_or_path, 
                                device=device, 
                                pretrained_vilt_name=model_name_or_path)

    cl_model = ViltContinualLearner(ordered_cl_tasks=ordered_cl_tasks, 
                                    encoder=encoder, 
                                    encoder_dim=model_config['encoder_dim'], 
                                    task_configs=task_configs,
                                    use_TAB = use_TAB)
    logger.info("Successfully created and initialized ViLT Continual Leaner model")

    return cl_model

def convert_batch_to_vilt_input_dict(batch: Dict):
    '''
    Convert inputs from batch_collate into format consumable by the ViltProcessor
    '''
    return {'images': batch['images'],
            'texts': batch['raw_texts']}

def convert_seq_batch_to_vilt_input_dict(batch: List, mean_image: Image):
    return {'images': [mean_image],
            'texts': list(batch[0])}

def convert_mc_batch_to_vilt_input_dict(batch: List, mean_image: Image):
    texts_a, texts_b = batch[0], batch[1]
    bs = len(texts_a)

    texts_b = list(itertools.chain(*texts_b)) #texts_b (n_choice, bs) -> (n_choice*bs,)
    text_pairs = [[texts_a[i%bs], tb] for i, tb in enumerate(texts_b)] # extend text_a & pair w/ text_b

    return {'images': [mean_image],
            'texts': text_pairs}
