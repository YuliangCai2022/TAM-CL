import copy

import torch
from timm.models.layers import trunc_normal_
from torch import nn
import logging
import utils as cutils
from TAB import ClassAttention, Block
from task_configs import task_configs
from typing import List, Dict

logger = logging.getLogger(__name__)

class ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n


class DyTox(nn.Module):
    """"DyTox for the win!

    :param transformer: The base transformer.
    :param nb_classes: Thhe initial number of classes.
    :param individual_classifier: Classifier config, DyTox is in `1-1`.
    :param head_div: Whether to use the divergence head for improved diversity.
    :param head_div_mode: Use the divergence head in TRaining, FineTuning, or both.
    :param joint_tokens: Use a single TAB forward with masked attention (faster but a bit worse).
    """
    def __init__(
        self,
        transformer,
        nb_classes,
        individual_classifier='Yes',
        task_list = None,
        teacher_model = None,
        joint_tokens=False,
        resnet=False
    ):
        super().__init__()

        self.transformer = transformer
        self.nb_classes = nb_classes
        self.embed_dim = transformer.encoder_dim
        self.individual_classifier = "Yes"
        self.joint_tokens = joint_tokens
        self.in_finetuning = False
        self.teacher_model = teacher_model
        self.nb_classes_per_task = [nb_classes]
        self.task_list = task_list
        #self.patch_embed = transformer.patch_embed
        #self.pos_embed = transformer.pos_embed
        self.sabs = transformer.vilt_encoder
        self.tabs = transformer.TAB
        self.task_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1,self.embed_dim))])

        
        in_dim, out_dim = self._get_ind_clf_dim()
        self.head = transformer.task_layer_dict

    def end_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def add_model(self, nb_new_classes):
        """Expand model as per the DyTox framework given `nb_new_classes`.

        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        # Class tokens ---------------------------------------------------------
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        # ----------------------------------------------------------------------


        # classifier are created at the very beginning of vilt+tab initialization.

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.

        By default, DyTox is in 1-1.
        """
        in_dim = self.embed_dim
        out_dim = self.nb_classes_per_task[-1]
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.task_tokens, requires_grad=requires_grad)
            elif name == 'sab':
                if self.use_resnet:
                    self.backbone.eval()
                    cutils.freeze_parameters(self.backbone, requires_grad=requires_grad)
                else:
                    self.sabs.eval()
                    cutils.freeze_parameters(self.patch_embed, requires_grad=requires_grad)
                    cutils.freeze_parameters(self.pos_embed, requires_grad=requires_grad)
                    cutils.freeze_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.freeze_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'ca': self.tabs.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                              if self.individual_classifier else \
                              self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
            'head_div': self.head_div.parameters() if self.head_div is not None else None
        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.task_tokens)):
                for j in range(i + 1, len(self.task_tokens)):
                    dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        return log

    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses

    def forward_features(self, task_key: str, images: List, texts: List[str]):
        # Shared part, this is the ENCODER
        B = len(images)

        vilt_output = self.transformer(task_key=task_key, images=images, texts=texts)
        vilt_output = vilt_output.reshape(vilt_output.shape[0],1,vilt_output.shape[1])
        tokens = []
        attentions = []
        mask_heads = None
        
        for task_token in self.task_tokens:
            task_token = task_token.expand(B, -1, -1)
            task_token_add = task_token

            while task_token.shape[2] != vilt_output.shape[2]:
                logger.info("task token shape and vilt output shape is not the same")
                task_token=torch.cat((task_token,task_token_add), dim=2)

            task_token, attn, v = self.tabs(torch.cat((task_token, vilt_output), dim=1), mask_heads=mask_heads)
            
            attentions.append(attn)
            tokens.append(task_token[:, 0])

        self._class_tokens = tokens
        return tokens, tokens[-1], attentions, vilt_output.reshape(vilt_output.shape[0],vilt_output.shape[2])


    def forward_classifier(self, task_key: str, tokens, last_token):
        """Once all task embeddings e_1, ..., e_t are extracted, classify.

        Classifier has different mode based on a pattern x-y:
        - x means the number of task embeddings in input
        - y means the number of task to predict

        So:
        - n-n: predicts all task given all embeddings
        But:
        - 1-1: predict 1 task given 1 embedding, which is the 'independent classifier' used in the paper.

        :param tokens: A list of all task tokens embeddings.
        :param last_token: The ultimate task token embedding from the latest task.
        """
        logits_div = None
        tasks = self.task_list
        logits = None
        for i in range(len(tokens)):
            if tasks[i] == task_key:
                logits = self.head[tasks[i]](tokens[i]) ####!!!!!!!!remember to change there back to task_key!!!!!!!!!
        #for logit in logits:
        #    logger.info(logit.shape)
        #logits = torch.cat(logits, dim=1)


        return {
            'logits': logits,
            'div': logits_div,
            'tokens': tokens
        }

    def forward(self, task_key: str, images: List, texts: List[str]):
        tokens, last_token, _, _ = self.forward_features(task_key, images=images, texts=texts)
        return self.forward_classifier(tokens = tokens, task_key = task_key, last_token = last_token)


def eval_training_finetuning(mode, in_ft):
    if 'tr' in mode and 'ft' in mode:
        return True
    if 'tr' in mode and not in_ft:
        return True
    if 'ft' in mode and in_ft:
        return True
    return False



def update_dytox(model, task_id, args, teacher_model):
    if task_id == 0:
        print(f'Creating DyTox!')
        model = DyTox(
            model,
            nb_classes=task_configs[args.ordered_cl_tasks[0]]['num_labels'],
            individual_classifier='1-1',
            task_list=args.ordered_cl_tasks
            #head_div=args.head_div > 0.,
            #head_div_mode=args.head_div_mode,
            #joint_tokens=args.joint_tokens,
            #resnet=args.resnet
        )
    else:
        # the num here is the output dim of current task's header
        model.add_model(task_configs[args.ordered_cl_tasks[task_id]]['num_labels'])

    return model