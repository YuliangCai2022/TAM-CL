import json
import pickle as pkl
import os
from os import listdir
import torch
# from utils.word_utils import normalize_word
from tqdm import tqdm


def create_gqa_labels(root):
    # train_path = os.path.join(root, 'train_all_questions')
    # train_list = listdir(os.path.join(root, 'train_all_questions'))

    # TODO: update to balanced dataset
    train_path = os.path.join(root, 'train_balanced_questions.json')
    val_path = os.path.join(root, 'val_balanced_questions.json')
    answer2label = {}
    num_label = 0

    print("Loading train files...")
    # for train_file in train_list:
    train = json.load(open(train_path))
    for question_id, question_object in tqdm(train.items()):
        answer = question_object['answer']
        if answer not in answer2label.keys():
            label = num_label
            answer2label[answer] = label
            num_label += 1
        else:
            label = answer2label[answer]

    print("Loading val files...")
    val = json.load(open(val_path))
    for question_id, question_object in tqdm(val.items()):
        answer = question_object['answer']
        if answer not in answer2label.keys():
            label = num_label
            answer2label[answer] = label
            num_label += 1
        else:
            label = answer2label[answer]
    print("type of answer2label: {} with size {}".format(type(answer2label), len(answer2label)))

    pkl.dump(answer2label, open(os.path.join(root, 'cached_gqa_data', 'gqa_answer2label.pkl'), 'wb'))


def target_tensor(num_labels, labels):
    """ create the target by labels """
    target = torch.zeros(num_labels)
    target[labels] = 1
    return target

# helper function to check image in gqa image dataset
def check_image():
    directory = '/home/shared/MCL/gqa/images'
    for image in listdir(directory):
        if image == '501156.jpg':
            print("found 501156.jpg")
            break
    print("done search")

if __name__ == '__main__':
    #create_gqa_labels('/home/shared/MCL/gqa/questions')
    create_gqa_labels('/project/rostamim_919/caiyulia/GQA/')
    
    # root = '/home/shared/MCL/gqa/questions'
    # train_path = os.path.join(root, 'train_all_questions')
    # train_list = listdir(os.path.join(root, 'train_all_questions'))
    # val_path = os.path.join(root, 'val_all_questions.json')

    # print("Loading train files...")
    # for train_file in train_list:
    #     train = json.load(open(os.path.join(train_path,train_file)))
    #     for question_id, question_object in tqdm(train.items()):
    #         if '1238592' in question_id:
    #             print("1238592 found")
    #             print(question_object)
    #             break

    
            
