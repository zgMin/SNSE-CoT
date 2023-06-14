import os
from torch.utils.data import Dataset
import os
import json
import random
import numpy as np
import torch
from utils_prompt import *
from generate_negation import *

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
}

def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids,

def load_data_img(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]
    name_maps = json.load(open('vision_features/name_map.json'))

    # check
    if args.img_type == "resnet":
        image_features = np.load('vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load('vision_features/clip.npy')
    elif args.img_type == "detr":
        image_features = np.load('vision_features/detr.npy')
    else:
        image_features = np.load('vision_features/detr.npy')
    print("img_features size: ", image_features.shape)

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, name_maps, image_features

class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }
class Softneg(Dataset):
    def __init__(
            self, problems, qids, target_len, args
    ):

        self.data = {qid: problems[qid] for qid in qids}
        self.summ_len = target_len
        self.solution = []
        self.lecture = []
        self.choice = []
        self.answer = []

        for qid in self.data:
            solution, lecture, choice, answer = build_train_pair(problems, qid, args,enhance = True)
            self.lecture.append(lecture)
            self.choice.append(choice)
            self.answer.append(answer)

            self.solution.append(solution)


    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def generate(self,  num = 1):
        all_texts = []
        for index in range(len(self.lecture)):
            solution = str(self.solution[index])
            choice = str(self.choice[index])
            answer = str(self.answer[index])
            lecture = str(self.lecture[index])
            negation_texts = []

            if solution != "":
                for _ in range(num):
                    negation_text = convert_text(solution, type=5, choice=choice, answer=answer)
                    for text in negation_text:
                        negation_texts.append(" ".join(f"Solution: {lecture} {text}.".split()))
            else:
                for _ in range(num):
                    negation_text = convert_text(lecture, type=5, choice=choice, answer=answer)
                    for text in negation_text:
                        negation_texts.append(" ".join(f"Solution: {text}{solution}.".split()))
            all_texts.append(negation_texts)
        return all_texts




class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None,soft_negations=None, epoch = 1, mode = None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.epoch = epoch
        self.enhance_LE = False
        if args.enhance_LE is True and args.prompt_format == 'QCM-LE' and mode=='train':
            self.enhance_LE = True
            self.soft_negations = soft_negations
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.lecture = []
        self.choice = []
        self.answer = []
        self.image_ids = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None

            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))
    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)
    def set_config(self,enhance_LE=False):
        self.enhance_LE=enhance_LE
    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""


        source_text = str(self.source_text[index])
        source_text = " ".join(source_text.split())

        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]
        image_ids = torch.tensor(image_ids).squeeze()

        # cleaning data so as to ensure data is in string type




        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        if self.enhance_LE:
            # 在这个位置进行软负样本构造，主要是为了动态的生成
            # 即每个轮次都随机化软负样本，防止软负例分步到一边

            negation_ids = []
            tmp = random.randint(0, self.epoch - 1)
            negation_text = self.soft_negations[index][tmp * 5:tmp * 5 + 5]
            # negation_text = self.soft_negations[index][5: 5 + 5]
            for text in negation_text:
                negation_ids.append(self.tokenizer.batch_encode_plus(
                    [text],
                    max_length=self.summ_len,
                    pad_to_max_length=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )["input_ids"].squeeze().tolist())

            target_text = " ".join(target_text.split())

            target = self.tokenizer.batch_encode_plus(
                [target_text],
                max_length=self.summ_len,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            target_ids = target["input_ids"].squeeze().tolist()
            return {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "image_ids": image_ids,
                "labels": target_ids,
                "soft_negation": negation_ids,
            }

        target_text = " ".join(target_text.split())

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target_ids = target["input_ids"].squeeze().tolist()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }
