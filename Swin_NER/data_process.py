import os
import sys
sys.path.append(os.getcwd() + '/Swin_NER')
import json
import logging
import itertools
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)


class NERDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]


class NERDataModule(LightningDataModule):
    def __init__(self, config, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)

        self.config = config

        # raw data
        self.raw_train_path = 'Data/raw_data_clue/train.json'
        self.raw_dev_path = 'Data/raw_data_clue/dev.json'
        self.raw_test_path = 'Data/raw_data_clue/test.json'
        # tokenized data
        tokenized_dir = 'Data/tokenized_data_clue/'
        if not os.path.exists(tokenized_dir):
            os.makedirs(tokenized_dir)
        self.tokenized_train_path = tokenized_dir + 'train.pt'
        self.tokenized_dev_path = tokenized_dir + 'dev.pt'
        self.tokenized_test_path = tokenized_dir + 'test.pt'

        # ner dataprocess
        self.label2idx, self.idx2label = self._label2idx()

        # tokenizer params
        self.max_seq_len = config.model_args.max_seq_length
        self.tokenizer_name = config.plm_name

        # utils
        self.train_data, self.dev_data, self.test_data = None, None, None
        self.train_len = 0
        self.batch_size = config.model_args.train_batch_size
        self.num_workers = config.num_processes
        self.test_batch_size = 0
    
    def _label2idx(self):
        tags = ['B-', 'I-', 'E-', 'S-'] if self.config.ner_tagging == 'BIOES' else (
            ['B-', 'I-'] if self.config.ner_tagging == 'BIO' else None)
        cartesian = itertools.product(tags, self.config.label_list)
        labels = ['O'] + [''.join([label[0], label[1]]) for label in cartesian]
        label_num = len(labels)
        label2idx = {labels[i]: i for i in range(label_num)}
        idx2label = {i: labels[i] for i in range(label_num)}
        return label2idx, idx2label

    def my_prepare_data(self):
        def _get_bio(sentence_len, labels, raw2token):
            bioes_label = ['O'] * sentence_len
            for entity_type, entity_dict in labels.items():
                for name, pos_list in entity_dict.items():
                    for pos in pos_list:
                        start, end = raw2token[pos[0]], raw2token[pos[1]]
                        bioes_label[start] = 'B-{}'.format(entity_type)
                        for i in range(start + 1, end + 1):
                            bioes_label[i] = 'I-{}'.format(entity_type)
            bioes_label = [self.label2idx[label] for label in bioes_label]
            return bioes_label

        def _preprocess_data(data, tokenizer, type):
            input_data = tokenizer.encode_plus(data['text'], max_length = self.max_seq_len, 
                            return_offsets_mapping=True, padding='max_length', truncation=True, 
                            return_tensors="pt")
            raw2token = {}
            for idx, offset in enumerate(input_data['offset_mapping'].squeeze()):
                if offset.numpy().sum() != 0:
                    for i in range(offset[0], offset[1]):
                        raw2token[i] = idx
            if type == 'test':
                return {
                    "input_ids": input_data["input_ids"].squeeze(),
                    "attention_mask": input_data["attention_mask"].squeeze(),
                }
            else: 
                bio_label = _get_bio(self.max_seq_len, data['label'], raw2token)
                return {
                    "input_ids": input_data["input_ids"].squeeze(),
                    "attention_mask": input_data["attention_mask"].squeeze(),
                    "y_true_bio": torch.tensor(bio_label).long()
                } 

        def _load_data(path):
            file = open(path, 'r', encoding='utf-8')
            data = []
            for line in file.readlines():
                d = json.loads(line)
                data.append(d)
            return data

        def _save_data(data, path):
            torch.save(data, path)
        
        def _main_process(in_path, out_path, tokenizer, type):
            data = _load_data(in_path)
            data = [_preprocess_data(d, tokenizer, type) for d in tqdm(data)]
            _save_data(data, out_path)
        
        tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name)
        _main_process(self.raw_train_path, self.tokenized_train_path, tokenizer, 'train')
        _main_process(self.raw_dev_path, self.tokenized_dev_path, tokenizer, 'dev')
        _main_process(self.raw_test_path, self.tokenized_test_path, tokenizer, 'test')

    def setup(self, stage):
        if stage in (None, "fit"):
            if not os.path.isfile(self.tokenized_train_path) or not os.path.isfile(self.tokenized_dev_path):
                raise ValueError(
                    "train_data and eval_data not prepared!"
                    )
            train_data = torch.load(self.tokenized_train_path)
            self.train_data = NERDataSet(train_data)
            dev_data = torch.load(self.tokenized_dev_path)
            self.dev_data = NERDataSet(dev_data)
            self.train_len = self.train_data.__len__()
        # if stage in (None, "test"):
        #     if not os.path.isfile(self.tokenized_test_path):
        #         raise ValueError(
        #             "train_data and eval_data not prepared!"
        #             )
        #     test_data = torch.load(self.tokenized_test_path)
        #     self.test_data = NERDataSet(test_data)
        #     self.test_batch_size = self.test_data.__len__()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size = self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data, batch_size = self.test_batch_size, num_workers=self.num_workers)
