import os.path
import json

import numpy as np
import jieba
import torch
from torch.utils.data import Dataset

import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self), indent=2) + '\n'


class QQRProcessor:
    TASK = 'KUAKE-QQR'

    def __init__(self, data_dir):
        self.task_dir = os.path.join(data_dir)

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_train.json'))

    def get_dev_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_dev.json'))

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_test.json'))

    def get_labels(self):
        return ['0', '1', '2']

    def _create_examples(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        examples = []
        for sample in samples:
            guid = sample['id']
            text_a = sample['query1']
            text_b = sample['query2']
            label = sample.get('label', None)

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class QQRDataset(Dataset):
    def __init__(self, examples: List[InputExample], label_list: List[Union[str, int]], vocab_mapping: Dict,
                 max_length: int = 64):
        """
        :param examples:
        :param label_list:
        :param vocab_mapping:
        :param max_length:
        """
        super(QQRDataset, self).__init__()
        self.examples = examples
        self.vocab_mapping = vocab_mapping
        self.max_length = max_length

        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

    def __len__(self):
        return len(self.examples)

    def _tokenize(self, text):
        # ????????????
        tokens = list(jieba.cut(text))

        token_ids = []
        for token in tokens:
            # ?????????????????????????????????????????????ID
            if token in self.vocab_mapping:
                token_id = self.vocab_mapping[token]
                token_ids.append(token_id)
            else:  # ???????????????????????????
                # ??????????????????????????????????????????????????????????????????ID
                if len(token) > 1:
                    for t in list(token):
                        if t in self.vocab_mapping:
                            token_ids.append(self.vocab_mapping[t])
                        else:
                            token_ids.append(np.random.choice(len(self.vocab_mapping), 1)[0])
                # ????????????????????????????????????????????????????????????????????????????????????????????????ID??????????????????ID
                else:
                    token_ids.append(np.random.choice(len(self.vocab_mapping), 1)[0])

        # ???????????????????????????????????????????????????max_length
        token_ids, attention_mask = self._pad_truncate(token_ids)
        return token_ids, attention_mask

    def _pad_truncate(self, token_ids: List[int]):
        """
        :param token_ids:
        :return:
            token_ids:???????????????????????????ID??????
            attention_mask:????????????????????????
        """
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(token_ids)

            diff = self.max_length - len(token_ids)
            token_ids.extend([0] * diff)
            attention_mask.extend([0] * diff)

        return token_ids, attention_mask

    def __getitem__(self, index):
        example = self.examples[index]
        label = None
        if example.label is not '':
            label = self.label2id[example.label]

        text_a_token_ids, text_a_attention_mask = self._tokenize(example.text_a)
        text_b_token_ids, text_b_attention_mask = self._tokenize(example.text_b)

        return {'text_a_input_ids': text_b_token_ids,
                'text_b_input_ids': text_b_token_ids,
                'text_a_attention_mask': text_a_attention_mask,
                'text_b_attention_mask': text_b_attention_mask,
                'label': label}

class DataCollator:
    def __call__(self, features: List[Dict[str, Any]]):
        """
        # ?????????batch???????????????????????????Tensor
        :param features:
        :return:
        """
        text_a_input_ids = []
        text_b_input_ids = []
        text_a_attention_mask = []
        text_b_attention_mask = []
        labels = []
        for item in features:
            text_a_input_ids.append(item['text_a_input_ids'])
            text_b_input_ids.append(item['text_b_input_ids'])
            text_a_attention_mask.append(item['text_a_attention_mask'])
            text_b_attention_mask.append(item['text_b_attention_mask'])
            if item['label'] is not None:
                labels.append(item['label'])
        text_a_input_ids = torch.tensor(text_a_input_ids, dtype=torch.long)
        text_b_input_ids = torch.tensor(text_b_input_ids, dtype=torch.long)
        text_a_attention_mask = torch.tensor(text_a_attention_mask, dtype=torch.long)
        text_b_attention_mask = torch.tensor(text_b_attention_mask, dtype=torch.long)
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = None

        return {'text_a_input_ids': text_a_input_ids,
                'text_b_input_ids': text_b_input_ids,
                'text_a_attention_mask': text_a_attention_mask,
                'text_b_attention_mask': text_b_attention_mask,
                'labels': labels
                }