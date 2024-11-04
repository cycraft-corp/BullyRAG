import collections
from typing import Dict, List
import random
import sys
import os

import torch
import torch.distributed as dist
import json

def get_dataset(dataset_name):
    return getattr(sys.modules[__name__], dataset_name)

class BaseDataset:
    def __init__(self, path_to_data):
        self.qa_pairs = self.load_data(path_to_data) 
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor < len(self.qa_pairs):
            next_element = self.qa_pairs[self.cursor]
            self.cursor += 1
            return next_element
        else:
            raise StopIteration

    def load_data(self, path_to_data: str) -> dict:
        path_to_data = os.path.join(os.getcwd(), path_to_data)
        with open(path_to_data) as f:
            data = json.load(f)
        return self.preprocess(data)

    def preprocess(self, data) -> dict:
        for i in range(len(data)):
            questions = []
            answers = []
            for qa_pair in data[i]['qa-pairs']:
                questions.append(qa_pair['question'])
                answers.append(qa_pair['answer'])
            data[i]['qa_pairs'] = {'questions' : questions, 'answers' : answers}
        return data