import collections
from typing import Dict, List
import random
import sys

import torch
import torch.distributed as dist
import json

class BaseDataset:
    def __init__(self, path_to_data):
        data = self.load_data(path_to_data) 
        self.questions = data["questions"]
        self.paragraphs = data["paragraphs"]
        self.true_answer = data["true_answer"]
        self.false_answer = data["false_answer"]

    def load_data(path_to_data: str) -> dict:
        with open(path_to_data) as f:
            data = json.load(f)
        return self.preprocess(data)

    def preprocess(data) -> dict:
        return data