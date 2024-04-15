import collections
from typing import Dict, List
import random
import sys
import os

import torch
import torch.distributed as dist
import json

from src.attack_utils import prompt_composer, BaseAttack

class PromptInjectionAttack(BaseAttack):
    def __init__(self, model, dataset, path_to_result): # change path_to_result = args.path_to_result
        self.model = model
        self.dataset = dataset
        self.path_to_result = os.join(os.getcwd(), path_to_result)

    def attack(self) -> None:
        print("prompt injection attack")