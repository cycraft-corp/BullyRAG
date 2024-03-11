import collections
from typing import Dict, List
import random
import sys

import torch
import torch.distributed as dist
import json



class BaseAttack():
    def __init__():
        pass

class PromptInjectionAttack(BaseAttack):
    def __init__(self, model_name_or_path, model_type):
        if model_type == LLAMA:
            pass
        elif model_type == HUGGING_FACE:
            pass
        elif model_type == OPENAI:
            pass
        else:
            raise NotImplementedError()

    def attack() -> None:
        pass

class PrefernceAttack(BaseAttack):
    def __init__(self, model_name_or_path, model_type):
        if model_type == LLAMA:
            pass
        elif model_type == HUGGING_FACE:
            pass
        elif model_type == OPENAI:
            pass
        else:
            raise NotImplementedError()

    def attack() -> None:
        pass