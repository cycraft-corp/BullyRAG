import dataclasses
import os
import time
import random
import pathlib
import json

import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from src.arguments import parse_arguments
from src.data_processor import get_dataset
from src.data_processor import BaseDataset
from src.models import HuggingFaceModel, OpenAIModel, LlamaCppModel
from src.attack import PromptInjectionAttack, PreferenceAttack
from config import BASE_URL, OPENAI_API_KEY
# from src.evaluator import evaluate_tasks

def main(args):
    random.seed(42)
    args.path_to_result.mkdir(parents=True, exist_ok=True)

    if args.model_interface == "HuggingFace":
        model = HuggingFaceModel(args.model)
    elif args.model_interface == "OpenAI":
        model = OpenAIModel(args.model, BASE_URL, OPENAI_API_KEY)
    else:
        model = LlamaCppModel(args.model)
    # print(model.query("How are you today?"))
    dataset = BaseDataset(args.path_to_data)
    attack = PreferenceAttack(model, dataset, os.path.join(args.path_to_result, "preference_attack_llama_cpp.json"))
    attack.attack()
    print(attack.evaluate())


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
