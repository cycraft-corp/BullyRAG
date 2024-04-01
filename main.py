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

from src.arguments import DataArguments, EvaluatorArguments, TrainingArguments, ModelArguments
from src.data_processor import get_dataset
from src.data_processor import BaseDataset
from src.models import HuggingFaceModel, OpenAIModel
from src.attack import PromptInjectionAttack, PreferenceAttack
# from src.evaluator import evaluate_tasks

DATA_PATH = "../data/qa-pair-2024.json"
PATH_TO_RESULT = 'results/preference_attack.json'

def main(data_args, evaluator_args, training_args, model_args):
    random.seed(42)
    training_args.path_to_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # initialize_logging(
    #     path_to_logging_dir=training_args.path_to_checkpoint_dir, 
    #     level=training_args.log_level
    # )

    # model = HuggingFaceModel('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    model = OpenAIModel("gpt-3.5-turbo")
    dataset = BaseDataset(DATA_PATH)
    attack = PreferenceAttack(model, dataset, PATH_TO_RESULT)
    attack.attack()
    print(attack.evaluate())


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((
        DataArguments, EvaluatorArguments, TrainingArguments, ModelArguments
    ))
    data_args, evaluator_args, training_args, model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(data_args, evaluator_args, training_args, model_args)
