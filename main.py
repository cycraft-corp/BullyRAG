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
# from src.evaluator import evaluate_tasks

DATA_PATH = "../data/qa-pair-2024.json"

def main(data_args, evaluator_args, training_args, model_args):
    random.seed(42)
    training_args.path_to_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # initialize_logging(
    #     path_to_logging_dir=training_args.path_to_checkpoint_dir, 
    #     level=training_args.log_level
    # )

    # model_class = get_model(model_args.model_name)
    # model = model_class(path_to_model_weight)
    model = HuggingFaceModel('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    # print(model.query("Hello! How are you?"))

    model = OpenAIModel("gpt-3.5-turbo")
    # print(model.query("Hello, who are you?"))

    # dataset_class = get_dataset(data_args.dataset_name)
    # with open(data_args.path_to_data, 'r') as json_file:
    #     data = json.load(json_file)
    # dataloader = DataLoader(data)
    # for elem in dataloader:
    #    print(elem)
    # dataset = BaseDataset(data_args.dataset_name)
    dataset = BaseDataset(DATA_PATH)
    for data in dataset:
        print(data)
    # dataset = dataset_class(
    #     data_args.path_to_data,
    #     tokenizer,
    #     data_args,
    #     model_args.model_max_length,
    #     training_args.device,
    #     all_batch_size=dist.get_world_size()*training_args.per_device_train_batch_size
    # )
    # evaluate_tasks(training_args.path_to_checkpoint_dir)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((
        DataArguments, EvaluatorArguments, TrainingArguments, ModelArguments
    ))
    data_args, evaluator_args, training_args, model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(data_args, evaluator_args, training_args, model_args)