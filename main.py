import dataclasses
import os
import time

import transformers
import torch
import torch.distributed as dist

from src.arguments import DataArguments, EvaluatorArguments, TrainingArguments, ModelArguments

def main(data_args, evaluator_args, training_args, model_args):
    set_random_seed(42)
    training_args.path_to_checkpoint_dir.mkdir(exist_ok=True)

    initialize_logging(
        path_to_logging_dir=training_args.path_to_checkpoint_dir, 
        level=training_args.log_level
    )

    pipeline_class = get_pipeline(evaluator_args.pipeline_name)
    pipeline = pipeline_class(
        model=model, 
        tokenizer=tokenizer,
        incontext_provider=incontext_provider,
        device=training_args.device
    )
    evaluate_tasks(pipeline, training_args.path_to_checkpoint_dir)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((
        DataArguments, EvaluatorArguments, TrainingArguments, ModelArguments
    ))
    data_args, evaluator_args, training_args, model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(data_args, evaluator_args, training_args, model_args)
