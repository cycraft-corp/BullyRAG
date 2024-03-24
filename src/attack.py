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
    def __init__(self, model):
        if model_type == LLAMA:
            pass
        elif model_type == HUGGING_FACE:
            pass
        elif model_type == OPENAI:
            pass
        else:
            raise NotImplementedError()

    def attack() -> None:
        print("prompt injection attack")

class PreferenceAttack(BaseAttack):
    def __init__(self, model):
        self.model = model
        if model_type == LLAMA:
            pass
        elif model_type == HUGGING_FACE:
            pass
        elif model_type == OPENAI:
            pass
        else:
            raise NotImplementedError()

    def attack() -> None:
        print("preference attack")

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Attack")
    parser.add_argument('--model_type', type=str, help='Please specify model type, should be either huggingface_model, llama_cpp_model, or openai_model')
    parser.add_argument('--model_name_or_path', type=str, help='Please specify model name, either path-to-huggingface-model or something like gpt-3.5-turbo.')
    parser.add_argument('--dataset', type=str, help='The dataset must be given in a list of dictionaries. Each dictionary should include a passage and a list of QA Pairs.')
    parser.add_argument('--attack_name', type=str, help='PromptInjection or Preference')
    return parser.parse_args()


def main(args):
    # change to get_model() later
    if args.model_type == "huggingface_model":
        model = HuggingFaceModel(args.model_name_or_path)
    elif args.model_type == "llama_cpp_model":
        pass
        model = LlamaCppModel()
    elif args.model_type == "openai_model":
        model = OpenAIModel(args.model_name_or_path)
    else:
        raise NotImplementedError

    if args.attack_name == "PromptInjection":
        attacker = PromptInjectionAttack(model)
    elif args.attack_name == "Preference":
        attacker = PreferenceAttack(model)
    else:
        raise NotImplementedError

    attacker.attack()


if __name__ == '__main__':
    args = parse_args()
    main(args)
