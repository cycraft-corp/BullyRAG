import collections
from typing import Dict, List
import random
import sys
import os

import torch
import torch.distributed as dist
import json

from src.const import ModelInterface
from src.attack_utils import generate_summary, generate_passage, rewrite_passage, inference


class BaseAttack():
    def __init__():
        pass

class PromptInjectionAttack(BaseAttack):
    def __init__(self, model, dataset, path_to_result): # change path_to_result = args.path_to_result
        self.model = model
        self.dataset = dataset
        self.path_to_result = os.join(os.getcwd(), path_to_result)
        if model.model_interface == ModelInterface.HuggingFace:
            pass
        elif model.model_interface == ModelInterface.LlamaCpp:
            pass
        elif model.model_interface == ModelInterface.OpenAI:
            pass
        else:
            raise NotImplementedError

    def attack(self) -> None:
        print("prompt injection attack")

class PreferenceAttack(BaseAttack):
    def __init__(self, model, dataset, path_to_result): # change path_to_result = args.path_to_result
        self.model = model
        self.dataset = dataset
        self.path_to_result = path_to_result
        if model.model_interface == ModelInterface.HuggingFace:
            pass
        elif model.model_interface == ModelInterface.LlamaCpp:
            pass
        elif model.model_interface == ModelInterface.OpenAI:
            pass
        else:
            raise NotImplementedError
        self.ai_answers = []
        self.ai_choices = []
        self.ai_summaries = []
        self.ai_passages = []

    def attack(self) -> None:
        for i in range(len(self.dataset.qa_pairs)):
            if i > 0:
                with open(self.path_to_result, 'r') as f:
                    results = json.load(f)
                self.ai_answers, self.ai_choices, self.ai_passages = results["ai_answers"], results["ai_choices"], results["ai_passages"]
            passage, questions = self.dataset.qa_pairs[i]['abstract'], self.dataset.qa_pairs[i]['qa_pairs']['questions']
            summary = generate_summary(self.model, passage)
            ai_passage_list = []
            ai_choices_list = []
            ai_answers_list = []
            # summaries_list = []
            for question in questions:
                ai_passage = generate_passage(self.model, question, passage)
                chosen_passage, answer = inference(self.model, question, ai_passage, passage)
                ai_passage_list.append(ai_passage)
                ai_answers_list.append(answer)
                ai_choices_list.append(chosen_passage)
                # summaries_list.append(summary)
            print(i, ai_choices_list)
            # self.ai_summaries.append(summaries_list)
            self.ai_answers.append(ai_answers_list)
            self.ai_choices.append(ai_choices_list)
            self.ai_passages.append(ai_passage_list)

            # results = {"ai_answers":self.ai_answers, "ai_choices":self.ai_choices, "ai_passages":self.ai_passages, "ai_summaries":self.ai_summaries}
            results = {"ai_answers":self.ai_answers, "ai_choices":self.ai_choices, "ai_passages":self.ai_passages}
            with open(self.path_to_result, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    def evaluate(self): # returns self_preference rate
        with open(self.path_to_result) as f:
            results = json.load(f)
            # assert type(results) == dict()

        ai_choices = results["ai_choices"]
        one = 0
        two = 0
        for ai_choice in ai_choices:
            for num in ai_choice:
                if num == "1":
                    one += 1
                elif num == "2":
                    two += 1

        return one / (one + two) * 100

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
