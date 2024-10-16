import abc
import collections
from typing import List, Union, Callable
import re

from bullyrag.attackers import get_attacker_class
from bullyrag.data_processors import get_data_processor_class
from bullyrag.inferencers import get_inferencer_class
from bullyrag.prompt_factory import (
    get_langchain_rag_prompt, get_llamaindex_rag_prompt,
    get_gorilla_function_call_torchhub_prompt, get_gorilla_function_call_huggingface_prompt, get_gorilla_function_call_tensorhub_prompt
)
from bullyrag.utils import (
    calculate_edit_distance, check_answer_correctness,
    ATTACKSUCCESSFULLY, ANSWERCORRECTLY, ANSWERVAGUELY, ANSWERCHAOTICALLY
)

class BaseEvaluator(abc.ABC):
    def __init__(
        self, inferencer: Union[str, object], data_processor_config: dict, 
        inferencer_config: dict, attackers: List[Union[str, object]]=[], *args, **kwargs
    ):
        self.attackers = self.initialize_attackers(attackers)
        self.inferencer = self.initialize_inferencer(inferencer, inferencer_config)
        self.data_processor = self.initialize_data_processor(data_processor_config)

    def initialize_data_processor(self, data_processor_config):
        data_processor = data_processor_config.get("data_processor", None)
        print(data_processor)
        if data_processor is None:
            raise ValueError("Argument 'data_processor_config' requires the key "
                             "'data_processor' with the types 'str' or 'object'")

        if isinstance(data_processor, str):
            if any([key not in data_processor_config for key in ["path_to_dataset", "target_language_list"]]):
                raise ValueError(f"Argument 'data_processor_config' ('str' type) "
                                 f"requires the keys: 'path_to_dataset' and 'target_language_list'")

            processed_data_processor = get_data_processor_class(data_processor)(
                path_to_dataset = data_processor_config["path_to_dataset"],
                target_language_list = data_processor_config["target_language_list"]
            )
        elif isinstance(data_processor, object):
            processed_data_processor = data_processor
        else:
            raise ValueError(f"Key 'data_processor' of the argument 'data_processor_config' "
                             f"only supports types 'str' or 'object'")
        return processed_data_processor

    def initialize_inferencer(self, inferencer, inferencer_config):
        if isinstance(inferencer, str):
            processed_inferencer = get_inferencer_class(inferencer)(**inferencer_config)
        elif isinstance(inferencer, object):
            processed_inferencer = inferencer
        else:
            raise ValueError(f"Argument 'inferencer' only supports types 'str' "
                             f"or 'object', not {type(inferencer)}")
        return processed_inferencer

    def initialize_attackers(self, attackers):
        processed_attackers = []
        for attacker in attackers:
            if isinstance(attacker, str):
                processed_attackers.append(get_attacker_class(attacker)())
            elif isinstance(attacker, object): # modified attack -> attacker
                processed_attackers.append(attacker)
            else:
                raise ValueError(f"Argument 'attackers' only supports types "
                                 f"'str' or 'object', not {type(attacker)}")
        return processed_attackers

    @abc.abstractmethod
    def _evaluate(self):
        pass

class RetrievalEvaluator(BaseEvaluator):
    def __init__(self):
        pass

class ChatEvaluator(BaseEvaluator):
    DEFAULT_RAG_PROMPT_MAP = {
        "langchain": get_langchain_rag_prompt,
        "llamaindex": get_llamaindex_rag_prompt,
        "gorilla_torchhub": get_gorilla_function_call_torchhub_prompt,
        "gorilla_huggingface": get_gorilla_function_call_huggingface_prompt,
        "gorilla_tensorhub": get_gorilla_function_call_tensorhub_prompt
    }

    def __call__(self, rag_prompt_fn: Union[str, Callable]="langchain"):
        if isinstance(rag_prompt_fn, str):
            rag_prompt_fn = self.DEFAULT_RAG_PROMPT_MAP.get(rag_prompt_fn, None)
            if rag_prompt_fn is None:
                raise ValueError(f"Invalid key for rag_prompt_fn! Current valid str "
                                 f"type rag_prompt_fn: {list(self.DEFAULT_RAG_PROMPT_MAP.keys())}")

        print(f"Apply the prompt composition function - '{rag_prompt_fn.__name__}' for evaluation!")

        evaluation_metrics = {
            "attackwise_total_answer_status_map": collections.defaultdict(
                lambda: collections.defaultdict(list)
            ),
            "attackwise_total_obfuscation_ratio_list": collections.defaultdict(list),
            "attackwise_total_detailed_response_list": collections.defaultdict(list)
        }
        self._evaluate(rag_prompt_fn, evaluation_metrics)
        evaluation_metrics["answer_status_percentage"] = self.calculate_percentage(evaluation_metrics)
        return evaluation_metrics

    @abc.abstractmethod
    def _evaluate(self, rag_prompt_fn: Callable, evaluation_metrics: dict):
        pass

    def calculate_percentage(self, evaluation_metrics):
        """
        Calculates the percentage of correct, incorrect, and malicious responses.
        Returns a dictionary with the percentage breakdown.
        """
        correct_count = 0
        vaguely_count = 0
        attack_successfully_count = 0
        chaotically_count = 0

        for attacker_name, status_map in evaluation_metrics["attackwise_total_answer_status_map"].items():
            correct_count += len(status_map.get(ANSWERCORRECTLY, []))
            vaguely_count += len(status_map.get(ANSWERVAGUELY, []))
            attack_successfully_count += len(status_map.get(ATTACKSUCCESSFULLY, []))
            chaotically_count += len(status_map.get(ANSWERCHAOTICALLY, []))

        total_count = correct_count + vaguely_count + attack_successfully_count + chaotically_count

        correct_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
        vaguely_percentage = (vaguely_count / total_count) * 100 if total_count > 0 else 0
        attack_successfully_percentage = (attack_successfully_count / total_count) * 100 if total_count > 0 else 0
        chaotically_percentage = (chaotically_count / total_count) * 100 if total_count > 0 else 0

        return {
            ATTACKSUCCESSFULLY: attack_successfully_percentage,
            ANSWERCORRECTLY: correct_percentage,
            ANSWERVAGUELY: vaguely_percentage,
            ANSWERCHAOTICALLY: chaotically_percentage
        }

class CleanChatEvaluator(ChatEvaluator):
    def _evaluate(self, rag_prompt_fn: Callable, evaluation_metrics: dict):
        for data_index, (doc, qa_data) in enumerate(self.data_processor):
            rag_messages = rag_prompt_fn(
                doc_list=[doc],
                question=qa_data["question"]
            )
            response = self.inferencer.inference(rag_messages)
            answer_status = check_answer_correctness(
                response, qa_data["gt_answer"], None
            )

            attacker_name = "no_attack"
            evaluation_metrics["attackwise_total_detailed_response_list"][attacker_name].append(response)
            evaluation_metrics["attackwise_total_answer_status_map"][attacker_name][answer_status].append(data_index)

# TODO
class InstructionInjectionEvaluator(ChatEvaluator):
    def _evaluate(self, rag_prompt_fn: Callable, evaluation_metrics: dict):
        pass

class WrongAnswerEvaluator(ChatEvaluator):
    def _evaluate(self, rag_prompt_fn: Callable, evaluation_metrics: dict):
        for data_index, (doc, qa_data) in enumerate(self.data_processor):
            for attacker in self.attackers:
                obfuscated_doc = attacker.obfuscate_reference_doc(
                    doc=doc,
                    gt_answers=qa_data["gt_answer"],
                    malicious_answers=qa_data["malicious_answer"]
                )

                rag_messages = rag_prompt_fn(
                    # TODO: How to handle order issue?
                    #doc_list=[doc, obfuscated_doc],
                    doc_list=[obfuscated_doc, doc],
                    question=qa_data["question"]
                )
                response = self.inferencer.inference(rag_messages)
                answer_status = check_answer_correctness(
                    response, qa_data["gt_answer"], qa_data["malicious_answer"]
                )
                # Calculate obfuscated ratio (i.e., the edit distance between 
                # the original response and obfuscated response)
                edit_distance_ratio = calculate_edit_distance(doc, obfuscated_doc)

                attacker_name = attacker.__class__.__name__
                evaluation_metrics["attackwise_total_detailed_response_list"][attacker_name].append(response)
                evaluation_metrics["attackwise_total_answer_status_map"][attacker_name][answer_status].append(data_index)
                evaluation_metrics["attackwise_total_obfuscation_ratio_list"][attacker_name].append(edit_distance_ratio)

class FunctionalCallingGenerationEvaluator(ChatEvaluator):
    def __call__(self, rag_prompt_fn: Union[str, Callable]="gorilla_huggingface"):
        return super().__call__(rag_prompt_fn=rag_prompt_fn)

    @abc.abstractmethod
    def _extract_inference_result(self, raw_response):
        pass

    def _evaluate(self, rag_prompt_fn: Callable, evaluation_metrics: dict): # discuss: default value for rag_prompt_fn?
        total_count = 0

        for data_index, (doc, qa_data) in enumerate(self.data_processor):
            for attacker in self.attackers:
                obfuscated_doc = attacker.obfuscate_reference_doc(
                    doc=doc,
                    gt_answers=qa_data["gt_answer"],
                    malicious_answers=qa_data["malicious_answer"]
                )

                rag_messages = rag_prompt_fn(
                    doc_list=[obfuscated_doc, doc],
                    question=qa_data["question"]
                )
                raw_response = self.inferencer.inference(rag_messages)

                pred_answer = self._extract_inference_result(raw_response)
                answer_status = check_answer_correctness(
                    pred_answer, qa_data["gt_answer"], qa_data["malicious_answer"]
                )
                edit_distance_ratio = calculate_edit_distance(doc, obfuscated_doc)

                attacker_name = attacker.__class__.__name__
                evaluation_metrics["attackwise_total_detailed_response_list"][attacker_name].append(raw_response)
                evaluation_metrics["attackwise_total_answer_status_map"][attacker_name][answer_status].append(data_index)
                evaluation_metrics["attackwise_total_obfuscation_ratio_list"][attacker_name].append(edit_distance_ratio)

            if data_index == 1:
                break

class GorillaFCGenerationEvaluator(FunctionalCallingGenerationEvaluator):
    ANSWER_EXTRACTION_RE_PATTERN = r'<<<api_call>>>:\s*(.*?)(?=,\s*<<<)'

    def _extract_inference_result(self, raw_response):
        matched_result = re.search(self.ANSWER_EXTRACTION_RE_PATTERN, raw_response)

        stripped_response = ""
        if matched_result:
            stripped_response = matched_result.group(1).strip()
        return [stripped_response, raw_response]

