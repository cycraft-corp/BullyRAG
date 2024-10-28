from abc import ABC, abstractmethod
from typing import Union
import sys

from bullyrag.utils import load_json

def get_data_processor_class(class_name):
    return getattr(sys.modules[__name__], class_name)

class DataProcessor(ABC):
    def __init__(self, path_to_dataset, target_language_list: Union[str, list], *args, **kwargs):
        self.index = 0
        self.replayinfo_map = {}

        if isinstance(target_language_list, str):
            target_language_list = [target_language_list]
        self._initialize(path_to_dataset, target_language_list, *args, **kwargs)

        if not hasattr(self, "processed_data"):
            raise ValueError("Please initialize the variable 'processed_data' in '_initialize()'")

    @abstractmethod
    def _initialize(self, path_to_dataset, *args, **kwargs):
        pass

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self):
            qa_data = self.processed_data[self.index]
            doc = self.doc_list[qa_data["doc_index"]]

            kwargs = {}
            for attacker_name in self.replayinfo_map:
                attacker_replayinfo_map = self.replayinfo_map[attacker_name]
                key = f"{qa_data['question']}_{qa_data['gt_answer']}"
                kwargs[attacker_name] = {
                    "obfuscated_doc": attacker_replayinfo_map[key]
                }

            self.index += 1
            return doc, qa_data, kwargs
        raise StopIteration

    def __len__(self):
        return len(self.processed_data)

class QADataProcessor(DataProcessor):
    def _initialize(self, path_to_dataset, target_language_list, *args, **kwargs):
        data = load_json(path_to_dataset)

        doc_list = []
        processed_data = []
        extract_target_languages_data = []

        # TODO: different languages?
        for target_language in target_language_list:
            for d in data:
                target_language_d = d["processed_data"].get(target_language, None)
                if target_language_d is None:
                    continue

                for qa_pair in target_language_d["qa-pairs"]:
                    processed_data.append({
                        "question": qa_pair["question"],
                        "gt_answer": qa_pair["gt_answer"],
                        "malicious_answer": qa_pair["malicious_answer"],
                        "doc_index": len(doc_list)
                    })
                doc_list.append(target_language_d["doc"])

        self.doc_list = doc_list
        self.processed_data = processed_data

        path_to_replayinfo_map = kwargs.get("path_to_replayed_logging", None)
        if path_to_replayinfo_map is not None:
            replayinfo_map = load_json(path_to_replayinfo_map).get("attackwise_replayinfo_map", None)
            if replayinfo_map is None:
                raise ValueError(f"There is no replayinfo_map in {path_to_replayinfo_map}."
                                 " Please ensure that you have set the flag 'enable_replaylogging' to 'True' for the evaluator to log the info successfully.")

            self.replayinfo_map = replayinfo_map
            print(f"Successfully loaded replayinfo_map from '{path_to_replayinfo_map}'." 
                   " Data will be automatically loaded from replayinfo with priority.")
