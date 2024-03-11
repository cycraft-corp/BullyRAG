from collections import OrderedDict
import os
from typing import List
import json
import sys
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, T5EncoderModel, AutoConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer


def get_model(model_name):
    return getattr(sys.modules[__name__], model_name)


class BaseModel(nn.Module):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()
        pass

    # @abstractmethod
    def query(user_qry: str) -> str:
        pass

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        super(HuggingFaceModel, self).__init__(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to("cuda") #to(self.device)

    def query(self, user_qry: str) -> str:
        model_name_or_path = self.model_name_or_path
        tokenizer = self.tokenizer
        model = self.model

        messages = [ {"role": "user", "content": user_qry}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=20)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

class LlamaCppModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.model_args = model_args
        # tokenizer
        self.model = LlamaCpp(
            model_path=model_name_or_path,
            temperature=model_args.temperature,
            max_tokens=model_args.max_tokens,
            top_p=model_args.top_p,
            callback_manager=callback_manager, # ?
            verbose=True, # Verbose is required to pass to the callback manager
            )

    def query(self, user_qry: str) -> str:
        pass

class OpenAIModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.model_args = model_args

    def query(self, user_qry):
        result = openai.chat.completions.create(
            messages=user_qry, model="gpt-3.5-turbo", temperature=model_args.temperature
        )
        return result.choices[0].message  # Attribute access  