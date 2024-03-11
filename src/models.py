from collections import OrderedDict
import os
from typing import List
import json
import sys
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, T5EncoderModel, AutoConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer


class BaseModel(nn.Module):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init()__()
        self.model_name_or_path = model_name_or_path
        pass

    def query(user_qry: str) -> str:
        pass

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def query(self, user_qry: str) -> str:
        # query
        # decode with tokenizer
        pass

class LlamaCppModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.model = LlamaCpp(
            model_path=model_name_or_path,
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=True, # Verbose is required to pass to the callback manager
            )

    def query(self, user_qry: str) -> str:
        pass

class OpenAIModel(BaseModel):
    def __init__():
        pass

    def query():
        result = openai.chat.completions.create(
            messages=messages, model="gpt-3.5-turbo", temperature=0
        )
        return result.choices[0].message  # Attribute access  