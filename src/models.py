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


class BaseModel(nn.Module):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init()__()
        pass

    def query(user_qry: str) -> str:
        pass

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def query(self, user_qry: str) -> str:
        model_id = self.model_id
        tokenizer = self.tokenizer
        model = self.model

        messages = [ {"role": "user", "content": user_qry}]

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        outputs = model.generate(inputs, max_new_tokens=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

class LlamaCppModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
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
        self.model_name_or_path = model_name_or_path
        pass

    def query():
        result = openai.chat.completions.create(
            messages=messages, model="gpt-3.5-turbo", temperature=0
        )
        return result.choices[0].message  # Attribute access  