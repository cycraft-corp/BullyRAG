from collections import OrderedDict
import os
from typing import List
import json
import sys
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
from llama_cpp import Llama

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, T5EncoderModel, AutoConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer
from config import BASE_URL, OPENAI_API_KEY
from src.const import ModelInterface


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
        self.model_interface = ModelInterface.HuggingFace

    def query(self, user_qry: str) -> str:
        model_name_or_path = self.model_name_or_path
        tokenizer = self.tokenizer
        model = self.model

        messages = [ {"role": "user", "content": user_qry}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=500)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("user_qry", user_qry)
        # print("output", decoded_output)
        tar = "<|assistant|>"
        pos = decoded_output.find(tar)
        if pos == -1:
            return "fail to parse completions"
        else:
            return decoded_output[pos + len(tar):]

class LlamaCppModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        # self.model_args = model_args
        self.llm = Llama(model_path = self.model_name_or_path)
        self.model_interface = ModelInterface.LlamaCpp

    def query(self, user_qry: str) -> str:
        output = self.llm(user_qry,
            max_tokens=150,
            stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=False # Don't echo the prompt back in the output
        )
        return output['choices'][0]['text']

class OpenAIModel(BaseModel):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.model_interface = ModelInterface.OpenAI
        # self.model_args = model_args

    def get_chat_completions(self, user_prompt, model, client, system_prompt=None, temperature=0.0, max_tokens=1000):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            #max_tokens=max_tokens,
            #temperature=temperature
        )
        return chat_completion.choices[0].message.content

    def query(self, user_qry):
        base_url = BASE_URL
        api_key = OPENAI_API_KEY

        openai_client = OpenAI(base_url=base_url, api_key=api_key)
        # print(user_qry)
        result = self.get_chat_completions(
            user_prompt=user_qry,
            model=self.model_name_or_path,
            client=openai_client,
        )
        return result  # Attribute access  
