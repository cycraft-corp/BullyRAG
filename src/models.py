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
from src.model_utils import get_chat_completions


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
        self.model = Llama(
            model_path = self.model_name_or_path,
            chat_format = "llama-2",
            n_ctx = 2048, # increase max tokens
            n_gpu_layers=-1 #set all to move to GPU
        )

    def query(self, user_qry: str) -> str:
        chat_completion = self.model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": user_qry
            }],
        )

        # prompt = get_prompt(user_qry, self.conv_template)
        # output = self.llm(prompt,
        #     max_tokens=150,
        #     # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        #     echo=False # Don't echo the prompt back in the output
        # )
        # return output['choices'][0]['text']
        return chat_completion['choices'][0]['message']['content']

class OpenAIModel(BaseModel):
    def __init__(self, model_name_or_path, base_url, api_key, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _get_chat_completions(user_prompt, model, client, system_prompt=None, temperature=0.0, max_tokens=1000):
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
        result = self._get_chat_completions(
            user_prompt=user_qry,
            model=self.model_name_or_path,
            client=self.client,
        )
        return result  # Attribute access  
