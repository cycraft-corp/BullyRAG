from typing import List
import sys

try:
    import torch
    import transformers
except:
    print("Please install 'torch' or 'transformers' to enable 'HuggingFaceInferencer'")

try:
    from sentence_transformers import SentenceTransformer
except:
    print("Please install 'sentence_transformers' to enable 'SentenceTransformerInferencer'")

try:
    from openai import OpenAI
except:
    print("Please install 'openai' to enable 'OpenAIInferencer'")

def get_inferencer_class(class_name):
    return getattr(sys.modules[__name__], class_name)

class SentenceTransformerInferencer:
    def __init__(self, model_name_or_path, device_map="auto"):
        self.model = SentenceTransformer(model_name, device=device_map)

    def infer_embedding_response(self, sentence_list: List[str], *args, **kwargs):
        return self.model.encode(sentence_list)

class HuggingFaceInferencer:
    def __init__(
        self, model_name_or_path, torch_dtype="float16", 
        device_map="auto", max_tokens=256, temperature=1
    ):
        self.model_name_or_path = model_name_or_path
        self.pipeline = transformers.pipeline(
            "text-generation", model=self.model_name_or_path,
            model_kwargs={"torch_dtype": torch_dtype}, device_map=device_map
        )

        self.max_tokens = max_tokens
        self.temperature = temperature

    def infer_chat_response(self, messages, max_tokens=None, temperature=None, *args, **kwargs):
        outputs = self.pipeline(
            messages, 
            max_new_tokens=max_tokens if max_tokens is not None else self.max_tokens, 
            temperature=temperature if temperature is not None else self.temperature, 
            *args, **kwargs
        )
        return outputs[0]["generated_text"][-1]["content"]

class OpenAIInferencer:
    def __init__(self, model, api_key, base_url="https://api.openai.com/v1/", max_tokens=256, temperature=1):
        self.model = model

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.max_tokens = max_tokens
        self.temperature = temperature

    def infer_chat_response(self, messages, max_tokens=None, temperature=None, model=None, *args, **kwargs):
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model if model is not None else self.model,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            **kwargs
        )
        return chat_completion.choices[0].message.content

    def infer_embedding_response(self, sentence_list: List[str], model=None, *args, **kwargs):
        embeddings = self.client.embeddings.create(
            input=sentence_list, 
            model=model if model is not None else self.model
        )
        return [d.embedding for d in embeddings.data]

class AnthropicInferencer:
    def __init__(self, model, api_key, base_url=None, max_tokens=256, temperature=1):
        self.client = anthropic.Anthropic(api_key)

    def infer_chat_response(self, messages, max_tokens=None, temperature=None, model=None, *args, **kwargs):
        response = self.client.messages.create(
            messages=messages,
            model=model if model is not None else self.model,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        return response.content[0].text

# TODO
class LlamaCppInferencer:
    def __init__(self):
        pass
