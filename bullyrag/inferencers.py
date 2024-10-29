import sys

try:
    import torch
    import transformers
except:
    print("Please install 'torch' or 'transformers' to enable 'HuggingFaceInferencer'")

try:
    from openai import OpenAI
except:
    print("Please install 'openai' to enable 'OpenAIInferencer'")

def get_inferencer_class(class_name):
    return getattr(sys.modules[__name__], class_name)

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

    def inference(self, messages, max_tokens=None, temperature=None, *args, **kwargs):
        outputs = self.pipeline(
            messages, 
            max_new_tokens=max_tokens if max_tokens is not None else self.max_tokens, 
            temperature=temperature if temperature is not None else self.temperature, 
            *args, **kwargs
        )
        return outputs[0]["generated_text"][-1]["content"]

class OpenAIInferencer:
    def __init__(self, model, base_url, api_key, max_tokens=256, temperature=1):
        self.model = model

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.max_tokens = max_tokens
        self.temperature = temperature

    def inference(self, messages, max_tokens=None, temperature=None, model=None, *args, **kwargs):
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model if model is not None else self.model,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            **kwargs
        )
        return chat_completion.choices[0].message.content

# TODO
class AnthropicInferencer:
    def __init__(self):
        pass

# TODO
class LlamaCppInferencer:
    def __init__(self):
        pass
