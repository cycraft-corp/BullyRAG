from enum import Enum

class ModelInterface(Enum):
    HuggingFace = 0
    LlamaCpp = 1
    OpenAI = 2