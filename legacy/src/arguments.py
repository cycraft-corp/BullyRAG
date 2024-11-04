import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluator Tool")

    parser.add_argument("--title", type=str, default="Evaluator_Tool",
                        help="Title of the tool (default: Evaluator_Tool)")
    parser.add_argument("--path-to-data", type=Path, required=True,
                        help="Path to the input data file")
    parser.add_argument("--path-to-result", type=Path, required=True,
                        help="Directory to the output result file")
    parser.add_argument("--model-interface", type=str, choices=["LlamaCpp", "OpenAI", "HuggingFace"], required=True,
                        help="Type of model interface, can be LlamaCpp, OpenAI, or HuggingFace")
    parser.add_argument("--model", type=str, action=ModelAction, nargs="?",
                        help="Model to be used")

    return parser.parse_args()

class ModelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Set default model based on model interface
        if values is None:
            if namespace.model_interface == "LlamaCpp":
                setattr(namespace, self.dest, "/mnt/nas/erichuang_dir/reference_llm/mistral_ggml/7b/mistral-7b-instruct-v0.1.Q4_K_S.gguf")
                # need a step to create gguf
            elif namespace.model_interface == "HuggingFace":
                setattr(namespace, self.dest, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            elif namespace.model_interface == "OpenAI":
                setattr(namespace, self.dest, "gpt-3.5-turbo")
        else:
            setattr(namespace, self.dest, values)
