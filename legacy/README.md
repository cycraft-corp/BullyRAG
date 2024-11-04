# LLM_Evaluator_Tool
The RAG Attack Evaluation Tool is a versatile Python package designed to assess the robustness of Retrieve And Generate (RAG) frameworks against various attacks. It offers high modularity, allowing users to seamlessly integrate different models, types of attacks, and datasets.

## Features
### Models
The tool supports interfaces for widely-used RAG models, including Hugging Face models, OpenAI, and Llama cpp.

### Attacks
The tool detects various types of attacks on RAG models, such as injection of malicious instructions and inducing inaccurate retrievals by adjusting reference documents as well as additional attacks proposed in research papers.

### Dataset
We offer real-time auto-update dataset, continuously fetching and parsing fresh data from ArXiv and news websites.

## Installation
The tool can be easily installed via pip:
```
pip install rag-attack-evaluation-tool
```
Alternatively, you can clone this repository and install the tool from the source.

## Usage
### Fetch Data
To fetch real-time data from ArXiv:
```bash
python fetch_data.py
```
To generate QA pairs from passages:
```bash
python data_generation.py
```

### Attach + Evaluation
To perform an attack and evaluation using the RAG Attack Evaluation Tool:
```
python main.py \
    --title "Evaluator_Tool" \
    --path-to-data sample_data/qa-pair-2024.json \
    --path-to-result results \
    --model-interface OpenAI \
    --model gpt-3.5-turbo
```
Options:
* `--title`: Title of the evaluation tool.
* `--path-to-data`: Path to the QA data file.
* `--path-to-result`: Directory to store result log files.
* `--model-interface`: Type of model interface. Options: HuggingFace, OpenAI, Llama Cpp.
* `--model`: Name of the model. For example, "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for HuggingFace, "gpt-3.5-turbo" for OpenAI, and preprocessed gguf files for Llama cpp.
The path specified for `--path-to-result` is the directory where all result log files will be stored.