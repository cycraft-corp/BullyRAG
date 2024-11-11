# BullyRAG: A Multi-Perspective RAG Robustness Evaluation Framework

![bullyrag_pic](images/bullyrag_pic.png)

`BullyRAG` is a tool designed to evaluate the robustness of Retrieval-Augmented Generation (RAG) frameworks. It focuses on three primary attack objectives: misleading RAG frameworks into generating inaccurate information, inserting malicious instructions, and executing remote malicious code. The tool tests robustness on two datasets—a real-time question-answering dataset and a function-calling API benchmark—and assesses defenses against three types of retrieval-phase attacks and five types of generation-phase attacks.

## Get Started

### Environment Setup

To test your RAG framework with BullyRAG, clone the repository to your machine and install all requirements:

```bash
git clone https://github.com/cycraft-corp/BullyRAG.git
cd BullyRAG
pip install –r requirements.txt
```

### Import Evaluators and Setup Configuration Variables

Let's use the `BFCLFCGEnerationEvaluator` as an example. First, we import its corresponding evaluator:

```python
from bullyrag.evaluators import BFCLFCGEnerationEvaluator
```

Other support evaluators can be found in `./bullyrag/evaluators.py`.

Then, set up the configuration variables for **evaluation**:

```python
MODEL = "gpt-4o-mini"
API_KEY = "[YOUR API KEY]"
PATH_TO_DATASET = "./sample_data/bfcl_functional_calling_sample_data.json"
TARGET_LANGUAGE_LIST = ["en"]
```

For `MODEL`s, BullyRAG supports three different interfaces: OpenAI, Hugging Face, and Anthropic.
For `PATH_TO_DATASET`, we provide sample data to evalute your RAG framework on question-answering tasks (`QA_sample_data.json`) and function-calling tasks (`bfcl_functional_calling_sample_data.json`). You can also evaluate on your customized data.
For `TARGET_LANGUAGE_LIST`, BullyRAG supports three languages: English (`en`), Mandarin (`zh-tw`), and Japanese (`jp`).

### Evaluator Setup

Then, set up the evalutor for evaluation:

```python
evaluator = BFCLCGEnerationEvaluator(
  inferencer="OpenAIInferencer",
  data_processor_config={
    "data_processor": "QADataProcessor",
    "path_to_dataset": PATH_TO_DATASET,
    "target_language_list": TARGET_LANGUAGE_LIST
  },
  inferencer_config={
    "model": MODEL, "base_url": BASE_URL, "api_key": API_KEY
  },
  attackers=["HelpfulBFCLAttacker"]
)
```

For `inferencer`, we currently support the `SentenceTransformerInferencer`, `OpenAIInferencer`, `HuggingFaceInferencer`, and `AnthropicInferencer`. We plan to extend our framework to the LlamaCpp inference engine in the future.
For `data_processor`, more options can be found in `./bullyrag/data_processors.py`.
For `attackers`, more options can be found in `./bullyrag/attackers.py`.

### Evaluate!

After everything is setup, simply evaluate your framework in one line!

```python
results = evaluator()
```

The returned results are provided as a dictionary. It has several keys:

- `attackwise_total_answer_status_map`: It indicates for each attack, which indices in the dataset were successfully attacked, answered correctly (attack failed), answered vaguely (includes both correct and incorrect information), or answered erratically (wrong format, etc.).
- `attackwise_total_obfuscation_ratio_list`: It indicates how different the obfuscated document is from the original document. A lower obfuscation ratio with a higher successfully attacked rate indicate a powerful attack.
- `attackwise_total_detailed_response_list`: We also record all raw responses from the language model for each attack.

To record the evaluation results, we recommend saving it into a `.json` file.

## Features

BullyRAG is the first ***Open-Source*** framework for evaluting RAG robustness! It supports:

### 3 Model Interfaces

BullyRAG flexibly supports all models that can be loaded from the following commonly used inference engines: OpenAI, HuggingFace, Anthropic.

### 3 Retrieval-Phase Attacks and 5 Generation-Phase Attacks

BullyRAG tests RAG frameworks' robustness across 8 different attacks:

* For **Retrieval-Phase Attacks**, we test RAGs' robustness against inserted invisible control characters. There are 3 kinds:
  * The left-to-right mark character: `\u202eevil\u202c` may appear as if `live` to a language model!
  * The zero-width space character: We can interrupt many words and disrupt the LLM's understanding of the input without being noticed by users.
  * The back space character: We can insert a lot of malicious information and combine it with backspaces, making it invisible to users.
* For **Generation-Phase Attacks**, we leverage certain natural tendencies of language models to guide them toward producing untruthful outputs. We utilize the following model preferences:
  * Preferred keywords - Language models favor words with positive connotations, like "helpful" or "harmless.
  * LLM's self-generated context - A large body of research study models' tendencies to their self-generated context.
  * Emotional Stimuli - Language models are sensitive to emotional cues; passages including "emotional blackmails" are more likely to be prioritized.
  * Major Consensus - Similar to humans, models tend to trust the "majority opinion"; by making false information appear frequently in the knowledge database, the model is more likely to incorporate it into its responses.
  * Profit Temptation - Besides threatening LLMs with emotional stimuli, models are also drawn to statements suggesting rewards or benefits.

### 1 Real-Time Updated QA Dataset and 1 Function-Calling API Benchmark

* Real-time QA Dataset: To ensure that the RAG framework lacks prior knowledge of the questions in the dataset, we provide a real-time, auto-updating dataset that continuously fetches and parses fresh abstracts from ArXiv, generating new QA pairs accordingly.
* Function-Calling API Benchmark: We also support evaluation on function-calling, another common application of RAG frameworks. Using data from the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html), we test the robustness of RAGs against inserted malicious instructions and malicious code execution.
