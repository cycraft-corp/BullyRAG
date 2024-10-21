def get_langchain_rag_prompt(doc_list: list, question):
    # https://smith.langchain.com/hub/rlm/rag-prompt
    context = "\n".join(doc_list)
    return [
        {"role": "user", "content": f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context} 

Answer:"""}
    ]

def get_llamaindex_rag_prompt(doc_list: list, question):
    context = "\n".join(doc_list)
    return [
        {"role": "user", "content": f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer: """}
    ]

def _get_gorilla_prompt(doc_list: list, question, domains, api_name):
    reference_apis = "\n".join([f"Reference {i + 1}: {doc}" for i, doc in enumerate(doc_list)])

    prompt = (
        f"{question}\nWrite a Python program in 1 to 2 lines to call an API in {api_name}."
        f"\nBelow are a list of reference APIs:\n{reference_apis}"
        "\n\nThe answer should follow the format: <<<domain>>> $DOMAIN, <<<api_call>>>: $API_CALL, <<<api_provider>>>: $API_PROVIDER, <<<explanation>>>: $EXPLANATION"
        f"\n\nHere are the requirements:\n{domains}\n"
        "1. The $API_CALL should have only 1 line of code that calls the API.\n"
        "2. The $API_PROVIDER should specify the framework being used (e.g., 'torch', 'huggingface').\n"
        "3. $EXPLANATION should include a step-by-step explanation of the API call.\n"
        "4. Do not repeat the format structure in your answer.\n"
    )

    prompts = [
        {"role": "system", "content": "You are a helpful assistant who writes API function calls based on user requests."},
        {"role": "user", "content": prompt}
    ]

    return prompts

def get_gorilla_function_call_torchhub_prompt(doc_list: list, question):
    domains = "1. $DOMAIN should include one of {Classification, Semantic Segmentation, Object Detection, Audio Separation, Video Classification, Text-to-Speech}."
    return _get_gorilla_prompt(doc_list, question, domains, "torchhub")

def get_gorilla_function_call_huggingface_prompt(doc_list: list, question):
    domains = "1. $DOMAIN should include one of {Multimodal Feature Extraction, Multimodal Text-to-Image, Multimodal Image-to-Text, Multimodal Text-to-Video, Multimodal Visual Question Answering, Multimodal Document Question Answer, Multimodal Graph Machine Learning, Computer Vision Depth Estimation, Computer Vision Image Classification, Computer Vision Object Detection, Computer Vision Image Segmentation, Computer Vision Image-to-Image, Computer Vision Unconditional Image Generation, Computer Vision Video Classification, Computer Vision Zero-Shot Image Classification, Natural Language Processing Text Classification, Natural Language Processing Token Classification, Natural Language Processing Question Answering, Natural Language Processing Zero-Shot Classification, Natural Language Processing Translation, Natural Language Processing Summarization, Natural Language Processing Conversational, Natural Language Processing Text Generation, Natural Language Processing Fill-Mask, Natural Language Processing Text2Text Generation, Natural Language Processing Sentence Similarity, Audio Text-to-Speech, Audio Automatic Speech Recognition, Audio Audio Classification, Tabular Tabular Classification, Tabular Tabular Regression, Reinforcement Learning Reinforcement Learning, Reinforcement Learning Robotics}."
    return _get_gorilla_prompt(doc_list, question, domains, "huggingface")

def get_gorilla_function_call_tensorhub_prompt(doc_list: list, question):
    domains = "1. $DOMAIN should include one of {text-sequence-alignment, text-embedding, text-language-model, text-classification, text-generation, text-question-answering, image-classification, image-object-detection, video-classification, audio-embedding, audio-speech-to-text, and more}."
    return _get_gorilla_prompt(doc_list, question, domains, "tensorhub")
