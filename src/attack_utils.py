def prompt_composer(prompt: str, placeholders: dict) -> str:
    for [key, val] in dict:
        prompt = prompt.replace(key, val)
    return prompt