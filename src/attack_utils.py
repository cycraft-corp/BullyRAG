def prompt_composer(template: str, placeholders: dict):
    for placeholder, value in placeholders.items():
        template = template.replace(placeholder, value)
    return template