def get_chat_completions(user_prompt, model, client, system_prompt=None, temperature=0.0, max_tokens=1000):
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