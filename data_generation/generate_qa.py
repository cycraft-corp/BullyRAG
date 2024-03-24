import json
from openai import OpenAI
from config import BASE_URL, OPENAI_API_KEY

DATA_PATH = "../../data/data-2024.json"
SAVE_PATH = "../../data/qa-pair-2024.json"

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

def ask_gpt3(user_qry):
    base_url = BASE_URL
    api_key = OPENAI_API_KEY

    openai_client = OpenAI(base_url=base_url, api_key=api_key)
    result = get_chat_completions(
        user_prompt=user_qry,
        model="gpt-3.5-turbo",
        client=openai_client,
    )
    return result  # Attribute access

PROMPT = "Please generate three question-answer pairs that can be answered using the following passage. The answers must be  less than five words. Answer with a Python List of three elements. Each element is a dictionary with the keys 'question' and 'answer'. It will be parsed directly by Python. Use double quotes. Answer must be retrieved from the passage. Keep the answers as short as possible. Don't add formatting.\n\n"
TRIALS = 5
results = []

with open(DATA_PATH, 'r') as json_file:
    papers = json.load(json_file)

for i in range(5):
    abstract = papers[i]['abstract']
    prompt = PROMPT + abstract
    for j in range(TRIALS):
        try:
            res = ask_gpt3(prompt)
            lst = json.loads(res)
            results.append({'abstract':abstract, 'title':papers[i]['title'], 'published_time':papers[i]['published_time'], 'qa-pairs':lst})
            break
        except Exception as e:
            print("fail on " + str(i))
            if j == TRIALS - 1:
                results.append({'abstract':abstract, 'title':papers[i]['title'], 'published_time':papers[i]['published_time'], 'qa-pairs':[]})
            continue

with open(SAVE_PATH, 'w') as json_file:
    json.dump(results, json_file, indent=4)

# assert len(results) == len(papers)