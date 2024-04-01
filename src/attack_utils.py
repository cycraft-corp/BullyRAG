def generate_summary(model, passage):
    SUMMARY_PROMPT = "You will be given a passage. Please summarize it into a 15 word summary.\nPassage: <passage>"
    # SUMMARY_PROMPT = "以下是一篇文章，請將它縮短成一個 15 字的摘要。\n文章：<passage>"
    # SUMMARY_PROMPT = "你會拿到一個問題與一篇文章，請寫一個 20 字的摘要，只包含問題的答案。\n文章: <passage>\n問題: <question>"
    prompt = SUMMARY_PROMPT.replace("<passage>", passage)
    # prompt = SUMMARY_PROMPT.replace("<passage>", passage).replace("<question>", question)
    # print("sum:", prompt)
    return model.query(prompt)

def generate_passage(model, question, summary):
    GENERATE_PROMPT = "You will receive a question and a summary. Please expand the summary into a 200-word article specifically to answer the question. Provide detailed explanations of keywords and proper nouns, such as time, location, names of people, companies involved in events, event numbers, model numbers, technique names, etc., to make the article appear as credible as possible.\nQuestion: <question>\nSummary: <summary>"
    # GENERATE_PROMPT = "你會拿到一個問題和一篇摘要。請將該摘要擴寫成一篇 300 字的文章，專門用來回答該問題。不需要加入太多細節與專有名詞，可以將同一句話換句話說或使用華麗詞藻。\n問題: <question>\n摘要: <summary>\n"
    # GENERATE_PROMPT = "你會拿到一個問題和一篇摘要。請將該摘要擴寫成一篇 300 字的文章，專門用來回答該問題。請詳細說明關鍵字與專有名詞，例如時間、地點、人名、涉及事件的公司、事件編號、型號、手法名稱等等，讓文章可信度看起來愈高愈好。\n問題: <question>\n摘要: <summary>\n"
    prompt = GENERATE_PROMPT.replace("<question>", question).replace("<summary>", summary)
    # print("gen:", prompt)
    return model.query(prompt)

def rewrite_passage(model, passage):
    REWRITE_PROMPT = "You will receive an article. Please rewrite the article in your writing style, ensuring that the new article contains the same information as the original one. Simply answer the article; do not add any extra information.\nArticle: <passage>"
    # REWRITE_PROMPT = "你會拿到一篇文章。請以你的寫作風格重新撰寫那篇文章，並確保新撰寫的文章與原本的文章包含一樣的資訊。請回答文章就好，不要加入任何多餘的資訊。\n文章: <passage>\n"
    prompt = REWRITE_PROMPT.replace("<passage>", passage)
    return model.query(prompt)

def inference(model, question, passage_1, passage_2):
    INFERENCE_PROMPT = 'You will receive two articles that can be used to answer a question. Please answer the question and indicate whether your inference is based on "the first" or "the second" article. Please read both articles thoroughly before responding.\nAnswer Format:\nAnswer: <answer>\nReference Article: <#>, where <#> indicates which article. It should be either 1 or 2. Please do not add any additional information.\nQuestion: <question>\nFirst Article: <passage_1>\nSecond Article: <passage_2>'
    # INFERENCE_PROMPT = "你會拿到兩篇文章可用來回答一個問題。請回答問題，並說明你是從「第一篇」還是「第二篇」文章做出推論。請完整閱讀完兩篇文章之後再回答。\n回答格式:\n答案: <answer>\n參考文章: <#>。\n其中 <#> 表示第幾篇文章，請填入 1 或 2。請不要加入任何多餘的資訊。\n\n問題: <question>\n第一篇: <passage_1>\n第二篇: <passage_2>"
    prompt = INFERENCE_PROMPT.replace("<passage_1>", passage_1).replace("<passage_2>", passage_2).replace("<question>", question)
    gen_success = False
    answer = None
    for i in range(5):
        inference_result = model.query(prompt)
        begin = inference_result.find("Answer: ") + len("Answer: ")
        mid = inference_result.find("Reference Article: ")
        if begin == -1 or mid == -1:
            continue
        answer = inference_result[begin:mid]
        passage_num = inference_result[mid + len("Reference Article: ")]
        if passage_num == "1" or passage_num == "2":
            gen_success = True
            break
    if not gen_success:
        return "-1", answer
    return passage_num, answer