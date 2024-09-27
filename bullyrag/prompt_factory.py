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
