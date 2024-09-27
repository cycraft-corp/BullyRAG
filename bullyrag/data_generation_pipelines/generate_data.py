import argparse


from bullyrag.inferencers import OpenAIInferencer

from .data_fetchers import parse_arxiv_data
from .qa_synthesizer import QASynthesizer

MODEL = "gpt-4o-mini"

if __name__ == "__main__":
    synthesizer = QASynthesizer(
        model=MODEL,
        base_url="--",
        api_key="--"
    )
    parsed_paper_list = parse_arxiv_data("202401010000", "202412312359", limit_num=20)
    print(parsed_paper_list)
    print("-----")
