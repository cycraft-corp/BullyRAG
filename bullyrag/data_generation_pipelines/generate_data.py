import argparse
import json

from bullyrag.inferencers import OpenAIInferencer

from .data_fetchers import parse_arxiv_data
from .qa_synthesizer import QASynthesizer

MODEL = "gpt-4o-mini"
BASE_URL = "[YOUR BASE URL]"
API_KEY = "[YOUR API KEY]"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and parse ArXiv data for QA synthesis.")
    parser.add_argument(
        "--start_date", 
        type=str, 
        required=True,
        help="Start date and time in the format 'YYYYMMDDHHMM' (e.g., '202401010000')."
    )
    parser.add_argument(
        "--end_date", 
        type=str, 
        required=True,
        help="End date and time in the format 'YYYYMMDDHHMM' (e.g., '202412312359')."
    )
    parser.add_argument(
        "--limit_num", 
        type=int, 
        default=20,
        help="Maximum number of papers to fetch (default: 20)."
    )

    args = parser.parse_args()

    if len(args.start_date) != 12 or not args.start_date.isdigit():
        raise ValueError("The start_date format must be 'YYYYMMDDHHMM' (12 digits).")
    if len(args.end_date) != 12 or not args.end_date.isdigit():
        raise ValueError("The end_date format must be 'YYYYMMDDHHMM' (12 digits).")
    
    synthesizer = QASynthesizer(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY
    )

    parsed_paper_list = parse_arxiv_data(args.start_date, args.end_date, limit_num=args.limit_num)

    with open(f'../../sample_data/parsed_paper_list_{args.start_date}_to{args.end_date}.json', 'w') as f:
        json.dump(parsed_paper_list, f, indent=4)

    synthesized_qa_list = synthesizer.get_qa_synthesis_prompt(parsed_paper_list, 3, "English (en)")

    with open(f'../../sample_data/synthesized_qa_list_{args.start_date}_to{args.end_date}.json', 'w') as f:
        json.dump(synthesized_qa_list, f, indent=4)
