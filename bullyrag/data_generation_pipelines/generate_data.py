import argparse


from bullyrag.inferencers import OpenAIInferencer

from .data_fetchers import parse_arxiv_data
from .qa_synthesizer import QASynthesizer

MODEL = "gpt-4o-mini"

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
        base_url="--",
        api_key="--"
    )

    parsed_paper_list = parse_arxiv_data(args.start_date, args.end_date, limit_num=args.limit_num)

    print(parsed_paper_list)
    print("-----")
