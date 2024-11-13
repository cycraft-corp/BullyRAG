# Data Generation Pipeline Usages

## Overview

This directory includes 3 files:

1. **`generate_data.py`**: A script to fetch and generate QA-pairs for passages from a specified time interval using real-time arXiv data.
2. `data_fetchers.py`: It includes a data fetcher that can fetch data from ArXiv.
3. `qa_synthesizer.py`: It uses language models, particularly, OpenAI API, to simulate authentic question and answer pairs related to the document.

## Description

To generate your own real-time ArXiv data, you need to substitute `API_KEY` and `BASE_URL` in the script `generate_data.py` with your own keys. Then, using `generate_data.py`, you can use command-line arguments to fetch and generate QA-pairs for passages within a specific time range.

There are 3 command line arguments for the script `generate_data.py`.

- **`--start_date`**: The start date and time for the data fetch in the format `YYYYMMDDHHMM` (e.g., `202401010000` for January 1st, 2024, at 00:00 hours).
- **`--end_date`**: The end date and time for the data fetch in the format `YYYYMMDDHHMM` (e.g., `202412312359` for December 31st, 2024, at 23:59 hours).
- **`--limit_num`**: The maximum number of abstracts to fetch from arXiv. The default value is 20.

## Example usage

```bash
python generate_data.py \
    --start_date 202401010000 \
    --end_date 202412312359 \
    --limit_num 15 \
```

## Outputs

The script generates 2 files:

1. **`parsed_paper_list.json`**: A JSON file containing the parsed abstracts fetched from arXiv.
2. **`synthesized_qa_list.json`**: A JSON file containing synthesized QA-pairs generated from the fetched abstracts.
