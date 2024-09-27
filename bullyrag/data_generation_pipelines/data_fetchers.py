import requests
import xml.etree.ElementTree as ET

def parse_arxiv_data(start_date, end_date, limit_num, seen_title_set=set()):
    if not isinstance(limit_num, int) or limit_num <= 0:
        raise ValueError("'limit_num' should be positive integer!")
    parsed_paper_list = []

    base_url = f"http://export.arxiv.org/api/query?search_query=cat:cs.CV&start=0&max_results={limit_num}&sortBy=lastUpdatedDate&sortOrder=descending&submittedDate:[{start_date}+TO+{end_date}]"

    response = requests.get(base_url)
    root = ET.fromstring(response.content)
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        published_time = entry.find('{http://www.w3.org/2005/Atom}published').text
        abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text

        if title in seen_title_set:
            continue
        seen_title_set.add(title)

        parsed_paper_list.append({
            'title': title, 
            'published_time': published_time, 
            'doc': abstract.replace("\n", " ").strip(),
            'language': 'en'
        })
    return parsed_paper_list
