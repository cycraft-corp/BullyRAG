import json
import requests
import xml.etree.ElementTree as ET

PATH = "../../data/data-2024.json"

def get_arxiv_papers(start_year, end_year):
    papers = []
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=cat:cs.CV&start=0&max_results=1000&sortBy=lastUpdatedDate&sortOrder=descending"
    
    for year in range(start_year, end_year + 1):
        query_with_year = f"{query}&submittedDate:[{year}01010000+TO+{year}12312359]"
        response = requests.get(base_url + query_with_year)
        root = ET.fromstring(response.content)
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            published_time = entry.find('{http://www.w3.org/2005/Atom}published').text
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
            if "2024" not in published_time:
                continue
            papers.append({'title': title, 'published_time': published_time, 'abstract': abstract})
    return papers

papers = get_arxiv_papers(2024, 2024)
print(papers[0]['abstract'])
# for idx, paper in enumerate(papers, 1):
    # print(f"{idx}. Title: {paper['title']}")
    # print(f"   Published Time: {paper['published_time']}")

with open(PATH, 'w') as json_file:
    json.dump(papers, json_file, indent=4)