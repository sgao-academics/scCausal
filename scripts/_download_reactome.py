"""Download Reactome interaction data for causal validation."""
import os, requests

url = 'https://reactome.org/download/current/interactors/reactome.homo_sapiens.interactions.tab-delimited.txt'
out = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactome_interactions.txt')
os.makedirs(os.path.dirname(out), exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0 (scCausal academic research)'}
r = requests.get(url, headers=headers, timeout=60)
r.raise_for_status()

with open(out, 'w', encoding='utf-8') as f:
    f.write(r.text)

size_kb = os.path.getsize(out) / 1024
n_lines = r.text.count('\n')
print(f'Downloaded: {size_kb:.0f} KB, {n_lines} lines')
