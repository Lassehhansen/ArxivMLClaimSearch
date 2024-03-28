import pandas as pd
import numpy as np
from pathlib import Path
from pdf_extraction import extract_text
from tqdm import tqdm
import re
import json

root_dir = Path('/path/to/pdfs_dir')

res = []
for i in tqdm(root_dir.glob('**/*.pdf')):
    text = extract_text(i, page_limit = 10)
    res.append({
        'text': text,
        'meta':{
            'timestamp': re.findall('([0-9]+)', str(i))[0],
            'arxiv_id': i.name,
            'url': str(i)
        }
    })

with open('./paper_texts.jsonl', 'w') as f:
    for r in res:
        json.dump(r, f)
        f.write('\n')