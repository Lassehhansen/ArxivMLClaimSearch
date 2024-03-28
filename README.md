# Arxiv and ML Conference Claim Search

This project focuses on conducting an extensive literature review of machine learning papers, exploring claims about the superiority of AUPRC (Area Under the Precision-Recall Curve) over AUROC (Area Under the Receiver Operating Characteristic curve) in cases of class imbalance. Utilizing large datasets from Arxiv, a variety of ML conferences and NeurIPS and leveraging GPT-3.5 and GPT-4.0 Turbo, we have conducted an innovative and comprehensive review of over 1.5 million papers from Arxiv and an extensive collection from NeurIPS between 1987 and 2019.

## Downloading the arXiv Dataset:

Detailed documentation for the download can be found [here](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).

To download only the files used for the litterature search, follow these steps: 

```bash
# Download the urls.txt file which contains URLs to all the datasets
wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'

# Get the urls related to each of the datasets in RedPajama
grep “arxiv” urls.txt > arxiv_urls.txt

# Use the modified script to download only the specific files
while read line; do
    dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
    mkdir -p $(dirname $dload_loc)
    wget "$line" -O "$dload_loc"
done < arxiv_urls.txt
```

## arXiv v1 Search Data Description
- 93.8 GB
- Total number of texts in JSONL files: 1558306
- Total number of texts that contain either AUROC OR AUPRC related keywords: 16022
- Total number of texts that contain both: 8244
- Texts after GPT 3.5 search: 2728
- Texts after GPT 4.0 Turbo search: 201

## arXiv v2 Search Data Description
- 93.8 GB
- Total number of texts in JSONL files: 1558306
- Total number of texts that contain either AUROC OR AUPRC related keywords: 51816
- Total number of texts that contain both: 4606
    - This gives 29498 Context Windows 
- Texts after GPT 3.5 search: 3472
- Texts after GPT 4.0 Turbo search: 576

## NeurIPS Scraper and Dataset

To complement our research from Arxiv, we also developed a scraper for the NeurIPS conference papers. Our goal was to identify papers discussing AUPRC and AUROC from NeurIPS between 1987 and 2019.

### How to Download NeurIPS Papers:

1.  Ensure you have Python and necessary libraries (requests, beautifulsoup4, lxml, tqdm) installed.
2.  Use the provided script, setting the start and end year flags to your desired range and specifying the output folder and filename.
3.  Run the script to begin scraping. It will automatically fetch and save metadata and paper texts in a JSONL format.

```bash
python neurips_scraper.py -start 1987 -end 2019 -folder data -filename neurIPS_papers.jsonl
```

### NeurIPS Dataset Overview

1. Papers scraped between 1987 and 2019.
2. Out of a total of 9,680 texts, 78 were found to contain keywords from AUPRC and AUROC.
3. Using GPT-4, 2 relevant papers were identified for our thesis.
    

## Keyword Filtering

1.  **Creation of Keyword Lists:** : Two separate lists of keywords were created for AUROC and AUPRC, respectively. These lists are crucial in identifying relevant papers in the datasets, and can be found in the folder 'keywords'.
2.  **Script-Driven Search:** We used Python scripts to automate the search through the datasets. The scripts scanned the texts for occurrences of the keywords from both lists.
3.  **Dual Mention Filtering:** Papers that mentioned both AUROC and AUPRC were specifically filtered to ensure relevance to our research question. T

## AI-Assisted Review

1.  **Initial Screening with GPT-3.5:** The first round of AI-assisted review utilized OpenAI's GPT-3.5 model. The model was prompted to identify papers that explicitly made claims about the superiority of AUPRC over AUROC in cases of class imbalance.
2.  **Further Refinement with GPT-4.0 Turbo:** A more advanced review was conducted using GPT-4.0 Turbo.
   
## Data Sharing and Collaborative Review

*   **Google Docs for Collaboration:** All identified papers, along with their respective Arxiv IDs and the claims found by GPT-4.0 Turbo, have been compiled in a shared Google document for collaborative review and analysis.
*   **Color-Coding for Agreement:** Team members were encouraged to review the listed papers and color-code them based on their alignment with our research focus: green for papers supporting our interest points and red for those that do not align.

## System Prompt for GPT Assisted Research

To ensure precision and relevance in our literature review process, a specific system prompt was used for the GPT-3.5 and GPT-4.0 Turbo models. This prompt guided the AI in identifying and analyzing papers within the Arxiv dataset that discuss AUROC and AUPRC in the context of machine learning. The prompt's design is crucial for maintaining consistency and accuracy in the AI-assisted review process.

Here is the prompt used for guiding the AI models:

```bash
SYSTEM_PROMPT = (
    "You are an expert in machine learning and scientific literature review. "
    "For each chunk of a published paper (which may have typos, misspellings, and odd characters as a result of conversion from PDF), "
    "examine whether the paper makes any claim that the area under the receiver operating characteristic (AUROC) "
    "is superior as a general performance metric compared to the area under the precision recall curve (AUPRC) in a machine learning setting. "
    "If the paper makes a specific claim favoring AUROC over AUPRC, especially in contexts such as imbalanced datasets, "
    "respond with the format: Paper claim: 'INPUT PAPER CLAIM HERE'. "
    "If the paper makes no such claims, respond with 'NONE'."
)
```

