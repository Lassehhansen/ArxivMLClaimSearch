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

### NeurIPS Search Data Description V1

1. Papers scraped between 1987 and 2019.
2. Out of a total of 9,680 texts, 91 were found to contain keywords from AUPRC and AUROC.
3. Using GPT-4, 6 relevant papers were identified to be relevant.
   
### NeurIPS Search Data Description V2

1. Papers scraped between 1987 and 2019.
2. Out of a total of 9,680 texts, 78 were found to contain keywords from AUPRC and AUROC.
3. Using GPT-4, 6 relevant papers were identified for our thesis.

## Keyword Filtering -> This is Version 1

1.  **Creation of Keyword Lists:** : Two separate lists of keywords were created for AUROC and AUPRC, respectively. These lists are crucial in identifying relevant papers in the datasets, and can be found in the folder 'keywords'.
2.  **Script-Driven Search:** We used Python scripts to automate the search through the datasets. The scripts scanned the texts for occurrences of the keywords from both lists.
3.  **Dual Mention Filtering:** Papers that mentioned both AUROC and AUPRC were specifically filtered to ensure relevance to our research question.

## Regex-Based Filtering - Version 2

1.  **Compiling Regular Expressions** : We developed two sets of regular expressions tailored for AUROC and AUPRC, respectively. These expressions are designed to capture variations and contexts in which these terms appear within the texts, thereby improving the precision of our search. The compiled regex patterns for both AUROC and AUPRC can be found in the regex_definitions.py module.
2. **Automated Regex Search** : Utilizing Python, we implemented scripts that leverage the re library to systematically search the datasets. These scripts employ the compiled regular expressions to identify instances of AUROC and AUPRC mentions, accounting for the diverse ways these terms can be presented in the literature.
3. **Contextual and Dual Mention Identification** : To enhance the relevance of our findings, we not only looked for papers that mention either AUROC or AUPRC but also employed additional logic to filter for documents that discuss both terms. This step ensures that the selected papers are highly pertinent to our research objectives. Furthermore, by applying regex, we're able to extract and analyze the context surrounding these mentions, providing deeper insights into how these metrics are discussed and applied in the field. 

## AI-Assisted Review

1.  **Initial Screening with GPT-3.5:** The first round of AI-assisted review utilized OpenAI's GPT-3.5 model. The model was prompted to identify papers that explicitly made claims about the superiority of AUPRC over AUROC in cases of class imbalance.
2.  **Further Refinement with GPT-4.0 Turbo:** A more advanced review was conducted using GPT-4.0 Turbo.
   
## Data Sharing and Collaborative Review

*   **Google Docs for Collaboration:** All identified papers, along with their respective Arxiv IDs and the claims found by GPT-4 Turbo, have been compiled in a shared Google document for collaborative review and analysis.
*   **Color-Coding for Agreement:** Team members were encouraged to review the listed papers and color-code them based on their alignment with our research focus: green for papers supporting our interest points and red for those that do not align.

## Model versions used

- "gpt-3.5-turbo-1106"
- "gpt-4-0125-preview"


## System Prompt for GPT Assisted Research

To ensure precision and relevance in our literature review process, a specific system prompt was used for the GPT-3.5 and GPT-4.0 Turbo models. This prompt guided the AI in identifying and analyzing papers within the Arxiv dataset that discuss AUROC and AUPRC in the context of machine learning. The prompt's design is crucial for maintaining consistency and accuracy in the AI-assisted review process.

### System Prompt GPT 3.5 Assisted Research

```bash
SYSTEM_PROMPT GPT 3.5 = """
You are an expert in machine learning and scientific literature review.
For each chunk of a published paper (which may have typos, misspellings, and odd characters as a result of conversion from PDF), return a JSON object that states whether or not the paper makes any claim that the area under the precision recall curve (AUPRC) is superior or inferior as a general performance metric to the area under the receiver operating characteristic (AUROC) in an ML setting, in particular for imbalanced classification problems. A paper claiming that a model performs better under AUPRC vs. AUROC is *not* an example of this; instead a paper claiming that AUPRC should be used instead of AUROC in cases of class imbalance is an example of this metric commentary. Respond with format {"claims": [{"claim": DESCRIPTION OF CLAIM, "evidence_quote": SUBSTRING FROM INPUT STATING CLAIM}, ...]}. If the paper makes no claims, leave the "claims" key in the JSON object empty. If the claim made is that the AUPRC is superior to the AUROC in the case of class imbalance, use the string "AUPRC is superior to AUROC for imbalanced data" for the description of the claim. For other claims, use any appropriate free-text description.

Examples: 

Input: "AUROC: The horizontal and vertical coordinates of the Receiver Operating Characteristic (ROC) curve are the FPR and TPR, and the curve is obtained by calculating the FPR and TPR under multiple sets of thresholds. The area of the region enclosed by the ROC curve and the horizontal axis is often used to evaluate binary classification tasks, denoted as AUROC. The value of AUROC is within the range of [0, 1], and higher values indicate better performance. AUROC can visualize the generalization performance of the GVAED model and help to select the best alarm threshold In addition, the Equal Error Rate (EER), i.e., the proportion of incorrectly classified frames when TPR and FNR are equal, is also used to measure the performance of anomaly detection models. AP: Due to the highly unbalanced nature of positive and negative samples in GVAED tasks, i.e., the TN is usually larger than the TP, researchers think that the area under the Precision-Recall (PR) curve is more suitable for evaluating GVAED models, denoted as AP. The horizontal coordinates of the PR curve are the Recall (i.e., the TPR in Eq. 4), while the vertical coordinate represents the Precision, defined as Precision = TP TP+FP . A point on the PR curve corresponds to the Precision and Recall values at a certain threshold."
Output: {"claims": [{"claim": "AUPRC is superior to AUROC for imbalanced data", "evidence_quote": "Due to the highly unbalanced nature of positive and negative samples in GVAED tasks, i.e., the TN is usually larger than the TP, researchers think that the area under the Precision-Recall (PR) curve is more suitable for evaluating GVAED models, denoted as AP"}]}

Input: "As seen, it outperforms other approaches except in the cases of TinyImageNet for CIFAR-100. Our approach still has better AUROC, but the detection error and FPR at 95% TPR are slightly larger than ODIN’s. Interestingly, the MD approach is worse than max-softmax in some cases. Such a result has also been reporte"
Output: {"claims": []}

Input: "AUC-ROC measures the class separability at various threshold settings. ROC is the probability curve and AUC represents the degree of measures of separability. It compares true positive rate (sensitivity/recall) versus the false positive rate (1 - specificity). The higher the AUC-ROC, the bigger the distinction between the true positive and false negative. • AUC-PR: It combines the precision and recall, for various threshold values, it compares the positively predicted value (precision) vs the true positive rate (recall). Both precision and recall focus on the positive class (the lesion) and unconcerned about the true negative (not a lesion, which is the majority class). Thus, for class imbalance, PR is more suitable than ROC. The higher the AUC-PR, the better the model performance"
Output: {"claims": [{"claim": "AUPRC is superior to AUROC for imbalanced data", "evidence_quote": "Thus, for class imbalance, PR is more suitable than ROC"}]}

So please for each chunk of the input, return a JSON object that states whether or not the paper makes any claim that the area under the precision recall curve (AUPRC) is superior or inferior as a general performance metric to the area under the receiver operating characteristic (AUROC) in an ML setting, in particular for imbalanced classification problems. 
"""
```

### System Prompt GPT 4.0 Assisted Research

```bash
SYSTEM_PROMPT GPT 4.0 = """
You are an expert in machine learning and scientific literature review.
For each chunk of a published paper (which may have typos, misspellings, and odd characters as a result of conversion from PDF), return a JSON object that states whether or not the paper makes any claim that the area under the precision recall curve (AUPRC) is superior or inferior as a general performance metric to the area under the receiver operating characteristic (AUROC) in an ML setting, in particular for imbalanced classification problems. A paper claiming that a model performs better under AUPRC vs. AUROC is *not* an example of this; instead a paper claiming that AUPRC should be used instead of AUROC in cases of class imbalance is an example of this metric commentary. Respond with format {"claims": [{"claim": DESCRIPTION OF CLAIM, "evidence_quote": SUBSTRING FROM INPUT STATING CLAIM}, ...]}. If the paper makes no claims, leave the "claims" key in the JSON object empty. If the claim made is that the AUPRC is superior to the AUROC in the case of class imbalance, use the string "AUPRC is superior to AUROC for imbalanced data" for the description of the claim. For other claims, use any appropriate free-text description.

Examples: 

Input: "AUROC: The horizontal and vertical coordinates of the Receiver Operating Characteristic (ROC) curve are the FPR and TPR, and the curve is obtained by calculating the FPR and TPR under multiple sets of thresholds. The area of the region enclosed by the ROC curve and the horizontal axis is often used to evaluate binary classification tasks, denoted as AUROC. The value of AUROC is within the range of [0, 1], and higher values indicate better performance. AUROC can visualize the generalization performance of the GVAED model and help to select the best alarm threshold In addition, the Equal Error Rate (EER), i.e., the proportion of incorrectly classified frames when TPR and FNR are equal, is also used to measure the performance of anomaly detection models. AP: Due to the highly unbalanced nature of positive and negative samples in GVAED tasks, i.e., the TN is usually larger than the TP, researchers think that the area under the Precision-Recall (PR) curve is more suitable for evaluating GVAED models, denoted as AP. The horizontal coordinates of the PR curve are the Recall (i.e., the TPR in Eq. 4), while the vertical coordinate represents the Precision, defined as Precision = TP TP+FP . A point on the PR curve corresponds to the Precision and Recall values at a certain threshold."
Output: {"claims": [{"claim": "AUPRC is superior to AUROC for imbalanced data", "evidence_quote": "Due to the highly unbalanced nature of positive and negative samples in GVAED tasks, i.e., the TN is usually larger than the TP, researchers think that the area under the Precision-Recall (PR) curve is more suitable for evaluating GVAED models, denoted as AP"}]}

Input: "As seen, it outperforms other approaches except in the cases of TinyImageNet for CIFAR-100. Our approach still has better AUROC, but the detection error and FPR at 95% TPR are slightly larger than ODIN’s. Interestingly, the MD approach is worse than max-softmax in some cases. Such a result has also been reporte"
Output: {"claims": []}

Input: "AUC-ROC measures the class separability at various threshold settings. ROC is the probability curve and AUC represents the degree of measures of separability. It compares true positive rate (sensitivity/recall) versus the false positive rate (1 - specificity). The higher the AUC-ROC, the bigger the distinction between the true positive and false negative. • AUC-PR: It combines the precision and recall, for various threshold values, it compares the positively predicted value (precision) vs the true positive rate (recall). Both precision and recall focus on the positive class (the lesion) and unconcerned about the true negative (not a lesion, which is the majority class). Thus, for class imbalance, PR is more suitable than ROC. The higher the AUC-PR, the better the model performance"
Output: {"claims": [{"claim": "AUPRC is superior to AUROC for imbalanced data", "evidence_quote": "Thus, for class imbalance, PR is more suitable than ROC"}]}
"""
```

### User Prompt GPT 4.0 Assisted Research

Additionally, a user prompt was introduced in the GPT 4.0 search to refine the search criteria further. This includes an introduction statement before the context window of a text and a concluding statement afterward.

```bash
introduction_statement_prompt = """
Please carefully review the following text. We are specifically looking for claims where AUPRC is argued to be a superior metric to AUROC, especially in cases of class imbalance in machine learning applications. Any claim that discusses the preference of AUPRC over AUROC due to its effectiveness in such scenarios should be returned in the a JSON object. If no such claims are found, please leave the 'claims' key empty. Here is the text:
"""
end_statement_prompt = """
If you find any claim asserting the superiority of AUPRC over AUROC for imbalanced datasets, please provide your findings in a JSON object with the key 'claims'. Each claim should be a dictionary with 'claim' and 'evidence_quote' as keys, like this: {"claims": [{"claim": "DESCRIPTION OF CLAIM", "evidence_quote": "SUBSTRING FROM INPUT STATING CLAIM"}]}. If no relevant claims are found, the 'claims' key should have an empty list.
"""
```

