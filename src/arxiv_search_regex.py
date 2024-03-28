import json
import os
import re
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

def remove_latex_commands(s):
    """
    Removes LaTeX commands from a given string.

    Parameters:
    - s (str): A string potentially containing LaTeX commands.

    Returns:
    - str: The input string with LaTeX commands removed.

    Behavior:
    - This function cleans the input string by removing common LaTeX formatting commands,
      such as command keywords preceded by backslashes, and content within LaTeX math environments.
    - It aims to leave plain text content unaffected, making it more readable or suitable for further processing.
    """
    if s is None:
        return ''
    s = re.sub(r'\\[nrt]|[\n\r\t]', ' ', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'\\.', '', s)
    s = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', s, flags=re.DOTALL)
    s = re.sub(r'\$.*?\$', '', s)
    s = re.sub(r'\\[.*?\\]', '', s)
    s = re.sub(r'\\\(.*?\\\)', '', s)
    s = re.sub(r'\\\[.*?\\\]', '', s)
    s = re.sub(r'(?<=\W)\\|\\(?=\W)', '', s)
    return s.strip()

def process_file(file_path, auroc_regex, auprc_regex, metadata_keys, remove_latex):
    """
    Processes a single file to extract relevant information based on regex patterns and optionally removes LaTeX commands.

    Parameters:
    - file_path (str): Path to the file to be processed.
    - auroc_regex (compiled regex): A compiled regex pattern to search for AUROC mentions.
    - auprc_regex (compiled regex): A compiled regex pattern to search for AUPRC mentions.
    - metadata_keys (list of str): A list of keys to extract metadata from the file entries.
    - remove_latex (bool): Whether to remove LaTeX commands from the text.

    Returns:
    - tuple: A tuple containing two elements:
        1. A list of dictionaries with processed data from the file.
        2. The total number of texts processed.

    Behavior:
    - Reads and processes each line of the file assuming it is in JSON Lines format.
    - Searches for specified patterns and extracts metadata, handling LaTeX commands based on the remove_latex parameter.
    - Aggregates results into a list and counts the total number of processed texts.
    """
    output_data = []
    total_texts = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            total_texts += 1
            try:
                entry = json.loads(line)
                text = entry['text']
                if remove_latex:
                    text = remove_latex_commands(text)
                meta_data = entry.get('meta', {})

                contains_auroc = auroc_regex.search(text) is not None
                contains_auprc = auprc_regex.search(text) is not None

                if contains_auroc or contains_auprc:
                    row_data = {key: meta_data.get(key, None) for key in metadata_keys}
                    row_data['text'] = text
                    row_data['contains_auroc'] = contains_auroc
                    row_data['contains_auprc'] = contains_auprc

                    output_data.append(row_data)

            except json.JSONDecodeError as e:
                print(f"Error loading line in {file_path}: {line}. Error: {e}")

    return output_data, total_texts

def jsonl_folder_filtering(input_folder_path, auroc_regex, auprc_regex, metadata_keys=[], output_folder_path=None, remove_latex=True, save_file=True, filename="filtered_data.json", total_texts_filename="total_texts.txt"):
    """
    Filters files in a folder for specific patterns using multiprocessing, and optionally removes LaTeX commands from the text.

    Parameters:
    - input_folder_path (str): Path to the folder containing .jsonl files to be processed.
    - auroc_regex (compiled regex): Regex pattern for AUROC mentions.
    - auprc_regex (compiled regex): Regex pattern for AUPRC mentions.
    - metadata_keys (list of str, optional): Keys for metadata extraction. Defaults to an empty list.
    - output_folder_path (str, optional): Destination folder path for saving the filtered data. If None, data is not saved to a file.
    - remove_latex (bool, optional): Whether to remove LaTeX commands from the text. Defaults to True.
    - save_file (bool, optional): Whether to save the filtered data and total texts count to files. Defaults to True.
    - filename (str, optional): Filename for saving the filtered data. Defaults to "filtered_data.json".
    - total_texts_filename (str, optional): Filename for saving the total texts count. Defaults to "total_texts.txt".

    Returns:
    - pandas.DataFrame: A DataFrame containing the filtered data.

    Behavior:
    - Processes all .jsonl files found in the specified folder in parallel.
    - Applies regex filtering and LaTeX command removal based on parameters.
    - Compiles the results into a DataFrame, optionally saving it and the total texts count to files.
    """
    file_paths = [os.path.join(input_folder_path, file_name) for file_name in os.listdir(input_folder_path) if file_name.endswith(".jsonl")]

    # Set the number of processes to 6 explicitly
    num_processes = 6

    process_partial = partial(process_file, auroc_regex=auroc_regex, auprc_regex=auprc_regex, metadata_keys=metadata_keys, remove_latex=remove_latex)
    with Pool(num_processes) as p:
        results = p.map(process_partial, file_paths)

    output_data = [item for sublist, _ in results for item in sublist]
    total_texts = sum(total for _, total in results)

    df_output = pd.DataFrame(output_data)
    df_output['text_id'] = pd.factorize(df_output['text'])[0]
    keyword_columns = ['contains_auroc', 'contains_auprc']
    column_order = ['text', 'text_id'] + metadata_keys + keyword_columns
    df_output = df_output[column_order]

    if save_file and output_folder_path is not None:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        with open(os.path.join(output_folder_path, total_texts_filename), 'w') as f:
            f.write(str(total_texts))
        df_output.to_csv(os.path.join(output_folder_path, filename), index=False)
    elif save_file:
        print("Warning: Output folder path is not provided. The DataFrame is not saved to a file.")

    return df_output