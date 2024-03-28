import openai
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time


def process_with_gpt_with_retries(context_window, model, system_prompt, introduction_statement_prompt, end_statement_prompt, openai_api_key, max_retries=5):
    """
    Attempts to generate a response from the OpenAI API for a given context window,
    using specified model parameters. Handles rate limits with retries.

    Parameters:
    - context_window (str): The context or prompt to send to the model.
    - model (str): The model identifier (e.g., "text-davinci-003") for completion.
    - system_prompt (str): System-level instructions or context provided alongside the user prompt.
    - openai_api_key (str): The API key for authenticating with OpenAI's services.
    - max_retries (int, optional): Maximum number of retry attempts if rate limited. Defaults to 5.

    Returns:
    - str: The generated response text if successful, or an error message if an error occurs or retries are exceeded.

    Behavior:
    - Submits a chat completion request to the OpenAI API.
    - Implements an exponential backoff strategy for handling rate limit errors, with a maximum wait time limit.
    - Returns either the response from the model or an error message indicating the failure reason.
    """
    client = OpenAI(api_key=openai_api_key)
    retry_delay = 0.5  # Reduced initial delay in seconds for retries
    max_retry_delay = 16  # Maximum delay, to avoid long waits
    # Modify the context window to include the additional prompts
    modified_context_window = f"{introduction_statement_prompt} {context_window} {end_statement_prompt}"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": modified_context_window}
                ]
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            print(f"Rate limit reached, retrying in {retry_delay} seconds.")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)  # Exponential backoff with max limit
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)  # Exponential backoff with max limit
    return "Error: Max retries exceeded."

def process_all_context_windows(new_df, model, system_prompt, introduction_statement_prompt, end_statement_prompt, openai_api_key, texts_before_pause=1000, pause_duration=5, max_workers=1):
    """
    Processes a DataFrame of context windows to generate model responses in parallel,
    utilizing a specified number of worker threads.

    Parameters:
    - new_df (pandas.DataFrame): DataFrame containing the context windows to process. Must have a 'context_window' column.
    - model (str): Model identifier to use for generating responses.
    - system_prompt (str): System-level instructions or context for the model.
    - openai_api_key (str): API key for OpenAI services authentication.
    - texts_before_pause (int, optional): Number of texts to process before pausing, to manage rate limits or resource usage. Defaults to 1000.
    - pause_duration (int, optional): Duration in seconds to pause after processing `texts_before_pause` texts. Defaults to 5 seconds.
    - max_workers (int, optional): Maximum number of worker threads for parallel processing. Defaults to 1.

    Returns:
    - pandas.DataFrame: The input DataFrame, `new_df`, with an additional 'gpt_response' column containing the generated responses or error messages.

    Behavior:
    - Processes each context window in parallel, up to `max_workers` at a time.
    - Applies rate limit handling and error capturing for each request.
    - Optionally pauses processing after a specified number of texts to avoid overloading the API or the local system.
    - Updates the original DataFrame with responses, allowing for analysis or further processing.
    """
    responses = {}
    processed_texts = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers based on your environment
        future_to_idx = {executor.submit(process_with_gpt_with_retries, row['context_window'], model, system_prompt, introduction_statement_prompt, end_statement_prompt, openai_api_key): idx for idx, row in new_df.iterrows()}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                response = future.result()
            except Exception as exc:
                print(f'Context window at index {idx} generated an exception: {exc}')
                response = "Error: Exception in processing"
            
            responses[idx] = response
            processed_texts += 1

            # Indicator for how many texts have been processed
            if processed_texts % texts_before_pause == 0:
                print(f"Processed {processed_texts}/{len(new_df)} texts; pausing for {pause_duration} seconds...")
                time.sleep(pause_duration)

    # Update the DataFrame with the responses using .loc
    for idx, response in responses.items():
        if idx in new_df.index:
            new_df.loc[idx, 'gpt_response'] = response

    return new_df
