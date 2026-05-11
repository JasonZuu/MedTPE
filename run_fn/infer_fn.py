import polars as pl
import torch

from utils.algo_config import LLMConfig


@torch.no_grad()
def infer_llm_on_dataset(llm, 
                         dataset, 
                         sampling_params, 
                         algo_config: LLMConfig,
                         n_samples:int=None)->pl.DataFrame:
    """
    Perform inference on a dataset using the provided LLM.

    Args:
        llm (LLM): The LLM object to use for generation.
        dataset (Dataset): The dataset to infer on.
        sampling_params (SamplingParams): Sampling parameters for generation.
        algo_config (LLMConfig): The algorithm configuration.

    Returns:
        list: A list of dictionaries containing the messages, label, and generated text.
    """
    if n_samples is None:
        messages = dataset["message"]
        labels = dataset["label"]
    else:
        n_samples = min(n_samples, len(dataset["message"]))
        messages = dataset["message"][:n_samples]
        labels = dataset["label"][:n_samples]
    results = {"message": messages, "label": labels}
    if type(messages) is not list:
        messages = list(messages)
    if type(labels) is not list:
        labels = list(labels)

    responses = llm.chat(messages=messages, sampling_params=sampling_params)
    for i_response in range(len(responses[0].outputs)):
        generated_texts = [response.outputs[i_response].text for response in responses]
        results[f"generated_text_{i_response}"] = generated_texts

    results_df = pl.DataFrame(results)
    
    return results_df
