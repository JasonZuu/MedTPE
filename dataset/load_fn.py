from datasets import load_dataset
from dataset.map_fn import ectsum_sample_mapping_fn


def load_ectsum_hf_dataset(data_files: dict,
                           llm_tokenizer,
                           input_max_length,
                           data_split: str,
                           num_workers=None):
    """
    Load ECTSum JSON dataset and map to chat prompts for summarization.
    """
    dataset = load_dataset("json", data_files=data_files, split=data_split)
    remove_columns = [column for column in ["sample_id", "text", "summary"] if column in dataset.column_names]
    dataset = dataset.map(
        ectsum_sample_mapping_fn,
        batched=True,
        fn_kwargs={
            "llm_tokenizer": llm_tokenizer,
            "max_input_length": input_max_length,
        },
        num_proc=num_workers,
        remove_columns=remove_columns
    )
    return dataset
