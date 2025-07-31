from datasets import load_dataset
from dataset.map_fn import sample_mapping_fn
import os


def load_hf_dataset(data_files: dict, llm_tokenizer, input_max_length, 
                    data_split:str,
                    pe_method: str = "raw", 
                    num_workers=None):
    instruction = "Now make your prediction."
    if pe_method == "raw":
        pass
    elif pe_method == "cot":
        instruction += " Let think step by step."
    elif pe_method == "sft_qa": # only use for SFT QA Generation
        instruction += "Only output your answer in json without any other content."
    else:
        raise ValueError(f"Invalid pe_method: {pe_method}")
    
    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        columns=['label', "data", "question"],
        cache_dir=os.path.dirname(list(data_files.values())[0],),
        split=data_split
    )

    dataset = dataset.map(
        sample_mapping_fn,
        fn_kwargs={
            "instruction": instruction,
            "llm_tokenizer": llm_tokenizer,
            "max_input_length": input_max_length,
        },
        num_proc=num_workers,
        remove_columns=['data', 'question']  # Remove unnecessary columns
    )
    return dataset
