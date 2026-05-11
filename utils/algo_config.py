from dataclasses import dataclass


@dataclass
class LLMConfig:
    log_dir = 'log/'
    device = 'cuda'
    llm_name = 'data/hf_models/Qwen--Qwen2.5-1.5B-Instruct'
    max_model_len = None # depends on the model
    max_input_len = 4*1024  # 24k
    output_token_len = 8*1024  # 8k
    temperature=0.7
