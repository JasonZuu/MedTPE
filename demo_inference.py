import argparse

from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

from tpe.tpe_tokenizer_fast import TPETokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a MedTPE model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct",
        help="Original pretrained model path.",
    )
    parser.add_argument(
        "--sft_model_name",
        type=str,
        default="data/MedTPE_data/sft_models/Qwen2.5-1.5B-Instruct_task-cmedqa2_maxN-2_maxM-5000",
        help="Fine-tuned MedTPE model path.",
    )
    return parser.parse_args()


PROMPT = """## Data
A 42-year-old patient has had a dry cough, mild fever, sore throat, and fatigue for the past two days. The patient has no chest pain, shortness of breath, or known chronic lung disease. What are the likely causes, and what should the patient do next?

## Output Instructions
1. Analyze the patient's symptoms and main clinical concern.
2. Provide a concise, medically relevant answer.
3. Return plain English text.
"""


if __name__ == "__main__":
    args = parse_args()

    tpe_tokenizer = TPETokenizerFast.from_pretrained(args.sft_model_name)
    orig_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name)

    tokens_new = tpe_tokenizer.tokenize(PROMPT)
    tokens_orig = orig_tokenizer.tokenize(PROMPT)
    print("Tokens in MedTPE tokenizer:", tokens_new[:50])
    print("Tokens in original tokenizer:", tokens_orig[:50])
    print("Length of MedTPE tokens:", len(tokens_new))
    print("Length of original tokens:", len(tokens_orig))

    llm = LLM(args.sft_model_name, enforce_eager=True)
    llm.set_tokenizer(tpe_tokenizer)

    prompt_template = [{"role": "user", "content": PROMPT}]
    sampling_params = SamplingParams(
        n=1,
        max_tokens=1024,
        top_p=1.0,
        skip_special_tokens=True,
    )

    responses = llm.chat([prompt_template], sampling_params=sampling_params)
    for response in responses:
        for output in response.outputs:
            print(output.text)
