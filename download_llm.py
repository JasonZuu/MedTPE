from huggingface_hub import snapshot_download
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="The repository ID of the model to download.")
    parser.add_argument("--local_dir", type=str, default="data/hf_models", help="Local directory to save the downloaded model.")
    return parser.parse_args()

args = parse_args()
local_dir = Path(args.local_dir) / "--".join(args.repo_id.split("/"))  # download to a subfolder
local_dir.mkdir(parents=True, exist_ok=True)  # create the directory if it does not exist

snapshot_download(
    repo_id=args.repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # forbid symlinks
    revision="main"                # the branch name or commit hash to download
)
