from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and normalize public datasets used by the MedTPE public workflow."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "cmedqa2", "ectsum"],
        default="all",
        help="Dataset to prepare. Use 'all' to prepare both public datasets.",
    )
    parser.add_argument(
        "--output_root",
        default="data/hf_datasets",
        help="Directory where normalized dataset folders will be written.",
    )
    return parser.parse_args()


def _normalize_string(value) -> str:
    if value is None:
        return ""
    return str(value)


def _save_jsonl(dataset: Dataset, output_path: Path, columns: Iterable[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_columns = list(columns)
    missing = [column for column in keep_columns if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing required columns {missing} in dataset split for {output_path}")

    normalized = dataset.map(
        lambda row: {column: _normalize_string(row[column]) for column in keep_columns},
        remove_columns=[column for column in dataset.column_names if column not in keep_columns],
    )
    normalized.to_json(str(output_path), orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(normalized)} rows to {output_path}")


def prepare_cmedqa2(output_root: Path) -> None:
    from datasets import load_dataset

    dataset_dict = load_dataset("fzkuji/cMedQA2")
    output_dir = output_root / "fzkuji--cMedQA2"
    for split in ["train", "validation", "test"]:
        if split not in dataset_dict:
            raise ValueError(f"fzkuji/cMedQA2 does not provide expected split: {split}")
        _save_jsonl(dataset_dict[split], output_dir / f"{split}.json", ["question", "answer"])


def prepare_ectsum(output_root: Path) -> None:
    from datasets import load_dataset

    dataset_dict = load_dataset("jbrendsel/ECTSum")
    output_dir = output_root / "github--ECTSum"
    split_to_fname = {"train": "train.json", "validation": "val.json", "test": "test.json"}
    for split, fname in split_to_fname.items():
        if split not in dataset_dict:
            raise ValueError(f"jbrendsel/ECTSum does not provide expected split: {split}")
        _save_jsonl(dataset_dict[split], output_dir / fname, ["text", "summary"])


if __name__ == "__main__":
    args = parse_args()
    output_root = Path(args.output_root)

    if args.dataset in ("all", "cmedqa2"):
        prepare_cmedqa2(output_root)
    if args.dataset in ("all", "ectsum"):
        prepare_ectsum(output_root)
