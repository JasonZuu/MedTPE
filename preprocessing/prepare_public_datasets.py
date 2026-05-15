from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


KAGGLE_DATASET = "mingchengzhu/medtpe-data"


@dataclass(frozen=True)
class DatasetSpec:
    output_dir: str
    required_files: tuple[str, ...]
    aliases: tuple[str, ...]


DATASETS = {
    "cmedqa2": DatasetSpec(
        output_dir="cmedqa2",
        required_files=("train.json", "validation.json", "test.json"),
        aliases=("cmedqa2",),
    ),
    "ectsum": DatasetSpec(
        output_dir="ectsum",
        required_files=("train.json", "val.json", "test.json"),
        aliases=("ectsum",),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download public datasets used by the MedTPE public workflow from Kaggle."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "cmedqa2", "ectsum"],
        default="all",
        help="Dataset to prepare. Use 'all' to prepare both public datasets.",
    )
    parser.add_argument(
        "--output_root",
        default="data/medtpe_data",
        help="Directory where dataset folders will be written.",
    )
    parser.add_argument(
        "--kaggle_dataset",
        default=KAGGLE_DATASET,
        help="Kaggle dataset slug to download.",
    )
    return parser.parse_args()


def _download_kaggle_dataset(dataset_slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'kagglehub'. Install it with `pip install kagglehub` "
            "or rerun `pip install -r requirements.txt`."
        ) from exc

    return Path(kagglehub.dataset_download(dataset_slug))


def _contains_required_files(path: Path, required_files: tuple[str, ...]) -> bool:
    return all((path / filename).is_file() for filename in required_files)


def _find_dataset_dir(download_root: Path, spec: DatasetSpec) -> Path:
    direct_candidates = [
        download_root / spec.output_dir,
        download_root / "medtpe_data" / spec.output_dir,
        download_root / "data" / "medtpe_data" / spec.output_dir,
    ]
    for candidate in direct_candidates:
        if _contains_required_files(candidate, spec.required_files):
            return candidate

    if _contains_required_files(download_root, spec.required_files):
        return download_root

    alias_set = {alias.lower() for alias in spec.aliases}
    for candidate in download_root.rglob("*"):
        if (
            candidate.is_dir()
            and candidate.name.lower() in alias_set
            and _contains_required_files(candidate, spec.required_files)
        ):
            return candidate

    expected = ", ".join(spec.required_files)
    aliases = ", ".join(spec.aliases)
    raise FileNotFoundError(
        f"Could not find dataset files ({expected}) under {download_root}. "
        f"Looked for directories named one of: {aliases}."
    )


def _copy_dataset(source_dir: Path, output_root: Path, spec: DatasetSpec) -> None:
    output_dir = output_root / spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in spec.required_files:
        shutil.copy2(source_dir / filename, output_dir / filename)
    print(f"Wrote {spec.output_dir} files to {output_dir}")


def prepare_dataset(dataset_name: str, download_root: Path, output_root: Path) -> None:
    spec = DATASETS[dataset_name]
    source_dir = _find_dataset_dir(download_root, spec)
    _copy_dataset(source_dir, output_root, spec)


if __name__ == "__main__":
    args = parse_args()
    output_root = Path(args.output_root)
    download_root = _download_kaggle_dataset(args.kaggle_dataset)

    if args.dataset in ("all", "cmedqa2"):
        prepare_dataset("cmedqa2", download_root, output_root)
    if args.dataset in ("all", "ectsum"):
        prepare_dataset("ectsum", download_root, output_root)
