from __future__ import annotations

import argparse
from pathlib import Path
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_id", type=str, default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--response_dir", type=str, default="log/TPE/CMEDQA2")
    parser.add_argument("--log_dir", type=str, default="log/TPE/metrics")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CMEDQA2",
        choices=["MIMICIV", "EHRSHOT", "ECTSUM", "CMEDQA2"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cmedqa2",
        choices=[
            "icu_mortality",
            "icu_phenotyping",
            "guo_readmission",
            "new_pancan",
            "medqa",
            "arc_challenge",
            "hospnote_readmission",
            "hospnote_mdc",
            "ect_summary",
            "cmedqa2",
        ],
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="nl",
        choices=["nl", "json", "yaml", "xml"],
    )
    parser.add_argument(
        "--pe_method",
        type=str,
        default="raw",
        choices=["raw", "cot"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_input_len",
        type=str,
        default="8k",
        choices=["500", "1k", "2k", "4k", "6k", "8k", "12k", "16k", "24k"],
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument(
        "--avg_method",
        type=str,
        default="macro",
        choices=["binary", "macro", "micro"],
    )
    parser.add_argument(
        "--log_fname_postfix",
        type=str,
        default="bpe",
    )
    parser.add_argument("--tokenizer_type", type=str, default="bpe", choices=["bpe", "tpe"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--n_response", type=int, default=1)
    parser.add_argument("--cross_seed_std", action="store_true")
    parser.add_argument("--seed_list", nargs="+", type=int, default=None)
    return parser.parse_args()


def compute_rouge_scores(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    total_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    valid_count = 0
    for pred, ref in zip(preds, refs):
        if not pred or not ref:
            continue
        scores = scorer.score(ref, pred)
        total_scores["rouge1"] += scores["rouge1"].fmeasure
        total_scores["rouge2"] += scores["rouge2"].fmeasure
        total_scores["rougeL"] += scores["rougeL"].fmeasure
        valid_count += 1
    if valid_count == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "valid_count": 0}
    return {
        "rouge1": total_scores["rouge1"] / valid_count,
        "rouge2": total_scores["rouge2"] / valid_count,
        "rougeL": total_scores["rougeL"] / valid_count,
        "valid_count": valid_count,
    }


def compute_bertscore(preds, refs):
    _, _, f1_tensor = bertscore_score(preds, refs, lang="en")
    return {
        "f1": float(f1_tensor.mean().item()),
    }


def evaluate_ectsum_summaries(result_path: str, n_bootstrap: int, seed: int):
    df = pl.read_parquet(result_path)
    if "generated_text_0" not in df.columns or "label" not in df.columns:
        raise ValueError("ECTSUM results must contain 'generated_text_0' and 'label'.")
    preds = df["generated_text_0"].to_list()
    refs = df["label"].to_list()
    filtered_preds = []
    filtered_refs = []
    for pred, ref in zip(preds, refs):
        if isinstance(pred, str) and pred.strip() and isinstance(ref, str) and ref.strip():
            filtered_preds.append(pred.strip())
            filtered_refs.append(ref.strip())
    if not filtered_preds:
        point_metrics = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bertscore_f1": 0.0,
        }
        total_samples = len(preds)
        valid_samples = 0
        invalid_count = total_samples
        valid_ratio = 0.0
        metadata = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "invalid_count": invalid_count,
            "valid_ratio": valid_ratio,
        }
        return {
            "bootstrap_metrics": {},
            "point_metrics": point_metrics,
            "metadata": metadata,
        }
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for pred, ref in zip(filtered_preds, filtered_refs):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
    rouge1_scores = np.array(rouge1_scores, dtype=float)
    rouge2_scores = np.array(rouge2_scores, dtype=float)
    rougeL_scores = np.array(rougeL_scores, dtype=float)
    rouge_stats = {
        "rouge1": float(rouge1_scores.mean()),
        "rouge2": float(rouge2_scores.mean()),
        "rougeL": float(rougeL_scores.mean()),
        "valid_count": len(filtered_preds),
    }
    _, _, f1_tensor = bertscore_score(filtered_preds, filtered_refs, lang="en")
    f1_scores = f1_tensor.cpu().numpy().astype(float)
    bert_stats = {
        "f1": float(f1_scores.mean()),
    }
    total_samples = len(preds)
    valid_samples = rouge_stats["valid_count"]
    invalid_count = total_samples - valid_samples
    valid_ratio = valid_samples / total_samples if total_samples > 0 else 0.0
    point_metrics = {
        "rouge1": rouge_stats["rouge1"],
        "rouge2": rouge_stats["rouge2"],
        "rougeL": rouge_stats["rougeL"],
        "bertscore_f1": bert_stats["f1"],
    }
    bootstrap_metrics = {}
    if n_bootstrap > 0 and valid_samples > 1:
        rng = np.random.default_rng(seed=seed)
        indices = np.arange(valid_samples)
        rouge1_sample_means = []
        rouge2_sample_means = []
        rougeL_sample_means = []
        bert_f1_sample_means = []
        for _ in range(n_bootstrap):
            sample_idx = rng.choice(indices, size=valid_samples, replace=True)
            rouge1_sample_means.append(float(rouge1_scores[sample_idx].mean()))
            rouge2_sample_means.append(float(rouge2_scores[sample_idx].mean()))
            rougeL_sample_means.append(float(rougeL_scores[sample_idx].mean()))
            bert_f1_sample_means.append(float(f1_scores[sample_idx].mean()))
        bootstrap_metrics = {
            "rouge1": {
                "mean": float(np.mean(rouge1_sample_means)),
                "std": float(np.std(rouge1_sample_means)),
            },
            "rouge2": {
                "mean": float(np.mean(rouge2_sample_means)),
                "std": float(np.std(rouge2_sample_means)),
            },
            "rougeL": {
                "mean": float(np.mean(rougeL_sample_means)),
                "std": float(np.std(rougeL_sample_means)),
            },
            "bertscore_f1": {
                "mean": float(np.mean(bert_f1_sample_means)),
                "std": float(np.std(bert_f1_sample_means)),
            },
        }
    metadata = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_count": invalid_count,
        "valid_ratio": valid_ratio,
    }
    return {
        "bootstrap_metrics": bootstrap_metrics,
        "point_metrics": point_metrics,
        "metadata": metadata,
    }


def _infer_tokenizer_path(tokenizer_path: str | None, llm_id: str) -> str:
    if tokenizer_path is not None and tokenizer_path.strip():
        return tokenizer_path.strip()
    if Path(llm_id).exists():
        return llm_id
    hf_models_dir = Path("data/hf_models")
    if hf_models_dir.exists():
        candidates = sorted(hf_models_dir.glob(f"*--{llm_id}"))
        if candidates:
            return str(candidates[0])
        candidates = sorted(hf_models_dir.glob(llm_id))
        if candidates:
            return str(candidates[0])
    return llm_id


def _compute_valid_flags_ectsum(df: pl.DataFrame) -> np.ndarray:
    total_samples = df.height
    flags = np.zeros(total_samples, dtype=bool)
    if "generated_text_0" not in df.columns or "label" not in df.columns:
        return flags
    preds = df["generated_text_0"].to_list()
    refs = df["label"].to_list()
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        flags[i] = isinstance(pred, str) and pred.strip() and isinstance(ref, str) and ref.strip()
    return flags


def _compute_valid_flags_classification(df: pl.DataFrame, n_response: int) -> np.ndarray:
    total_samples = df.height
    flags = np.zeros(total_samples, dtype=bool)
    for row_idx, row in enumerate(df.iter_rows(named=True)):
        for i in range(n_response):
            col_name = f"generated_text_{i}"
            if col_name not in row or row[col_name] is None:
                continue
            is_valid, _ = parse_llm_output(row[col_name])
            if is_valid:
                flags[row_idx] = True
                break
    return flags


def _compute_output_token_lengths(df: pl.DataFrame, n_response: int, tokenizer) -> np.ndarray:
    total_samples = df.height
    lengths = np.zeros(total_samples, dtype=float)
    counts = np.zeros(total_samples, dtype=int)
    texts = []
    row_indices = []
    for i in range(n_response):
        col_name = f"generated_text_{i}"
        if col_name not in df.columns:
            continue
        col_values = df[col_name].to_list()
        for row_idx, text in enumerate(col_values):
            if isinstance(text, str) and text.strip():
                texts.append(text)
                row_indices.append(row_idx)
    if not texts:
        return lengths
    encoded = tokenizer(texts, add_special_tokens=False)
    token_counts = [len(ids) for ids in encoded["input_ids"]]
    for row_idx, tok_len in zip(row_indices, token_counts):
        lengths[row_idx] += tok_len
        counts[row_idx] += 1
    for i in range(total_samples):
        if counts[i] > 0:
            lengths[i] = lengths[i] / counts[i]
    return lengths


def _bootstrap_mean_std(values: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    n = int(values.size)
    if n_bootstrap <= 0 or n <= 1:
        return {}
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(n)
    sample_means = []
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        sample_means.append(float(values[sample_idx].mean()))
    return {
        "mean": float(np.mean(sample_means)),
        "std": float(np.std(sample_means)),
    }


def _is_text_task(dataset: str, task: str) -> bool:
    return dataset in ["ECTSUM", "CMEDQA2"] or task in ["ect_summary", "cmedqa2"]


def _evaluate_task_metrics(response_file: str, args, n_bootstrap: int) -> dict:
    if _is_text_task(args.dataset, args.task):
        return evaluate_ectsum_summaries(response_file, n_bootstrap=n_bootstrap, seed=args.seed)
    task_to_num_classes = {
        "icu_mortality": (2, False),
        "icu_phenotyping": (25, True),
        "hospnote_readmission": (2, False),
        "hospnote_mdc": (25, False),
        "guo_readmission": (2, False),
        "new_pancan": (2, False),
        "medqa": (5, False),
        "arc_challenge": (4, False),
    }
    if args.task not in task_to_num_classes:
        raise ValueError(f"Unknown task: {args.task}")
    num_classes, is_multi_label = task_to_num_classes[args.task]
    return llm_test_fn(
        response_file,
        num_classes=num_classes,
        n_response=args.n_response,
        avg_method=args.avg_method,
        n_bootstrap=n_bootstrap,
        random_seed=args.seed,
        use_bootstrap=n_bootstrap > 0,
        is_multi_label=is_multi_label,
    )


def _attach_common_metrics(test_metrics: dict, response_file: str, args, n_bootstrap: int) -> dict:
    df = pl.read_parquet(response_file)
    tokenizer_src = _infer_tokenizer_path(args.tokenizer_path, args.llm_id)
    if args.tokenizer_type == "tpe":
        tokenizer = TPETokenizerFast.from_pretrained(tokenizer_src)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    output_token_lens = _compute_output_token_lengths(df, args.n_response, tokenizer)
    output_token_len_point = float(output_token_lens.mean()) if df.height > 0 else 0.0
    output_token_len_boot = _bootstrap_mean_std(output_token_lens, n_bootstrap, args.seed)
    if _is_text_task(args.dataset, args.task):
        valid_flags = _compute_valid_flags_ectsum(df)
    else:
        valid_flags = _compute_valid_flags_classification(df, args.n_response)
    fcr_point = float(valid_flags.mean()) if df.height > 0 else 0.0
    fcr_boot = _bootstrap_mean_std(valid_flags.astype(float), n_bootstrap, args.seed)
    test_metrics.setdefault("point_metrics", {})
    test_metrics["point_metrics"]["fcr"] = fcr_point
    test_metrics["point_metrics"]["output_token_len"] = output_token_len_point
    test_metrics.setdefault("bootstrap_metrics", {})
    if fcr_boot:
        test_metrics["bootstrap_metrics"]["fcr"] = fcr_boot
    if output_token_len_boot:
        test_metrics["bootstrap_metrics"]["output_token_len"] = output_token_len_boot
    return test_metrics


def _format_results(test_metrics: dict, args, use_cross_seed_std: bool) -> dict:
    results = {}
    is_text_task = _is_text_task(args.dataset, args.task)
    point = test_metrics["point_metrics"]
    boot = test_metrics.get("bootstrap_metrics", {})
    cross = test_metrics.get("cross_seed_metrics", {})
    if is_text_task:
        if use_cross_seed_std:
            results["ROUGE-1"] = f"{cross['rouge1']['mean']:.4f} ({cross['rouge1']['std']:.4f})"
            results["ROUGE-2"] = f"{cross['rouge2']['mean']:.4f} ({cross['rouge2']['std']:.4f})"
            results["ROUGE-L"] = f"{cross['rougeL']['mean']:.4f} ({cross['rougeL']['std']:.4f})"
            results["BERTScore_F1"] = f"{cross['bertscore_f1']['mean']:.4f} ({cross['bertscore_f1']['std']:.4f})"
            results["FCR"] = f"{cross['fcr']['mean']:.3f} ({cross['fcr']['std']:.3f})"
            results["OutTokLen"] = f"{cross['output_token_len']['mean']:.1f} ({cross['output_token_len']['std']:.1f})"
            return results
        if boot:
            results["ROUGE-1"] = f"{boot['rouge1']['mean']:.4f} ({boot['rouge1']['std']:.4f})"
            results["ROUGE-2"] = f"{boot['rouge2']['mean']:.4f} ({boot['rouge2']['std']:.4f})"
            results["ROUGE-L"] = f"{boot['rougeL']['mean']:.4f} ({boot['rougeL']['std']:.4f})"
            results["BERTScore_F1"] = f"{boot['bertscore_f1']['mean']:.4f} ({boot['bertscore_f1']['std']:.4f})"
            results["FCR"] = f"{boot['fcr']['mean']:.3f} ({boot['fcr']['std']:.3f})" if "fcr" in boot else f"{test_metrics['metadata']['valid_ratio']:.3f}"
            results["OutTokLen"] = f"{boot['output_token_len']['mean']:.1f} ({boot['output_token_len']['std']:.1f})" if "output_token_len" in boot else f"{point['output_token_len']:.1f}"
            return results
        results["ROUGE-1"] = f"{point['rouge1']:.4f}"
        results["ROUGE-2"] = f"{point['rouge2']:.4f}"
        results["ROUGE-L"] = f"{point['rougeL']:.4f}"
        results["BERTScore_F1"] = f"{point['bertscore_f1']:.4f}"
        results["FCR"] = f"{test_metrics['metadata']['valid_ratio']:.3f}"
        results["OutTokLen"] = f"{point['output_token_len']:.1f}"
        return results
    if use_cross_seed_std:
        results["f1"] = f"{cross['f1']['mean']:.3f} ({cross['f1']['std']:.3f})"
        results["FCR"] = f"{cross['fcr']['mean']:.3f} ({cross['fcr']['std']:.3f})"
        results["OutTokLen"] = f"{cross['output_token_len']['mean']:.1f} ({cross['output_token_len']['std']:.1f})"
        return results
    if "f1" in boot:
        results["f1"] = f"{boot['f1']['mean']:.3f} ({boot['f1']['std']:.3f})"
    else:
        results["f1"] = f"{point['f1']:.3f}"
    results["FCR"] = f"{boot['fcr']['mean']:.3f} ({boot['fcr']['std']:.3f})" if "fcr" in boot else f"{test_metrics['metadata']['valid_ratio']:.3f}"
    results["OutTokLen"] = f"{boot['output_token_len']['mean']:.1f} ({boot['output_token_len']['std']:.1f})" if "output_token_len" in boot else f"{point['output_token_len']:.1f}"
    return results


def _aggregate_cross_seed_metrics(seed_metrics: list) -> dict:
    first_metrics = seed_metrics[0][1]
    aggregated = {"metadata": dict(first_metrics.get("metadata", {})), "point_metrics": {}, "bootstrap_metrics": {}, "cross_seed_metrics": {}, "seed_metrics": {}}
    metric_keys = set()
    for _, metrics in seed_metrics:
        metric_keys.update(metrics.get("point_metrics", {}).keys())
    for key in metric_keys:
        values = [float(metrics["point_metrics"][key]) for _, metrics in seed_metrics if key in metrics.get("point_metrics", {})]
        if not values:
            continue
        aggregated["point_metrics"][key] = float(np.mean(values))
        aggregated["cross_seed_metrics"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    for seed, metrics in seed_metrics:
        aggregated["seed_metrics"][str(seed)] = metrics.get("point_metrics", {})
    return aggregated


if __name__ == "__main__":
    args = get_args()

    import numpy as np
    import polars as pl
    from bert_score import score as bertscore_score
    from rouge_score import rouge_scorer
    from transformers import AutoTokenizer

    from run_fn.test_fn import llm_test_fn
    from tpe.tpe_tokenizer_fast import TPETokenizerFast
    from utils.misc import set_random_seed, get_log_fname, parse_llm_output

    set_random_seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_log_kwargs = {
        "task": args.task,
        "data_format": args.data_format,
        "max_input_len": args.max_input_len,
        "llm_name": args.llm_id,
        "pe_method": args.pe_method,
        "n_response": args.n_response if args.n_response > 1 else None,
        "postfix": args.log_fname_postfix,
    }
    if args.cross_seed_std:
        seeds = args.seed_list if args.seed_list is not None else [args.seed]
        log_fname = get_log_fname(seed=None, **base_log_kwargs)
        output_fpath = log_dir / f"{log_fname}_seed-agg.json"
    else:
        log_fname = get_log_fname(**base_log_kwargs)
        output_fpath = log_dir / f"{log_fname}.json"
    print("will save to ", output_fpath)
    if output_fpath.exists():
        print(f"Results already exist at: {output_fpath}")
        exit()
    if args.cross_seed_std:
        seed_metrics = []
        inference_times = []
        seeds = args.seed_list if args.seed_list is not None else [args.seed]
        for seed in seeds:
            args.seed = seed
            seed_log_fname = get_log_fname(seed=seed, **base_log_kwargs)
            response_file = Path(args.response_dir) / f"{seed_log_fname}.parquet"
            running_info_file = Path(args.response_dir) / f"running_info_{seed_log_fname}.json"
            if not response_file.exists():
                print(f"Results do not exist at: {response_file}")
                exit()
            test_metrics_seed = _evaluate_task_metrics(str(response_file), args, n_bootstrap=0)
            test_metrics_seed = _attach_common_metrics(test_metrics_seed, str(response_file), args, n_bootstrap=0)
            seed_metrics.append((seed, test_metrics_seed))
            if running_info_file.exists():
                with open(running_info_file, "r") as f:
                    running_info = json.load(f)
                inference_times.append(float(running_info["inference_time"] / 60))
        test_metrics = _aggregate_cross_seed_metrics(seed_metrics)
        test_metrics["metadata"]["seeds"] = seeds
        test_metrics["metadata"]["n_seeds"] = len(seeds)
        if len(inference_times) > 0:
            test_metrics["cross_seed_metrics"]["inference_time_min"] = {
                "mean": float(np.mean(inference_times)),
                "std": float(np.std(inference_times)),
            }
    else:
        response_file = Path(args.response_dir) / f"{log_fname}.parquet"
        running_info_file = Path(args.response_dir) / f"running_info_{log_fname}.json"
        if not response_file.exists():
            print(f"Results do not exist at: {response_file}")
            exit()
        test_metrics = _evaluate_task_metrics(str(response_file), args, n_bootstrap=args.n_bootstrap)
        test_metrics = _attach_common_metrics(test_metrics, str(response_file), args, n_bootstrap=args.n_bootstrap)
    test_metrics["metadata"]["task"] = args.task
    test_metrics["metadata"]["data_format"] = args.data_format
    test_metrics["metadata"]["pe_method"] = args.pe_method
    test_metrics["metadata"]["max_input_len"] = args.max_input_len
    test_metrics["metadata"]["llm_id"] = args.llm_id
    test_metrics["metadata"]["seed"] = None if args.cross_seed_std else args.seed
    test_metrics["results"] = _format_results(test_metrics, args, use_cross_seed_std=args.cross_seed_std)
    if args.cross_seed_std:
        infer_time_stats = test_metrics.get("cross_seed_metrics", {}).get("inference_time_min", None)
        if infer_time_stats is not None:
            test_metrics["results"]["inference_time"] = f"{infer_time_stats['mean']:.3f} ({infer_time_stats['std']:.3f}) min"
    elif running_info_file.exists():
        with open(running_info_file, "r") as f:
            running_info = json.load(f)
        infer_time_min = running_info["inference_time"] / 60
        test_metrics["results"]["inference_time"] = f"{infer_time_min:.3f} min"
    with open(output_fpath, "w") as f:
        json.dump(test_metrics, f, indent=4)
