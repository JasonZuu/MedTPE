import json
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score
from torch.nn import functional as F
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Any
from joblib import Parallel, delayed
from collections import Counter


from utils.misc import parse_llm_output


@torch.no_grad()
def regular_test_fn(model, test_loader, device, avg_method="macro",
                    is_multi_label=False, use_bootstrap=False, n_bootstrap=1000):
    """
    Tests the model on the validation dataset and returns the F1.

    :param model: Model to test.
    :param test_loader: DataLoader for the validation data.
    :return: Accuracy of the model on the validation data.
    """
    model.eval()
    num_classes = model.num_classes
    if not is_multi_label and num_classes == 2:
        avg_method = "binary"
    
    total_labels = []
    total_preds = []
    total_scores = []

    pbar = tqdm(total=len(test_loader), desc="Testing")
    for data in test_loader:
        demo, ts, label = data['demo'], data['ts'], data['label']
        demo, ts, label = demo.to(device), ts.to(device), label.to(device)

        y_scores = model(demo, ts)
        if is_multi_label:
            y_pred = (y_scores > 0.5).to(torch.int32)
        else:
            y_pred = torch.argmax(y_scores, dim=-1)
            label = torch.argmax(label, dim=-1) # convert to idx

        total_labels.extend(label.cpu().numpy())
        total_preds.extend(y_pred.cpu().numpy())
        total_scores.extend(y_scores.cpu().numpy())
        pbar.update(1)
    pbar.close()

    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)
    total_scores = np.array(total_scores)

    # bootstrap metrics
    if use_bootstrap:
        bootstrap_metrics = bootstrap_metrics_fn(
            total_labels, total_preds, n_bootstrap, avg_method
        )
        metadata = {
            "total_samples": len(total_labels),
            "num_classes": num_classes,
            "avg_method": avg_method,
            "n_bootstrap": n_bootstrap,
        }   
        return bootstrap_metrics, metadata
    else: # point metrics
        f1 = f1_score(total_labels, total_preds, average=avg_method)
        precision = precision_score(total_labels, total_preds, average=avg_method)
        recall = recall_score(total_labels, total_preds, average=avg_method)
        accuracy = accuracy_score(total_labels, total_preds)
        jaccard = jaccard_score(total_labels, total_preds, average=avg_method)
        loss = F.cross_entropy(torch.tensor(total_scores), torch.tensor(total_labels)).item()

        test_result = {"f1": f1, "precision": precision, "recall": recall, 
                       "accuracy": accuracy, "jaccard": jaccard,
                       "loss": loss}
    return test_result


def bootstrap_metrics_fn(
    y_true: List, 
    y_pred: List, 
    n_bootstrap: int,
    avg_method: str,
    random_seed: Optional[int] = None,
    n_jobs: int = -1
) -> Dict[str, Dict[str, float]]:
    """
    Faster version with joblib parallelism.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    def compute_one_iter(_):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        true_resampled = [y_true[i] for i in idx]
        pred_resampled = [y_pred[i] for i in idx]
        prec = precision_score(true_resampled, pred_resampled, average=avg_method, zero_division=0)
        recall = recall_score(true_resampled, pred_resampled, average=avg_method, zero_division=0)
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        return {
            "precision": prec,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy_score(true_resampled, pred_resampled),
            "jaccard": jaccard_score(true_resampled, pred_resampled, average=avg_method, zero_division=0)
        }

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_iter)(_) for _ in tqdm(range(n_bootstrap), desc="Bootstrapping")
    )

    # Aggregate results
    metrics = {key: np.array([res[key] for res in results_list]) for key in results_list[0]}

    summary = {
        metric: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }
        for metric, values in metrics.items()
    }

    return summary


def labels_preds_to_onehot(label_list, pred_list, num_classes, label_to_idx):
    """
    Convert labels and predictions to multi-hot encoding, with special handling for None predictions.
    
    Args:
        label_list: List[List[str]] true labels (e.g., [["A"], ["A","B"], [], ["C"]])
        pred_list: List[Optional[str]] predictions (may contain None)
        num_classes: Total number of classes
        label_to_idx: Dictionary mapping labels to indices
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
        - true_labels_onehot: (n_samples, num_classes) multi-hot matrix
        - pred_labels_onehot: (n_samples, num_classes) multi-hot matrix 
                             (None predictions are opposite of true labels)
    """
    # Initialize matrices
    true_onehot = np.zeros((len(label_list), num_classes))
    pred_onehot = np.zeros((len(pred_list), num_classes))
    
    # Process true labels
    for i, sublist in enumerate(label_list):
        for letter in sublist:
            if letter in label_to_idx:
                true_onehot[i, label_to_idx[letter]] = 1
    
    # Process predictions with None handling
    for i, pred in enumerate(pred_list):
        if pred is not None:
            for letter in pred:
                if letter not in label_to_idx:
                    print(f"Warning: {letter} not in label_to_idx. Skipping.")
                    continue
                if letter in label_to_idx:
                    pred_onehot[i, label_to_idx[letter]] = 1
        else:
            # For None predictions, use exact opposite of true labels
            y_true = true_onehot[i]
            y_pred = 1 - y_true
            pred_onehot[i] = y_pred
    
    return true_onehot, pred_onehot


def llm_test_fn(
    result_path: str, 
    num_classes: int,
    avg_method: str = "macro", # "macro" or "micro"
    use_bootstrap: bool = True,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None,
    is_multi_label: bool = False,
    n_response: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Evaluate LLM predictions from a parquet file with bootstrap statistics.
    Handles invalid predictions by counting them as incorrect.
    Converts string labels (A,B,C...) to one-hot encoded format for evaluation.
    
    Args:
        result_path: Path to the parquet file containing results
        multi_class: Whether to evaluate in multi-class mode (False for binary)
        n_bootstrap: Number of bootstrap samples for statistics
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - bootstrap_metrics: Bootstrap statistics for each metric
        - point_metrics: Single-point estimates of metrics
        - metadata: Sample statistics
        Or None if all predictions are invalid
    """
    # Load and process data
    df = pl.read_parquet(result_path)
    
    y_trues = []
    y_preds = []
    invalid_count = 0
    
    pbar = tqdm(total=df.height, desc="Evaluating")
    for row in df.iter_rows(named=True):
        label = row["label"]
        # 收集多回答
        answers_in_row = []
        for idx in range(n_response):
            col_name = f"generated_text_{idx}"
            if col_name not in row or row[col_name] is None:
                continue
            is_valid, ans = parse_llm_output(row[col_name])
            if is_valid:
                # Drop None
                ans = [a for a in ans if a]
                # 统一格式：去重 & 排序保证可 hash
                if isinstance(ans, list):
                    ans = tuple(sorted(set(ans)))  # make hashable
                # map to A,B,C if ans is num (1->A, 2->B, etc)
                if all([c.isdigit() for c in ans]):
                    print(f"Warning: Pred {ans} is a number. Mapping to A,B,C...")
                    ans = tuple([chr(ord('A') + int(c) - 1) for c in ans])
                # map to capitable letter
                ans = tuple([letter.upper() for letter in ans])
                answers_in_row.append(ans)

        # 决定最终预测
        if not answers_in_row:
            invalid_count += 1
            y_preds.append(None)
        elif len(answers_in_row) == 1:
            y_preds.append(list(answers_in_row[0]) if isinstance(answers_in_row[0], tuple) else answers_in_row[0])
        else: # majority vote
            if is_multi_label:
                best_ans = vote_multi_label_with_topk(answers_in_row)
            else:
                best_ans = vote_single_label(answers_in_row)
            y_preds.append(list(best_ans) if isinstance(best_ans, tuple) else best_ans)

        # map label to A,B,C if it is a number
        if all([c.isdigit() for c in label]):
            print(f"Warning: Label {label} is a number. Mapping to A,B,C...")
            label = tuple([chr(ord('A') + int(c) - 1) for c in label])
        y_trues.append(label if isinstance(label, list) else [label])
        pbar.update(1)
    pbar.close()
    
    # Convert string labels to numerical indices (A=0, B=1, etc.)
    all_labels = [chr(i) for i in range(ord('A'), ord('A')+num_classes)]
    label_to_idx = {letter: idx for idx, letter in enumerate(all_labels)}
    
    # Convert true labels to one-hot
    y_true_onehot, y_pred_onehot = labels_preds_to_onehot(
        y_trues, y_preds, num_classes, label_to_idx
    )
    if y_true_onehot.shape[-1] == 2:  # binary classification
        # Binary classification: convert to 1D
        print("Binary classification detected. Converting to 1D.")
        y_true_onehot = np.argmax(y_true_onehot, axis=-1)
        y_pred_onehot = np.argmax(y_pred_onehot, axis=-1)
        if avg_method != "binary":
            print(f"Warning: avg_method changed to binary for binary classification.")
            avg_method = "binary"

    
    # Calculate bootstrap metrics
    if use_bootstrap:
        bootstrap_metrics = bootstrap_metrics_fn(
            y_true_onehot, y_pred_onehot, n_bootstrap, avg_method, random_seed
        )
    else:
        bootstrap_metrics = {}
    
    # Calculate point metrics
    prec = precision_score(y_true_onehot, y_pred_onehot, average=avg_method, zero_division=0)
    recall = recall_score(y_true_onehot, y_pred_onehot, average=avg_method, zero_division=0)
    f1 = 2*prec* recall / (prec + recall) if (prec + recall) > 0 else 0
    point_metrics = {
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy_score(y_true_onehot, y_pred_onehot),
        "jaccard": jaccard_score(y_true_onehot, y_pred_onehot, average=avg_method, zero_division=0),
    }
    
    return {
        f"bootstrap_metrics": bootstrap_metrics,
        f"point_metrics": point_metrics,
        "metadata": {
            "total_samples": df.height,
            "valid_samples": df.height - invalid_count,
            "invalid_count": invalid_count,
            "valid_ratio": (df.height - invalid_count) / df.height,
            "num_classes": num_classes,
            "avg_method": avg_method,
        }
    }



def vote_multi_label_with_topk(answers_in_row: List[tuple]) -> List[str]:
    """Label‑wise voting: 先统计标签票数，再取 *平均回答长度 k* 的票数最高标签。

    * answers_in_row 里每个元素是 canonical tuple of labels (去重 + 排序)。
    * k = round(平均 tuple 长度).  若 k < 1 则 k = 1，若 k > num_labels 则 k = num_labels。
    * 票数相同按出现先后次序决定。
    """
    # 统计标签出现次数
    label_counter: Counter = Counter(t for ans in answers_in_row for t in ans)
    if not label_counter:
        return []

    # 平均回答长度
    avg_len = sum(len(a) for a in answers_in_row) / len(answers_in_row)
    k = max(1, int(round(avg_len)))
    k = min(k, len(label_counter))  # 不超过可用标签数

    # 排序：先按票数降序，再按第一次出现顺序
    first_order: Dict[str, int] = {}
    for ans in answers_in_row:
        for lab in ans:
            if lab not in first_order:
                first_order[lab] = len(first_order)

    sorted_labels = sorted(
        label_counter.items(),
        key=lambda x: (-x[1], first_order.get(x[0], 1e9))
    )
    top_k = [lab for lab, _ in sorted_labels[:k]]
    return top_k


def vote_single_label(answers_in_row):
    cnt = Counter(answers_in_row)
    best_ans = max(cnt.items(), key=lambda x: (x[1], -answers_in_row.index(x[0])))[0]
    return best_ans
