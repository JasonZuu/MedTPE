import json, re
from typing import Tuple, Dict, List
import polars as pl
import numpy as np
from tqdm import tqdm
from datetime import datetime


def _time_key(t: str) -> int:
    """
    → Convert to integer seconds for time sorting
    Supports two formats:
      1. 'YYYY-MM-DD HH:MM:SS' (or ISO‑8601 variants)
      2. 'Day0 13Hour 45Minute'
    """
    # ① ISO / common timestamp formats
    try:
        return int(datetime.fromisoformat(t).timestamp())
    except ValueError:
        pass

    # ② Day...Hour...Minute format
    m = re.match(r"Day(\d+)\s+(\d{2})Hour\s+(\d{2})Minute", t)
    if m:
        day, hh, mm = map(int, m.groups())
        return day * 24 * 3600 + hh * 3600 + mm * 60

    raise ValueError(f"Unsupported time string: {t}")


def calc_time_token_stats(
    df: pl.DataFrame,
    llm_tokenizer,
    *,
    data_col: str = "data",
    question_col: str = "question",
    instruction: str = "",
    data_prefix: str = "## Data\n",
) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """
    Returns:
      • time_stats_df : cumulative token distribution stats at each timestamp
      • final_stats   : token distribution statistics at the final timestamp (dict)
    """

    # ---------- 0) Parse the data column ----------
    data_dicts = [
        json.loads(x) if isinstance(x, str) else x
        for x in df[data_col].to_list()
    ]
    n_rows = len(df)

    # ---------- 1) Global timeline ----------
    all_times: set[str] = set()
    for d in data_dicts:
        all_times.update(t for t in d.keys() if t != "Static")
    times_sorted = sorted(all_times, key=_time_key)       # Sort by time while keeping original strings
    n_times = len(times_sorted)
    time2idx = {t: i for i, t in enumerate(times_sorted)}

    # ---------- 2) Token length cache ----------
    tok_cache: Dict[str, int] = {}

    def tok_len(s: str) -> int:
        if s not in tok_cache:
            tok_cache[s] = len(llm_tokenizer.tokenize(s))
        return tok_cache[s]

    # ---------- 3) Build cumulative matrix ----------
    accum_mat = np.zeros((n_rows, n_times), dtype=np.int32)

    pbar = tqdm(total=n_rows, desc="token-accum fast")
    for row_i, d in enumerate(data_dicts):
        static_sum = sum(tok_len(ev) for ev in (d.get("Static") or []))

        row_vec = np.zeros(n_times, dtype=np.int32)
        for t, events in d.items():
            if t == "Static" or not events:
                continue
            row_vec[time2idx[t]] = sum(tok_len(ev) for ev in events)

        accum_mat[row_i] = static_sum + np.cumsum(row_vec, dtype=np.int32)
        pbar.update(1)
    pbar.close()

    # ---------- 4) Statistic helper ----------
    def _stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "median": float(np.median(arr)),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "Q1":     float(np.percentile(arr, 25)),
            "Q3":     float(np.percentile(arr, 75)),
        }

    # ---------- 5) time_stats_df (optional) ----------
    # Uncomment below to return stats for each timestamp:
    # time_stats_df = pl.DataFrame({
    #     "timestamp": times_sorted,
    #     "mean":      np.mean(accum_mat, axis=0),
    #     "std":       np.std(accum_mat, axis=0, ddof=0),
    #     "median":    np.median(accum_mat, axis=0),
    #     "min":       np.min(accum_mat, axis=0),
    #     "max":       np.max(accum_mat, axis=0),
    #     "Q1":        np.percentile(accum_mat, 25, axis=0),
    #     "Q3":        np.percentile(accum_mat, 75, axis=0)
    # })

    # ---------- 6) Final cumulative token stats ----------
    final_stats = _stats(accum_mat[:, -1])

    return final_stats
