#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_BACKBONE = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "patent_backbone.csv.gz"
OUT_DIR = REPO_ROOT / "datasets" / "patent_evidence" / "output"
VAL_DIR = OUT_DIR / "validation"
DIAG_DIR = OUT_DIR / "diagnostics"
MODEL_DIR = OUT_DIR / "models"


TEXT_PATTERNS = {
    "data_collection": r"\b(data collection|collect(?:ing|ed)? data|data acquisition|dataset(?:s)?|observ(?:ed|ational) data|sample(?:d)? data|measurement data|telemetry data|sensor data|field data)\b",
    "empirical_analysis": r"\b(empirical|statistical (?:analysis|inference)|regression|hypothesis test|confidence interval|causal inference|estimate(?:d|s|ion)?|predictive model|infer(?:ence|red)|likelihood|bayesian|time[- ]series analysis)\b",
    "experimental_validation": r"\b(experiment(?:al)?|randomized|controlled trial|a/b test|testbed|validation (?:study|result|experiment)?|benchmark(?:ing|ed)?|ground truth|holdout set|cross[- ]validation|ablation)\b",
    "measurement_instrumentation": r"\b(sensor(?:s)?|instrumentation|detector|probe|meter|spectrometer|chromatograph|assay|biosensor|measurement system|measurement device|diagnostic device|imaging system)\b",
    "data_quality_calibration": r"\b(data quality|data cleaning|denois(?:e|ing)|outlier(?: detection)?|missing data|imput(?:e|ation)|calibrat(?:e|ion|ed)|verification|quality control|error correction|signal[- ]to[- ]noise|normalization)\b",
    "ml_training_data": r"\b(training data|training set|labeled data|supervised learning|unsupervised learning|machine learning|deep learning|neural network|model training|feature extraction|inference engine|classification model)\b",
    "boilerplate_digital": r"\b(receiv(?:e|ing) data|transmi(?:t|ssion) data|store(?:d|ing)? data|user interface|computer[- ]readable medium|processor configured to|network node|database server|memory device|execute instructions)\b",
}

SEMANTIC_POSITIVE_PROTOTYPES = [
    "invention uses experimental evidence and validation benchmarks from observed data",
    "method relies on sensor measurements, calibration, and data quality verification",
    "algorithm trained on labeled training data and evaluated using cross validation",
    "empirical estimation and statistical inference based on observational data",
]
SEMANTIC_NEGATIVE_PROTOTYPES = [
    "generic computer implemented method for transmitting and storing data",
    "digital interface and network communication protocol without empirical testing",
    "mechanical structure and hardware arrangement with no data analysis",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_text(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("patent_title", pd.Series("", index=frame.index)).fillna("")
    abst = frame.get("patent_abstract", pd.Series("", index=frame.index)).fillna("")
    return (title.astype(str) + " " + abst.astype(str)).str.lower()


def _build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    text = _safe_text(frame)
    n_tokens = text.str.count(r"\b\w+\b").fillna(0)

    out = pd.DataFrame(index=frame.index)
    out["patent_id"] = frame["patent_id"].astype(str)
    out["n_tokens"] = n_tokens

    for dim, pattern in TEXT_PATTERNS.items():
        out[f"{dim}_hits"] = text.str.count(pattern).fillna(0).astype(float)

    # Saturating component scores in [0, 1]
    out["data_collection_score"] = 1 - np.exp(-out["data_collection_hits"] / 2.0)
    out["empirical_analysis_score"] = 1 - np.exp(-out["empirical_analysis_hits"] / 2.0)
    out["experimental_validation_score"] = 1 - np.exp(-out["experimental_validation_hits"] / 2.0)
    out["measurement_instrumentation_score"] = 1 - np.exp(-out["measurement_instrumentation_hits"] / 2.0)
    out["data_quality_calibration_score"] = 1 - np.exp(-out["data_quality_calibration_hits"] / 2.0)
    out["ml_training_data_score"] = 1 - np.exp(-out["ml_training_data_hits"] / 2.0)
    out["boilerplate_score"] = 1 - np.exp(-out["boilerplate_digital_hits"] / 2.0)

    weighted = (
        1.15 * out["data_collection_score"]
        + 1.25 * out["empirical_analysis_score"]
        + 1.35 * out["experimental_validation_score"]
        + 1.05 * out["measurement_instrumentation_score"]
        + 1.25 * out["data_quality_calibration_score"]
        + 1.20 * out["ml_training_data_score"]
    )
    out["dictionary_score"] = weighted / (weighted + 2.5 + 1.4 * out["boilerplate_score"])

    # Metadata score from CPC features
    for c in [
        "has_measurement_cpc",
        "has_testing_cpc",
        "has_ml_cpc",
        "has_data_processing_cpc",
        "has_bio_assay_cpc",
        "has_statistics_cpc",
        "cpc_total_count",
    ]:
        if c not in frame.columns:
            frame[c] = 0
    has_measurement = pd.to_numeric(frame["has_measurement_cpc"], errors="coerce").fillna(0)
    has_testing = pd.to_numeric(frame["has_testing_cpc"], errors="coerce").fillna(0)
    has_ml = pd.to_numeric(frame["has_ml_cpc"], errors="coerce").fillna(0)
    has_data_proc = pd.to_numeric(frame["has_data_processing_cpc"], errors="coerce").fillna(0)
    has_bio = pd.to_numeric(frame["has_bio_assay_cpc"], errors="coerce").fillna(0)
    has_stats = pd.to_numeric(frame["has_statistics_cpc"], errors="coerce").fillna(0)
    cpc_total = pd.to_numeric(frame["cpc_total_count"], errors="coerce").fillna(0)

    metadata_signal = has_measurement + has_testing + has_ml + has_bio + has_stats
    out["metadata_signal_count"] = metadata_signal
    metadata_raw = (
        1.2 * has_measurement
        + 1.0 * has_testing
        + 1.15 * has_ml
        + 0.9 * has_data_proc
        + 1.1 * has_bio
        + 1.0 * has_stats
        + 0.12 * np.log1p(cpc_total)
    )
    out["metadata_score"] = _sigmoid((metadata_raw - 1.2).to_numpy())

    # Seed labels for weak supervision
    strong_evidence_hits = (
        out["experimental_validation_hits"]
        + out["ml_training_data_hits"]
        + out["data_quality_calibration_hits"]
        + out["measurement_instrumentation_hits"]
    )
    evidence_hits = (
        out["data_collection_hits"]
        + out["empirical_analysis_hits"]
        + out["experimental_validation_hits"]
        + out["measurement_instrumentation_hits"]
        + out["data_quality_calibration_hits"]
        + out["ml_training_data_hits"]
    )

    positive_seed = (
        (strong_evidence_hits >= 2)
        & (out["dictionary_score"] >= 0.30)
        & (out["boilerplate_score"] <= 0.65)
    ) | (
        (evidence_hits >= 2)
        & (out["dictionary_score"] >= 0.28)
        & (metadata_signal >= 1)
    ) | (
        (out["ml_training_data_hits"] >= 1)
        & (out["experimental_validation_hits"] >= 1)
    ) | (
        (out["data_quality_calibration_hits"] >= 1)
        & (out["empirical_analysis_hits"] >= 1)
    )
    negative_seed = (
        (evidence_hits == 0)
        & (metadata_signal == 0)
        & (out["dictionary_score"] <= 0.06)
        & ((out["boilerplate_score"] >= 0.20) | (has_data_proc >= 1))
    )
    negative_seed = negative_seed | (
        (evidence_hits == 0)
        & (metadata_signal == 0)
        & (out["dictionary_score"] <= 0.03)
        & (out["boilerplate_score"] >= 0.45)
    )
    negative_seed = negative_seed & (~positive_seed)
    seed = np.where(positive_seed, 1, np.where(negative_seed, 0, np.nan))
    out["seed_label"] = seed
    return out


def _threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, float(best_f1)


def _build_train_holdout_samples(chunksize: int, max_per_class_train: int, max_per_class_holdout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    train_counts = {0: 0, 1: 0}
    holdout_counts = {0: 0, 1: 0}

    for chunk_idx, chunk in enumerate(
        pd.read_csv(INPUT_BACKBONE, chunksize=chunksize, dtype={"patent_id": "string"}, low_memory=False),
        start=1,
    ):
        features = _build_feature_frame(chunk)
        chunk["text_for_model"] = _safe_text(chunk)
        joined = pd.concat([chunk, features.drop(columns=["patent_id"])], axis=1)
        seeded = joined[joined["seed_label"].notna()].copy()
        if seeded.empty:
            continue
        seeded["seed_label"] = seeded["seed_label"].astype(int)
        # deterministic split by patent id hash
        split_val = pd.util.hash_pandas_object(seeded["patent_id"], index=False).astype("uint64") % 10
        seeded["is_holdout"] = split_val >= 8

        for label in [0, 1]:
            # train sample
            train_pool = seeded[(seeded["seed_label"] == label) & (~seeded["is_holdout"])]
            remain_train = max_per_class_train - train_counts[label]
            if remain_train > 0 and not train_pool.empty:
                take = train_pool.sample(n=min(remain_train, len(train_pool)), random_state=42 + label)
                train_parts.append(take)
                train_counts[label] += len(take)

            # holdout sample
            hold_pool = seeded[(seeded["seed_label"] == label) & (seeded["is_holdout"])]
            remain_hold = max_per_class_holdout - holdout_counts[label]
            if remain_hold > 0 and not hold_pool.empty:
                take = hold_pool.sample(n=min(remain_hold, len(hold_pool)), random_state=84 + label)
                holdout_parts.append(take)
                holdout_counts[label] += len(take)

        if all(train_counts[l] >= max_per_class_train for l in [0, 1]) and all(
            holdout_counts[l] >= max_per_class_holdout for l in [0, 1]
        ):
            break
        if chunk_idx % 5 == 0:
            print(
                "Sampling weak labels "
                f"(chunk {chunk_idx}): train={train_counts} holdout={holdout_counts}",
                flush=True,
            )

    if not train_parts or not holdout_parts:
        raise RuntimeError("Could not build weak-supervision training/holdout samples.")

    train_df = pd.concat(train_parts, ignore_index=True)
    holdout_df = pd.concat(holdout_parts, ignore_index=True)

    def _balance(df: pd.DataFrame, seed: int) -> pd.DataFrame:
        counts = df["seed_label"].value_counts(dropna=True).to_dict()
        n0 = int(counts.get(0, 0))
        n1 = int(counts.get(1, 0))
        if n0 == 0 or n1 == 0:
            raise RuntimeError(f"Weak-supervision sample has single class only: {counts}")
        n = min(n0, n1)
        part0 = df[df["seed_label"] == 0].sample(n=n, random_state=seed)
        part1 = df[df["seed_label"] == 1].sample(n=n, random_state=seed + 1)
        return pd.concat([part0, part1], ignore_index=True)

    train_df = _balance(train_df, seed=101)
    holdout_df = _balance(holdout_df, seed=202)
    return train_df, holdout_df


def _fit_models(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[Dict[str, object], Dict[str, float], pd.DataFrame]:
    vectorizer = TfidfVectorizer(max_features=80_000, ngram_range=(1, 2), min_df=5, max_df=0.97)
    X_train_text = vectorizer.fit_transform(train_df["text_for_model"])
    X_hold_text = vectorizer.transform(holdout_df["text_for_model"])
    y_train = train_df["seed_label"].astype(int).to_numpy()
    y_hold = holdout_df["seed_label"].astype(int).to_numpy()

    text_clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=30, random_state=123)
    text_clf.fit(X_train_text, y_train)
    text_prob_train = text_clf.predict_proba(X_train_text)[:, 1]
    text_prob_hold = text_clf.predict_proba(X_hold_text)[:, 1]

    # Embedding-like semantic candidate via LSA
    svd = TruncatedSVD(n_components=128, random_state=123)
    X_train_emb = svd.fit_transform(X_train_text)
    X_hold_emb = svd.transform(X_hold_text)
    proto_pos = svd.transform(vectorizer.transform(SEMANTIC_POSITIVE_PROTOTYPES)).mean(axis=0)
    proto_neg = svd.transform(vectorizer.transform(SEMANTIC_NEGATIVE_PROTOTYPES)).mean(axis=0)
    proto_pos = proto_pos / (np.linalg.norm(proto_pos) + 1e-9)
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-9)
    hold_norm = X_hold_emb / (np.linalg.norm(X_hold_emb, axis=1, keepdims=True) + 1e-9)
    semantic_hold_raw = hold_norm @ proto_pos - hold_norm @ proto_neg
    semantic_hold_prob = _sigmoid(2.0 * semantic_hold_raw)

    stack_train = np.column_stack(
        [
            train_df["dictionary_score"].to_numpy(),
            train_df["metadata_score"].to_numpy(),
            train_df["boilerplate_score"].to_numpy(),
            text_prob_train,
        ]
    )
    stack_hold = np.column_stack(
        [
            holdout_df["dictionary_score"].to_numpy(),
            holdout_df["metadata_score"].to_numpy(),
            holdout_df["boilerplate_score"].to_numpy(),
            text_prob_hold,
        ]
    )
    hybrid = LogisticRegression(max_iter=1000, random_state=123)
    hybrid.fit(stack_train, y_train)
    hybrid_hold_prob = hybrid.predict_proba(stack_hold)[:, 1]

    # Candidate method comparison on holdout labels
    method_probs = {
        "dictionary": holdout_df["dictionary_score"].to_numpy(),
        "metadata": holdout_df["metadata_score"].to_numpy(),
        "semantic_lsa": semantic_hold_prob,
        "text_supervised": text_prob_hold,
        "hybrid_benchmark": hybrid_hold_prob,
    }
    metrics_rows = []
    for name, prob in method_probs.items():
        auc = float(roc_auc_score(y_hold, prob))
        t, best_f1 = _threshold_by_f1(y_hold, prob)
        metrics_rows.append({"method": name, "auc": auc, "best_f1": best_f1, "best_threshold": t})
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["auc", "best_f1"], ascending=False)
    benchmark_threshold = float(
        metrics_df.loc[metrics_df["method"] == "hybrid_benchmark", "best_threshold"].iloc[0]
    )

    artifacts = {
        "vectorizer": vectorizer,
        "text_clf": text_clf,
        "hybrid_clf": hybrid,
        "svd": svd,
        "semantic_pos": proto_pos,
        "semantic_neg": proto_neg,
    }
    thresholds = {
        "benchmark_threshold": benchmark_threshold,
        "dictionary_threshold": float(metrics_df.loc[metrics_df["method"] == "dictionary", "best_threshold"].iloc[0]),
        "metadata_threshold": float(metrics_df.loc[metrics_df["method"] == "metadata", "best_threshold"].iloc[0]),
    }
    return artifacts, thresholds, metrics_df


def _load_saved_artifacts() -> tuple[Dict[str, object], Dict[str, float]]:
    with open(MODEL_DIR / "text_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_DIR / "text_supervised_clf.pkl", "rb") as f:
        text_clf = pickle.load(f)
    with open(MODEL_DIR / "hybrid_benchmark_clf.pkl", "rb") as f:
        hybrid_clf = pickle.load(f)
    with open(MODEL_DIR / "semantic_svd.pkl", "rb") as f:
        semantic = pickle.load(f)
    thresholds = json.loads((VAL_DIR / "candidate_method_thresholds.json").read_text())
    artifacts = {
        "vectorizer": vectorizer,
        "text_clf": text_clf,
        "hybrid_clf": hybrid_clf,
        "svd": semantic["svd"],
        "semantic_pos": semantic["pos"],
        "semantic_neg": semantic["neg"],
    }
    return artifacts, thresholds


def _score_chunks(
    *,
    artifacts: Dict[str, object],
    thresholds: Dict[str, float],
    chunksize: int,
    chunk_start: int,
    chunk_end: int | None,
    append_output: bool,
) -> dict[str, int]:
    output_path = OUT_DIR / "patent_evidence_patent_level.csv.gz"
    if output_path.exists() and not append_output:
        output_path.unlink()

    vec = artifacts["vectorizer"]
    clf = artifacts["text_clf"]
    hybrid = artifacts["hybrid_clf"]
    svd = artifacts["svd"]

    first = not (append_output and output_path.exists())
    rows_written = 0
    chunks_processed = 0
    last_chunk = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(INPUT_BACKBONE, chunksize=chunksize, dtype={"patent_id": "string"}, low_memory=False),
        start=1,
    ):
        if chunk_idx < chunk_start:
            continue
        if chunk_end is not None and chunk_idx > chunk_end:
            break

        print(f"Scoring chunk {chunk_idx} ...", flush=True)
        features = _build_feature_frame(chunk)
        text = _safe_text(chunk)
        X_text = vec.transform(text)
        text_prob = clf.predict_proba(X_text)[:, 1]
        emb = svd.transform(X_text)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sem_prob = _sigmoid(2.0 * ((emb @ artifacts["semantic_pos"]) - (emb @ artifacts["semantic_neg"])))
        stack = np.column_stack(
            [
                features["dictionary_score"].to_numpy(),
                features["metadata_score"].to_numpy(),
                features["boilerplate_score"].to_numpy(),
                text_prob,
            ]
        )
        benchmark = hybrid.predict_proba(stack)[:, 1]
        conf = np.abs(benchmark - 0.5) * 2
        benchmark_flag = (benchmark >= thresholds["benchmark_threshold"]).astype(int)

        out = pd.DataFrame(
            {
                "patent_id": chunk["patent_id"].astype(str),
                "app_year": pd.to_numeric(chunk.get("app_year"), errors="coerce").astype("Int64"),
                "grant_year": pd.to_numeric(chunk.get("grant_year"), errors="coerce").astype("Int64"),
                "patent_date_year": pd.to_numeric(chunk.get("patent_date_year"), errors="coerce").astype("Int64"),
                "assignee_country_mode": chunk.get("assignee_country_mode"),
                "assignee_country_primary": chunk.get("assignee_country_primary"),
                "inventor_country_mode": chunk.get("inventor_country_mode"),
                "inventor_country_primary": chunk.get("inventor_country_primary"),
                "patent_title": chunk.get("patent_title"),
                "patent_abstract": chunk.get("patent_abstract"),
                "num_claims": pd.to_numeric(chunk.get("num_claims"), errors="coerce"),
                "data_collection_score": features["data_collection_score"],
                "empirical_analysis_score": features["empirical_analysis_score"],
                "experimental_validation_score": features["experimental_validation_score"],
                "measurement_instrumentation_score": features["measurement_instrumentation_score"],
                "data_quality_calibration_score": features["data_quality_calibration_score"],
                "ml_training_data_score": features["ml_training_data_score"],
                "boilerplate_score": features["boilerplate_score"],
                "dictionary_score": features["dictionary_score"],
                "metadata_score": features["metadata_score"],
                "metadata_signal_count": features["metadata_signal_count"],
                "semantic_lsa_score": sem_prob,
                "text_supervised_score": text_prob,
                "benchmark_score": benchmark,
                "benchmark_flag": benchmark_flag,
                "benchmark_confidence": conf,
                "seed_label": features["seed_label"],
                "has_measurement_cpc": pd.to_numeric(chunk.get("has_measurement_cpc"), errors="coerce"),
                "has_testing_cpc": pd.to_numeric(chunk.get("has_testing_cpc"), errors="coerce"),
                "has_ml_cpc": pd.to_numeric(chunk.get("has_ml_cpc"), errors="coerce"),
                "has_data_processing_cpc": pd.to_numeric(chunk.get("has_data_processing_cpc"), errors="coerce"),
                "has_bio_assay_cpc": pd.to_numeric(chunk.get("has_bio_assay_cpc"), errors="coerce"),
                "has_statistics_cpc": pd.to_numeric(chunk.get("has_statistics_cpc"), errors="coerce"),
                "cpc_total_count": pd.to_numeric(chunk.get("cpc_total_count"), errors="coerce"),
                "cpc_unique_subclass_count": pd.to_numeric(chunk.get("cpc_unique_subclass_count"), errors="coerce"),
            }
        )
        out.to_csv(output_path, mode="a", index=False, header=first, compression="gzip")
        first = False
        rows_written += len(out)
        chunks_processed += 1
        last_chunk = chunk_idx

        del features, text, X_text, emb, stack, out
        gc.collect()

    return {"rows_written": rows_written, "chunks_processed": chunks_processed, "last_chunk": last_chunk}


def _write_output_diagnostics(output_path: Path, threshold: float) -> int:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW scored AS
        SELECT
            patent_id,
            TRY_CAST(app_year AS INTEGER) AS app_year,
            TRY_CAST(grant_year AS INTEGER) AS grant_year,
            assignee_country_mode,
            inventor_country_mode,
            patent_title,
            TRY_CAST(benchmark_score AS DOUBLE) AS benchmark_score,
            TRY_CAST(dictionary_score AS DOUBLE) AS dictionary_score,
            TRY_CAST(metadata_score AS DOUBLE) AS metadata_score,
            TRY_CAST(semantic_lsa_score AS DOUBLE) AS semantic_lsa_score,
            TRY_CAST(text_supervised_score AS DOUBLE) AS text_supervised_score,
            TRY_CAST(benchmark_confidence AS DOUBLE) AS benchmark_confidence
        FROM read_csv_auto('{output_path.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        """
    )
    total_rows = int(con.execute("SELECT COUNT(*) FROM scored").fetchone()[0])

    top_df = con.execute(
        """
        SELECT
            patent_id,
            app_year,
            grant_year,
            assignee_country_mode,
            inventor_country_mode,
            benchmark_score,
            dictionary_score,
            metadata_score,
            semantic_lsa_score,
            text_supervised_score,
            benchmark_confidence,
            patent_title
        FROM scored
        ORDER BY benchmark_score DESC
        LIMIT 500
        """
    ).fetchdf()
    bottom_df = con.execute(
        """
        SELECT
            patent_id,
            app_year,
            grant_year,
            assignee_country_mode,
            inventor_country_mode,
            benchmark_score,
            dictionary_score,
            metadata_score,
            semantic_lsa_score,
            text_supervised_score,
            benchmark_confidence,
            patent_title
        FROM scored
        ORDER BY benchmark_score ASC
        LIMIT 500
        """
    ).fetchdf()
    amb_df = con.execute(
        f"""
        SELECT
            patent_id,
            app_year,
            grant_year,
            assignee_country_mode,
            inventor_country_mode,
            benchmark_score,
            dictionary_score,
            metadata_score,
            semantic_lsa_score,
            text_supervised_score,
            benchmark_confidence,
            patent_title,
            ABS(benchmark_score - {threshold}) AS dist_to_threshold
        FROM scored
        ORDER BY dist_to_threshold ASC
        LIMIT 500
        """
    ).fetchdf()

    top_df.to_csv(DIAG_DIR / "top_patents_by_benchmark_score.csv", index=False)
    bottom_df.to_csv(DIAG_DIR / "bottom_patents_by_benchmark_score.csv", index=False)
    amb_df.to_csv(DIAG_DIR / "most_ambiguous_patents.csv", index=False)
    return total_rows


def run_scoring(
    chunksize: int = 40_000,
    train_per_class: int = 120_000,
    holdout_per_class: int = 30_000,
    score_only: bool = False,
    chunk_start: int = 1,
    chunk_end: int | None = None,
    append_output: bool = False,
    skip_diagnostics: bool = False,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_sample_rows: int | None = None
    holdout_sample_rows: int | None = None

    if score_only:
        artifacts, thresholds = _load_saved_artifacts()
        print("Loaded saved model artifacts and thresholds.", flush=True)
    else:
        train_df, holdout_df = _build_train_holdout_samples(
            chunksize=chunksize,
            max_per_class_train=train_per_class,
            max_per_class_holdout=holdout_per_class,
        )
        train_sample_rows = int(len(train_df))
        holdout_sample_rows = int(len(holdout_df))
        print(
            f"Weak supervision samples ready: train={len(train_df):,} holdout={len(holdout_df):,}",
            flush=True,
        )
        artifacts, thresholds, metrics_df = _fit_models(train_df, holdout_df)
        print("Model fitting complete.", flush=True)

        # Save model artifacts
        with open(MODEL_DIR / "text_vectorizer.pkl", "wb") as f:
            pickle.dump(artifacts["vectorizer"], f)
        with open(MODEL_DIR / "text_supervised_clf.pkl", "wb") as f:
            pickle.dump(artifacts["text_clf"], f)
        with open(MODEL_DIR / "hybrid_benchmark_clf.pkl", "wb") as f:
            pickle.dump(artifacts["hybrid_clf"], f)
        with open(MODEL_DIR / "semantic_svd.pkl", "wb") as f:
            pickle.dump(
                {"svd": artifacts["svd"], "pos": artifacts["semantic_pos"], "neg": artifacts["semantic_neg"]},
                f,
            )

        metrics_df.to_csv(VAL_DIR / "candidate_method_comparison.csv", index=False)
        (VAL_DIR / "candidate_method_thresholds.json").write_text(json.dumps(thresholds, indent=2))

        # Build manual validation sample (stratified by score and disagreement)
        hold_sample = holdout_df.copy()
        vec = artifacts["vectorizer"]
        clf = artifacts["text_clf"]
        hybrid = artifacts["hybrid_clf"]
        svd = artifacts["svd"]
        X_hold = vec.transform(hold_sample["text_for_model"])
        hold_sample["text_supervised_score"] = clf.predict_proba(X_hold)[:, 1]
        emb = svd.transform(X_hold)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sem = emb @ artifacts["semantic_pos"] - emb @ artifacts["semantic_neg"]
        hold_sample["semantic_lsa_score"] = _sigmoid(2.0 * sem)
        stack = np.column_stack(
            [
                hold_sample["dictionary_score"].to_numpy(),
                hold_sample["metadata_score"].to_numpy(),
                hold_sample["boilerplate_score"].to_numpy(),
                hold_sample["text_supervised_score"].to_numpy(),
            ]
        )
        hold_sample["benchmark_score"] = hybrid.predict_proba(stack)[:, 1]
        hold_sample["method_disagreement"] = (
            hold_sample[
                [
                    "dictionary_score",
                    "metadata_score",
                    "semantic_lsa_score",
                    "text_supervised_score",
                ]
            ].std(axis=1)
        )
        hold_sample["manual_label"] = pd.NA
        hold_sample["manual_notes"] = ""
        sample_parts = [
            hold_sample.nlargest(300, "benchmark_score"),
            hold_sample.nsmallest(300, "benchmark_score"),
            hold_sample.assign(abs_mid=(hold_sample["benchmark_score"] - thresholds["benchmark_threshold"]).abs())
            .nsmallest(300, "abs_mid")
            .drop(columns=["abs_mid"]),
            hold_sample.nlargest(300, "method_disagreement"),
        ]
        manual_sample = pd.concat(sample_parts, ignore_index=True).drop_duplicates(subset=["patent_id"]).head(1200)
        keep_cols = [
            "patent_id",
            "patent_title",
            "patent_abstract",
            "seed_label",
            "dictionary_score",
            "metadata_score",
            "semantic_lsa_score",
            "text_supervised_score",
            "benchmark_score",
            "method_disagreement",
            "manual_label",
            "manual_notes",
        ]
        manual_sample[keep_cols].to_csv(VAL_DIR / "manual_validation_sample.csv", index=False)

    chunk_stats = _score_chunks(
        artifacts=artifacts,
        thresholds=thresholds,
        chunksize=chunksize,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        append_output=append_output,
    )

    output_path = OUT_DIR / "patent_evidence_patent_level.csv.gz"
    total_rows = None
    if not skip_diagnostics and output_path.exists():
        total_rows = _write_output_diagnostics(output_path, thresholds["benchmark_threshold"])

    summary = {
        "train_sample_rows": train_sample_rows,
        "holdout_sample_rows": holdout_sample_rows,
        "benchmark_threshold": thresholds["benchmark_threshold"],
        "output_file": str(output_path),
        "score_only": score_only,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "append_output": append_output,
        "chunk_rows_written": chunk_stats["rows_written"],
        "chunk_count_processed": chunk_stats["chunks_processed"],
        "last_chunk_processed": chunk_stats["last_chunk"],
        "total_output_rows": total_rows,
    }
    (OUT_DIR / "patent_evidence_scoring_summary.json").write_text(json.dumps(summary, indent=2))

    print("Scoring complete.")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build patent-level data/evidence-intensity scores.")
    p.add_argument("--chunksize", type=int, default=40_000)
    p.add_argument("--train-per-class", type=int, default=120_000)
    p.add_argument("--holdout-per-class", type=int, default=30_000)
    p.add_argument("--score-only", action="store_true", help="Skip fitting and use saved artifacts to score patents.")
    p.add_argument("--chunk-start", type=int, default=1, help="1-based chunk index to start scoring from.")
    p.add_argument("--chunk-end", type=int, default=None, help="Optional 1-based chunk index to stop scoring at.")
    p.add_argument("--append-output", action="store_true", help="Append scored rows to existing output file.")
    p.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip diagnostics generation (useful for intermediate chunked scoring runs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_scoring(
        chunksize=args.chunksize,
        train_per_class=args.train_per_class,
        holdout_per_class=args.holdout_per_class,
        score_only=args.score_only,
        chunk_start=args.chunk_start,
        chunk_end=args.chunk_end,
        append_output=args.append_output,
        skip_diagnostics=args.skip_diagnostics,
    )


if __name__ == "__main__":
    main()
