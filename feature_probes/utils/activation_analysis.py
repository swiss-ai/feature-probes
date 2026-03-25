from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from feature_probes.data.dataset import TokenizedProbingDatasetConfig, create_probing_dataset
from feature_probes.utils.model_utils import load_model_and_tokenizer


@dataclass
class DatasetSpec:
    hf_repo: str
    split: str
    max_length: int
    max_samples: int
    seed: int


@dataclass
class CollectionSpec:
    layers: List[int]
    max_tokens_per_label: int
    pos_neg_ratio: float = 1.0
    enforce_ratio_on_finalize: bool = True
    finalize_sample_total: int | None = None


def _sample_with_ratio(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    pos_neg_ratio: float,
    seed: int,
    sample_total: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if pos_neg_ratio <= 0:
        raise ValueError("pos_neg_ratio must be > 0")

    if len(pos) == 0 or len(neg) == 0:
        hidden_dim = pos.shape[1] if len(pos) else (neg.shape[1] if len(neg) else 0)
        empty = np.empty((0, hidden_dim), dtype=pos.dtype if len(pos) else neg.dtype if len(neg) else np.float32)
        return empty, empty

    max_pos = len(pos)
    max_neg = len(neg)

    if sample_total is not None:
        if sample_total < 2:
            raise ValueError("sample_total must be >= 2 when provided")
        n_pos_target = max(1, int(round(sample_total * pos_neg_ratio / (1.0 + pos_neg_ratio))))
        n_neg_target = max(1, sample_total - n_pos_target)
    else:
        n_neg_target = min(max_neg, int(max_pos / pos_neg_ratio))
        n_pos_target = max(1, int(round(n_neg_target * pos_neg_ratio)))

    n_pos = min(max_pos, n_pos_target)
    n_neg = min(max_neg, n_neg_target)

    # Re-project once so that returned counts follow the requested ratio as closely as possible.
    n_neg = min(n_neg, max(1, int(n_pos / pos_neg_ratio)))
    n_pos = min(n_pos, max(1, int(round(n_neg * pos_neg_ratio))))

    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(max_pos, size=n_pos, replace=False)
    neg_idx = rng.choice(max_neg, size=n_neg, replace=False)
    return pos[pos_idx], neg[neg_idx]


def build_tokenized_dataset(tokenizer, subset: str, spec: DatasetSpec):
    cfg = TokenizedProbingDatasetConfig(
        dataset_id=f"activation_{subset}",
        hf_repo=spec.hf_repo,
        subset=subset,
        split=spec.split,
        max_length=spec.max_length,
        default_ignore=False,
        ignore_buffer=0,
        last_span_token=False,
        pos_weight=1.0,
        neg_weight=1.0,
        shuffle=True,
        seed=spec.seed,
        process_on_the_fly=False,
        max_num_samples=spec.max_samples,
    )
    return create_probing_dataset(cfg, tokenizer)


def collect_multilayer_activations_for_model(
    *,
    model_key: str,
    model_name: str,
    subset: str,
    dataset_spec: DatasetSpec,
    collection_spec: CollectionSpec,
) -> Dict[str, Any]:
    print(f"\nLoading model/tokenizer: {model_key} ({model_name})")
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.eval()

    dataset = build_tokenized_dataset(tokenizer, subset=subset, spec=dataset_spec)
    print(f"Dataset size ({subset}): {len(dataset)}")

    layer_indices = sorted(collection_spec.layers)
    max_tokens_per_label = collection_spec.max_tokens_per_label

    collected = {
        layer: {"positive": [], "negative": []}
        for layer in layer_indices
    }

    progress = tqdm(range(len(dataset)), desc=f"Collecting {model_key}")
    for idx in progress:
        item = dataset[idx]
        if item is None:
            continue

        input_ids = item["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(model.device)
        labels = item["classification_labels"].cpu().numpy()
        valid = item["attention_mask"].cpu().numpy().astype(bool)

        pos_idx = np.where((labels == 1.0) & valid)[0]
        neg_idx = np.where((labels == 0.0) & valid)[0]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        for layer in layer_indices:
            hidden = hidden_states[layer + 1][0].float().cpu().numpy()

            if len(pos_idx) > 0:
                need = max_tokens_per_label - len(collected[layer]["positive"])
                if need > 0:
                    take = pos_idx[:need]
                    collected[layer]["positive"].extend(hidden[take])

            if len(neg_idx) > 0:
                need = max_tokens_per_label - len(collected[layer]["negative"])
                if need > 0:
                    take = neg_idx[:need]
                    collected[layer]["negative"].extend(hidden[take])

        enough = all(
            len(collected[layer]["positive"]) >= max_tokens_per_label
            and len(collected[layer]["negative"]) >= max_tokens_per_label
            for layer in layer_indices
        )
        if enough:
            break

        del outputs

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_layer = {}
    for layer in layer_indices:
        pos = np.array(collected[layer]["positive"])
        neg = np.array(collected[layer]["negative"])

        if collection_spec.enforce_ratio_on_finalize:
            pos, neg = _sample_with_ratio(
                pos,
                neg,
                pos_neg_ratio=collection_spec.pos_neg_ratio,
                seed=dataset_spec.seed + layer,
                sample_total=collection_spec.finalize_sample_total,
            )

        per_layer[layer] = {"positive": pos, "negative": neg}
        print(f"{model_key} layer {layer}: {len(pos)} hallucinated / {len(neg)} supported")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "subset": subset,
        "per_layer": per_layer,
    }


def equalize_sweep_results_for_fair_comparison(
    sweep_results: Dict[str, Any],
    layers: List[int],
    *,
    pos_neg_ratio: float = 1.0,
    sample_total_per_layer: int | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    model_keys = list(sweep_results.keys())
    out: Dict[str, Any] = {}

    # Initialize shallow metadata copy.
    for model_key in model_keys:
        src = sweep_results[model_key]
        out[model_key] = {
            "model_key": src["model_key"],
            "model_name": src["model_name"],
            "subset": src["subset"],
            "per_layer": {},
        }

    for layer in layers:
        feasible_counts: list[tuple[int, int]] = []
        for model_key in model_keys:
            pos = sweep_results[model_key]["per_layer"][layer]["positive"]
            neg = sweep_results[model_key]["per_layer"][layer]["negative"]

            # Build a temporary feasible sample for this model/layer, then intersect across models.
            pos_s, neg_s = _sample_with_ratio(
                pos,
                neg,
                pos_neg_ratio=pos_neg_ratio,
                seed=seed + layer,
                sample_total=sample_total_per_layer,
            )
            feasible_counts.append((len(pos_s), len(neg_s)))

        shared_n_pos = min(c[0] for c in feasible_counts)
        shared_n_neg = min(c[1] for c in feasible_counts)

        for i, model_key in enumerate(model_keys):
            pos = sweep_results[model_key]["per_layer"][layer]["positive"]
            neg = sweep_results[model_key]["per_layer"][layer]["negative"]

            if shared_n_pos == 0 or shared_n_neg == 0:
                hidden_dim = pos.shape[1] if len(pos) else (neg.shape[1] if len(neg) else 0)
                pos_eq = np.empty((0, hidden_dim), dtype=pos.dtype if len(pos) else np.float32)
                neg_eq = np.empty((0, hidden_dim), dtype=neg.dtype if len(neg) else np.float32)
            else:
                rng = np.random.default_rng(seed + layer * 100 + i)
                pos_idx = rng.choice(len(pos), size=shared_n_pos, replace=False)
                neg_idx = rng.choice(len(neg), size=shared_n_neg, replace=False)
                pos_eq = pos[pos_idx]
                neg_eq = neg[neg_idx]

            out[model_key]["per_layer"][layer] = {
                "positive": pos_eq,
                "negative": neg_eq,
            }

    return out


def _fisher_ratio(pos: np.ndarray, neg: np.ndarray) -> float:
    mu_diff = pos.mean(axis=0) - neg.mean(axis=0)
    between = float(np.sum(mu_diff ** 2))
    within = float(np.mean(pos.var(axis=0) + neg.var(axis=0))) + 1e-8
    return between / within


def _centroid_cosine_distance(pos: np.ndarray, neg: np.ndarray) -> float:
    a = pos.mean(axis=0)
    b = neg.mean(axis=0)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(1.0 - np.dot(a, b) / denom)


def compute_separation_metrics(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    seed: int = 42,
    pca_dims: int = 20,
    pos_neg_ratio: float = 1.0,
    sample_total: int | None = None,
) -> Dict[str, float]:
    if pos_neg_ratio <= 0:
        raise ValueError("pos_neg_ratio must be > 0")

    if len(pos) == 0 or len(neg) == 0:
        return {
            "n_pos": 0.0,
            "n_neg": 0.0,
            "pos_neg_ratio_used": float(pos_neg_ratio),
            "silhouette_pca10": np.nan,
            "centroid_l2_hidden": np.nan,
            "centroid_cosine_dist_hidden": np.nan,
            "fisher_ratio_hidden": np.nan,
            "linear_probe_auc_pca20": np.nan,
            "linear_probe_acc_pca20": np.nan,
            "kmeans_ari_pca20": np.nan,
            "kmeans_nmi_pca20": np.nan,
        }

    max_pos = len(pos)
    max_neg = len(neg)

    if sample_total is not None:
        if sample_total < 30:
            raise ValueError("sample_total must be >= 30 when provided")
        n_pos_target = max(1, int(round(sample_total * pos_neg_ratio / (1.0 + pos_neg_ratio))))
        n_neg_target = max(1, sample_total - n_pos_target)
    else:
        n_neg_target = min(max_neg, int(max_pos / pos_neg_ratio))
        n_pos_target = int(round(n_neg_target * pos_neg_ratio))

    n_pos = min(max_pos, n_pos_target)
    n_neg = min(max_neg, n_neg_target)

    if n_pos < 15 or n_neg < 15:
        return {
            "n_pos": float(n_pos),
            "n_neg": float(n_neg),
            "pos_neg_ratio_used": float(pos_neg_ratio),
            "silhouette_pca10": np.nan,
            "centroid_l2_hidden": np.nan,
            "centroid_cosine_dist_hidden": np.nan,
            "fisher_ratio_hidden": np.nan,
            "linear_probe_auc_pca20": np.nan,
            "linear_probe_acc_pca20": np.nan,
            "kmeans_ari_pca20": np.nan,
            "kmeans_nmi_pca20": np.nan,
        }

    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(max_pos, size=n_pos, replace=False)
    neg_idx = rng.choice(max_neg, size=n_neg, replace=False)

    pos_b = pos[pos_idx]
    neg_b = neg[neg_idx]
    X = np.concatenate([pos_b, neg_b], axis=0)
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    pca_k = min(pca_dims, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=pca_k, random_state=seed).fit_transform(X)

    pca_10 = min(10, X_pca.shape[1])
    sil = silhouette_score(X_pca[:, :pca_10], y)

    c_l2 = float(np.linalg.norm(pos_b.mean(axis=0) - neg_b.mean(axis=0)))
    c_cos = _centroid_cosine_distance(pos_b, neg_b)
    fisher = _fisher_ratio(pos_b, neg_b)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    kmeans = KMeans(n_clusters=2, n_init=20, random_state=seed)
    pred = kmeans.fit_predict(X_pca)
    ari = adjusted_rand_score(y, pred)
    nmi = normalized_mutual_info_score(y, pred)

    return {
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "pos_neg_ratio_used": float(n_pos / max(n_neg, 1)),
        "silhouette_pca10": float(sil),
        "centroid_l2_hidden": c_l2,
        "centroid_cosine_dist_hidden": c_cos,
        "fisher_ratio_hidden": float(fisher),
        "linear_probe_auc_pca20": float(auc),
        "linear_probe_acc_pca20": float(acc),
        "kmeans_ari_pca20": float(ari),
        "kmeans_nmi_pca20": float(nmi),
    }


def summarize_collection_counts(sweep_results: Dict[str, Any], layers: List[int]) -> pd.DataFrame:
    rows = []
    for model_key, model_res in sweep_results.items():
        for layer in layers:
            pos = model_res["per_layer"][layer]["positive"]
            neg = model_res["per_layer"][layer]["negative"]
            hidden_dim = pos.shape[1] if len(pos) else (neg.shape[1] if len(neg) else 0)
            rows.append(
                {
                    "model": model_key,
                    "layer": layer,
                    "hallucinated_tokens": len(pos),
                    "supported_tokens": len(neg),
                    "hidden_dim": hidden_dim,
                }
            )
    return pd.DataFrame(rows)


def build_metrics_table(sweep_results: Dict[str, Any], layers: List[int], seed: int = 42) -> pd.DataFrame:
    rows = []
    for model_key, model_res in sweep_results.items():
        for layer in layers:
            pos = model_res["per_layer"][layer]["positive"]
            neg = model_res["per_layer"][layer]["negative"]
            metrics = compute_separation_metrics(pos, neg, seed=seed)
            rows.append({"model": model_key, "layer": layer, **metrics})
    return pd.DataFrame(rows).sort_values(["model", "layer"]).reset_index(drop=True)


def build_metrics_table_for_ratio(
    sweep_results: Dict[str, Any],
    layers: List[int],
    *,
    pos_neg_ratio: float,
    sample_total: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for model_key, model_res in sweep_results.items():
        for layer in layers:
            pos = model_res["per_layer"][layer]["positive"]
            neg = model_res["per_layer"][layer]["negative"]
            metrics = compute_separation_metrics(
                pos,
                neg,
                seed=seed,
                pos_neg_ratio=pos_neg_ratio,
                sample_total=sample_total,
            )
            rows.append({
                "model": model_key,
                "layer": layer,
                "requested_pos_neg_ratio": pos_neg_ratio,
                **metrics,
            })
    return pd.DataFrame(rows).sort_values(["model", "layer"]).reset_index(drop=True)


def build_pca_plot_frame(
    sweep_results: Dict[str, Any],
    layers: List[int],
    *,
    max_points_per_label: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)

    for model_key, model_res in sweep_results.items():
        for layer in layers:
            pos = model_res["per_layer"][layer]["positive"]
            neg = model_res["per_layer"][layer]["negative"]

            if len(pos) == 0 or len(neg) == 0:
                continue

            n_pos = min(len(pos), max_points_per_label)
            n_neg = min(len(neg), max_points_per_label)
            pos_idx = rng.choice(len(pos), size=n_pos, replace=False)
            neg_idx = rng.choice(len(neg), size=n_neg, replace=False)

            pos_s = pos[pos_idx]
            neg_s = neg[neg_idx]
            X = np.concatenate([pos_s, neg_s], axis=0)
            y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

            pca = PCA(n_components=2, random_state=seed)
            X2 = pca.fit_transform(X)

            part = pd.DataFrame(
                {
                    "pc1": X2[:, 0],
                    "pc2": X2[:, 1],
                    "label": np.where(y == 1, "Hallucinated", "Supported"),
                    "model": model_key,
                    "layer": layer,
                    "pc1_var": pca.explained_variance_ratio_[0],
                    "pc2_var": pca.explained_variance_ratio_[1],
                }
            )
            rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["pc1", "pc2", "label", "model", "layer", "pc1_var", "pc2_var"])

    return pd.concat(rows, ignore_index=True)
