"""Active selection policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .models import TreeEnsembleRegressor


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _trim_by_prediction_quantiles(
    pred_mean: np.ndarray,
    *,
    budget: int,
    quantile_min: float,
    quantile_max: float,
) -> np.ndarray:
    if pred_mean.size == 0:
        return np.array([], dtype=int)
    q_min = float(np.clip(quantile_min, 0.0, 1.0))
    q_max = float(np.clip(quantile_max, 0.0, 1.0))
    if q_max < q_min:
        q_min, q_max = q_max, q_min
    lo = float(np.quantile(pred_mean, q_min))
    hi = float(np.quantile(pred_mean, q_max))
    keep = np.where((pred_mean >= lo) & (pred_mean <= hi))[0]
    if keep.size < max(1, budget):
        return np.arange(pred_mean.size, dtype=int)
    return keep.astype(int)


def _kmeanspp_greedy_indices(
    embeddings: np.ndarray,
    *,
    budget: int,
    random_state: int,
    seed_score: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy k-means++ style index selection."""
    n = int(embeddings.shape[0])
    budget = min(int(budget), n)
    if budget <= 0 or n <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(random_state)
    if seed_score is None:
        seed = int(rng.integers(0, n))
    else:
        seed = int(np.argmax(seed_score))
    selected = [seed]
    min_dist_sq = np.sum((embeddings - embeddings[seed]) ** 2, axis=1)
    min_dist_sq[seed] = 0.0
    while len(selected) < budget:
        probs = np.clip(min_dist_sq, 0.0, None)
        total = float(probs.sum())
        if total <= 1e-12:
            remaining = np.setdiff1d(np.arange(n, dtype=int), np.asarray(selected, dtype=int), assume_unique=False)
            if remaining.size == 0:
                break
            nxt = int(remaining[int(rng.integers(0, remaining.size))])
        else:
            probs = probs / total
            nxt = int(rng.choice(n, p=probs))
            while nxt in selected:
                nxt = int(rng.choice(n, p=probs))
        selected.append(nxt)
        dist_sq = np.sum((embeddings - embeddings[nxt]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        min_dist_sq[np.asarray(selected, dtype=int)] = 0.0
    return np.asarray(sorted(selected), dtype=int)


def select_random(candidate_df: pd.DataFrame, *, budget: int, random_state: int) -> np.ndarray:
    """Uniform random selection without replacement."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(len(candidate_df), size=budget, replace=False))


def select_diversity(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    random_state: int,
) -> np.ndarray:
    """Greedy farthest-point selection in the model feature space."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    features = model.transform(candidate_df)
    source_centroid = model.source_feature_centroid_
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")
    rng = np.random.default_rng(random_state)
    remaining = list(range(len(candidate_df)))
    selected: list[int] = []
    base_dist = np.linalg.norm(features - source_centroid.reshape(1, -1), axis=1)
    while remaining and len(selected) < budget:
        if not selected:
            scaled = _minmax_scale(base_dist[remaining])
            scores = [(idx, float(score + rng.uniform(0.0, 1e-9))) for idx, score in zip(remaining, scaled)]
        else:
            raw_scores = []
            for idx in remaining:
                dist_to_selected = np.linalg.norm(features[idx] - features[selected], axis=1)
                raw_scores.append(float(dist_to_selected.min()))
            scaled = _minmax_scale(np.asarray(raw_scores, dtype=float))
            scores = [(idx, float(score + rng.uniform(0.0, 1e-9))) for idx, score in zip(remaining, scaled)]
        chosen = max(scores, key=lambda pair: pair[1])[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return np.array(sorted(selected), dtype=int)


def select_shift_aware_diversity(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    random_state: int,
    shift_weight: float = 0.75,
    coverage_weight: float = 0.50,
) -> np.ndarray:
    """Greedy farthest-point selection after reweighting dimensions by estimated shift magnitude."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    features = model.transform(candidate_df)
    source_centroid = model.source_feature_centroid_
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")

    candidate_centroid = features.mean(axis=0)
    shift_vector = np.abs(candidate_centroid - source_centroid)
    dim_weights = 1.0 + float(max(shift_weight, 0.0)) * shift_vector
    weighted_features = features * dim_weights.reshape(1, -1)
    weighted_source_centroid = source_centroid * dim_weights

    rng = np.random.default_rng(random_state)
    remaining = list(range(len(candidate_df)))
    selected: list[int] = []
    base_dist = np.linalg.norm(weighted_features - weighted_source_centroid.reshape(1, -1), axis=1)
    plain_dist = np.linalg.norm(features - source_centroid.reshape(1, -1), axis=1)
    coverage_weight = float(np.clip(coverage_weight, 0.0, 1.0))

    while remaining and len(selected) < budget:
        if not selected:
            shifted = _minmax_scale(base_dist[remaining])
            plain = _minmax_scale(plain_dist[remaining])
            scores = [
                (
                    idx,
                    float(coverage_weight * shifted_score + (1.0 - coverage_weight) * plain_score + rng.uniform(0.0, 1e-9)),
                )
                for idx, shifted_score, plain_score in zip(remaining, shifted, plain)
            ]
        else:
            shifted_scores = []
            plain_scores = []
            for idx in remaining:
                shifted_dist = np.linalg.norm(weighted_features[idx] - weighted_features[selected], axis=1)
                plain_dist_to_selected = np.linalg.norm(features[idx] - features[selected], axis=1)
                shifted_scores.append(float(shifted_dist.min()))
                plain_scores.append(float(plain_dist_to_selected.min()))
            shifted = _minmax_scale(np.asarray(shifted_scores, dtype=float))
            plain = _minmax_scale(np.asarray(plain_scores, dtype=float))
            scores = [
                (
                    idx,
                    float(coverage_weight * shifted_score + (1.0 - coverage_weight) * plain_score + rng.uniform(0.0, 1e-9)),
                )
                for idx, shifted_score, plain_score in zip(remaining, shifted, plain)
            ]
        chosen = max(scores, key=lambda pair: pair[1])[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return np.array(sorted(selected), dtype=int)


def _farthest_point_subset(
    *,
    features: np.ndarray,
    subset_indices: np.ndarray,
    budget: int,
    source_centroid: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    if budget <= 0 or subset_indices.size == 0:
        return []
    remaining = [int(idx) for idx in subset_indices.tolist()]
    budget = min(int(budget), len(remaining))
    selected: list[int] = []
    base_dist = np.linalg.norm(features[remaining] - source_centroid.reshape(1, -1), axis=1)
    while remaining and len(selected) < budget:
        if not selected:
            scaled = _minmax_scale(base_dist)
            scores = [(idx, float(score + rng.uniform(0.0, 1e-9))) for idx, score in zip(remaining, scaled)]
        else:
            raw_scores = []
            for idx in remaining:
                dist_to_selected = np.linalg.norm(features[idx] - features[selected], axis=1)
                raw_scores.append(float(dist_to_selected.min()))
            scaled = _minmax_scale(np.asarray(raw_scores, dtype=float))
            scores = [(idx, float(score + rng.uniform(0.0, 1e-9))) for idx, score in zip(remaining, scaled)]
        chosen = max(scores, key=lambda pair: pair[1])[0]
        selected.append(chosen)
        chosen_pos = remaining.index(chosen)
        remaining.pop(chosen_pos)
        if len(base_dist) > chosen_pos:
            base_dist = np.delete(base_dist, chosen_pos)
    return selected


def select_stratified_diversity(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    random_state: int,
    strata_bins: int = 4,
    family_stratify: bool = True,
) -> np.ndarray:
    """Stratified diversity: allocate across prediction strata then farthest-point within each stratum."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)

    features = model.transform(candidate_df)
    source_centroid = model.source_feature_centroid_
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")
    bundle = model.predict_distribution(candidate_df)
    pred_mean = bundle.mean
    bins = max(int(strata_bins), 2)
    q_edges = np.quantile(pred_mean, np.linspace(0.0, 1.0, bins + 1))
    q_edges = np.unique(q_edges)
    pred_bucket = np.digitize(pred_mean, q_edges[1:-1], right=False)
    if family_stratify and "family" in candidate_df.columns:
        family_codes = candidate_df["family"].astype("category").cat.codes.to_numpy(dtype=int)
    else:
        family_codes = np.zeros(len(candidate_df), dtype=int)

    strata_keys: list[tuple[int, int]] = list(zip(pred_bucket.tolist(), family_codes.tolist()))
    strata_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(strata_keys):
        strata_to_indices.setdefault(key, []).append(idx)

    strata_items = [(key, np.array(indices, dtype=int)) for key, indices in strata_to_indices.items() if len(indices) > 0]
    if not strata_items:
        return select_diversity(candidate_df, model=model, budget=budget, random_state=random_state)

    rng = np.random.default_rng(random_state)
    # Allocate at least one sample per stratum while budget permits, then proportional fill.
    allocations = {key: 0 for key, _ in strata_items}
    sorted_items = sorted(strata_items, key=lambda item: item[1].size, reverse=True)
    remaining_budget = budget
    for key, idxs in sorted_items:
        if remaining_budget <= 0:
            break
        allocations[key] = min(1, idxs.size)
        remaining_budget -= allocations[key]
    if remaining_budget > 0:
        total = sum(idxs.size for _, idxs in sorted_items)
        for key, idxs in sorted_items:
            if remaining_budget <= 0:
                break
            cap = idxs.size - allocations[key]
            if cap <= 0:
                continue
            share = int(round(remaining_budget * (idxs.size / max(total, 1))))
            take = int(np.clip(share, 0, cap))
            allocations[key] += take
        assigned = sum(allocations.values())
        while assigned < budget:
            expandable = [item for item in sorted_items if allocations[item[0]] < item[1].size]
            if not expandable:
                break
            chosen_key, _ = expandable[int(rng.integers(0, len(expandable)))]
            allocations[chosen_key] += 1
            assigned += 1

    selected: list[int] = []
    for key, idxs in sorted_items:
        local_budget = int(allocations.get(key, 0))
        if local_budget <= 0:
            continue
        selected.extend(
            _farthest_point_subset(
                features=features,
                subset_indices=idxs,
                budget=local_budget,
                source_centroid=source_centroid,
                rng=rng,
            )
        )

    if len(selected) < budget:
        selected_set = set(selected)
        remaining = np.array([idx for idx in range(len(candidate_df)) if idx not in selected_set], dtype=int)
        selected.extend(
            _farthest_point_subset(
                features=features,
                subset_indices=remaining,
                budget=budget - len(selected),
                source_centroid=source_centroid,
                rng=rng,
            )
        )

    return np.array(sorted(selected[:budget]), dtype=int)


def select_uncertainty(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    quantile_min: float = 0.0,
    quantile_max: float = 1.0,
) -> np.ndarray:
    """Choose the most uncertain candidates according to ensemble spread."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    bundle = model.predict_distribution(candidate_df)
    keep = _trim_by_prediction_quantiles(
        bundle.mean,
        budget=budget,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
    )
    local_order = np.argsort(-bundle.std[keep])
    ordering = keep[local_order]
    return np.sort(ordering[:budget].astype(int))


def _pass_probability_from_bundle(bundle: object, *, threshold: float) -> np.ndarray:
    member_predictions = np.asarray(getattr(bundle, "member_predictions"), dtype=float)
    if member_predictions.ndim != 2:
        raise ValueError("Expected member_predictions to be a 2D array.")
    return (member_predictions <= threshold).mean(axis=0)


def select_entropy_threshold_diversity(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    threshold: float,
    entropy_weight: float,
    std_weight: float,
    threshold_weight: float,
    diversity_weight: float,
    threshold_beta: float,
    random_state: int,
    quantile_min: float = 0.0,
    quantile_max: float = 1.0,
) -> np.ndarray:
    """Greedy acquisition using pass-entropy, uncertainty spread, threshold proximity, and diversity."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)

    bundle = model.predict_distribution(candidate_df)
    keep = _trim_by_prediction_quantiles(
        bundle.mean,
        budget=budget,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
    )

    pass_probability = _pass_probability_from_bundle(bundle, threshold=threshold)
    p = np.clip(pass_probability, 1e-8, 1.0 - 1e-8)
    entropy_score = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    entropy_score = _minmax_scale(entropy_score)

    uncertainty = np.zeros(len(candidate_df), dtype=float)
    uncertainty[keep] = _minmax_scale(bundle.std[keep])
    threshold_score = np.exp(-np.abs(bundle.mean - threshold) / max(threshold_beta, 1e-6))
    threshold_score = _minmax_scale(threshold_score)

    features = model.transform(candidate_df)
    source_centroid = model.source_feature_centroid_
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")

    rng = np.random.default_rng(random_state)
    remaining = list(keep.tolist())
    selected: list[int] = []
    base_diversity = np.linalg.norm(features - source_centroid.reshape(1, -1), axis=1)
    base_diversity = _minmax_scale(base_diversity)

    while remaining and len(selected) < budget:
        if selected:
            raw_diversity = []
            for idx in remaining:
                dist_to_selected = np.linalg.norm(features[idx] - features[selected], axis=1)
                raw_diversity.append(float(dist_to_selected.min()))
            scaled_diversity = _minmax_scale(np.asarray(raw_diversity, dtype=float))
            diversity_lookup = {idx: float(score) for idx, score in zip(remaining, scaled_diversity)}
        else:
            diversity_lookup = {idx: float(base_diversity[idx]) for idx in remaining}

        candidate_scores: list[tuple[int, float]] = []
        for idx in remaining:
            diversity = diversity_lookup[idx]
            score = (
                entropy_weight * float(entropy_score[idx])
                + std_weight * float(uncertainty[idx])
                + threshold_weight * float(threshold_score[idx])
                + diversity_weight * float(diversity)
                + float(rng.uniform(0.0, 1e-9))
            )
            candidate_scores.append((idx, score))
        best_idx = max(candidate_scores, key=lambda pair: pair[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(sorted(selected), dtype=int)


def select_badge_proxy(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    random_state: int,
    quantile_min: float = 0.0,
    quantile_max: float = 1.0,
) -> np.ndarray:
    """BADGE-style proxy using std-weighted feature embeddings and k-means++ seeding."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    bundle = model.predict_distribution(candidate_df)
    keep = _trim_by_prediction_quantiles(
        bundle.mean,
        budget=budget,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
    )
    features = model.transform(candidate_df)
    std = np.asarray(bundle.std, dtype=float)
    embeddings = features[keep] * std[keep].reshape(-1, 1)
    seed_score = std[keep] * (np.linalg.norm(embeddings, axis=1) + 1e-12)
    local_idx = _kmeanspp_greedy_indices(
        embeddings,
        budget=budget,
        random_state=random_state,
        seed_score=seed_score,
    )
    return np.sort(keep[local_idx].astype(int))


def select_density_ratio_uncertainty_diversity(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    random_state: int,
    ratio_weight: float = 0.75,
    diversity_weight: float = 0.25,
    quantile_min: float = 0.0,
    quantile_max: float = 1.0,
) -> np.ndarray:
    """Shift-aware selection using density ratio * uncertainty with diversity filtering."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    source_ref = getattr(model, "source_feature_reference_", None)
    if source_ref is None or len(source_ref) == 0:
        return select_uncertainty(
            candidate_df,
            model=model,
            budget=budget,
            quantile_min=quantile_min,
            quantile_max=quantile_max,
        )
    bundle = model.predict_distribution(candidate_df)
    keep = _trim_by_prediction_quantiles(
        bundle.mean,
        budget=budget,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
    )
    features = model.transform(candidate_df)
    X_train = np.vstack([source_ref, features])
    y_train = np.concatenate([np.zeros(len(source_ref), dtype=int), np.ones(len(features), dtype=int)])
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=400,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    p_target = np.clip(clf.predict_proba(features)[:, 1], 1e-6, 1.0 - 1e-6)
    density_ratio = p_target / (1.0 - p_target)
    ratio_std = _minmax_scale(density_ratio * np.clip(bundle.std, 0.0, None))

    source_centroid = model.source_feature_centroid_
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")
    rng = np.random.default_rng(random_state)
    remaining = list(keep.tolist())
    selected: list[int] = []
    base_diversity = np.linalg.norm(features - source_centroid.reshape(1, -1), axis=1)
    base_diversity = _minmax_scale(base_diversity)
    ratio_weight = float(np.clip(ratio_weight, 0.0, 1.0))
    diversity_weight = float(np.clip(diversity_weight, 0.0, 1.0))
    norm = max(ratio_weight + diversity_weight, 1e-12)
    ratio_weight /= norm
    diversity_weight /= norm

    while remaining and len(selected) < budget:
        if selected:
            raw_diversity = []
            for idx in remaining:
                dist_to_selected = np.linalg.norm(features[idx] - features[selected], axis=1)
                raw_diversity.append(float(dist_to_selected.min()))
            scaled_diversity = _minmax_scale(np.asarray(raw_diversity, dtype=float))
            diversity_lookup = {idx: float(score) for idx, score in zip(remaining, scaled_diversity)}
        else:
            diversity_lookup = {idx: float(base_diversity[idx]) for idx in remaining}
        scores = []
        for idx in remaining:
            score = (
                ratio_weight * float(ratio_std[idx])
                + diversity_weight * float(diversity_lookup[idx])
                + float(rng.uniform(0.0, 1e-9))
            )
            scores.append((idx, score))
        chosen = max(scores, key=lambda pair: pair[1])[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return np.asarray(sorted(selected), dtype=int)


def select_two_stage_explore_exploit(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    random_state: int,
    explore_fraction: float = 0.5,
    quantile_min: float = 0.0,
    quantile_max: float = 1.0,
) -> np.ndarray:
    """Two-stage selection: diversity exploration then uncertainty-weighted exploitation."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    n_explore = int(np.clip(np.ceil(budget * float(np.clip(explore_fraction, 0.2, 0.8))), 1, budget))
    first = select_diversity(candidate_df, model=model, budget=n_explore, random_state=random_state)
    if n_explore >= budget:
        return np.asarray(sorted(first), dtype=int)
    bundle = model.predict_distribution(candidate_df)
    features = model.transform(candidate_df)
    keep = _trim_by_prediction_quantiles(
        bundle.mean,
        budget=budget,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
    )
    remaining = [int(idx) for idx in keep.tolist() if int(idx) not in set(first.tolist())]
    if not remaining:
        return np.asarray(sorted(first[:budget]), dtype=int)
    rng = np.random.default_rng(random_state + 17)
    selected = list(first.tolist())
    std_scaled = _minmax_scale(bundle.std)
    while remaining and len(selected) < budget:
        raw_diversity = []
        for idx in remaining:
            dist_to_selected = np.linalg.norm(features[idx] - features[np.asarray(selected, dtype=int)], axis=1)
            raw_diversity.append(float(dist_to_selected.min()))
        div_scaled = _minmax_scale(np.asarray(raw_diversity, dtype=float))
        scores = []
        for idx, div_s in zip(remaining, div_scaled):
            score = 0.7 * float(std_scaled[idx]) + 0.3 * float(div_s) + float(rng.uniform(0.0, 1e-9))
            scores.append((idx, score))
        chosen = max(scores, key=lambda pair: pair[1])[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return np.asarray(sorted(selected), dtype=int)


def select_uncertainty_threshold_diversity(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    threshold: float,
    uncertainty_weight: float,
    threshold_weight: float,
    diversity_weight: float,
    threshold_beta: float,
    random_state: int,
    quantile_min: float = 0.0,
    quantile_max: float = 1.0,
) -> np.ndarray:
    """Greedy acquisition using uncertainty, near-threshold interest, and diversity."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)

    bundle = model.predict_distribution(candidate_df)
    keep = _trim_by_prediction_quantiles(
        bundle.mean,
        budget=budget,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
    )
    uncertainty = np.zeros(len(candidate_df), dtype=float)
    uncertainty[keep] = _minmax_scale(bundle.std[keep])
    threshold_score = np.exp(-np.abs(bundle.mean - threshold) / max(threshold_beta, 1e-6))
    threshold_score = _minmax_scale(threshold_score)
    features = model.transform(candidate_df)
    source_centroid = model.source_feature_centroid_
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")

    rng = np.random.default_rng(random_state)
    remaining = list(keep.tolist())
    selected: list[int] = []
    base_diversity = np.linalg.norm(features - source_centroid.reshape(1, -1), axis=1)
    base_diversity = _minmax_scale(base_diversity)

    while remaining and len(selected) < budget:
        if selected:
            raw_diversity = []
            for idx in remaining:
                dist_to_selected = np.linalg.norm(features[idx] - features[selected], axis=1)
                raw_diversity.append(float(dist_to_selected.min()))
            scaled_diversity = _minmax_scale(np.asarray(raw_diversity, dtype=float))
            diversity_lookup = {idx: float(score) for idx, score in zip(remaining, scaled_diversity)}
        else:
            diversity_lookup = {idx: float(base_diversity[idx]) for idx in remaining}

        candidate_scores: list[tuple[int, float]] = []
        for idx in remaining:
            diversity = diversity_lookup[idx]
            score = (
                uncertainty_weight * float(uncertainty[idx])
                + threshold_weight * float(threshold_score[idx])
                + diversity_weight * float(diversity)
                + float(rng.uniform(0.0, 1e-9))
            )
            candidate_scores.append((idx, score))
        best_idx = max(candidate_scores, key=lambda pair: pair[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(sorted(selected), dtype=int)


def select_uncertainty_threshold_diversity_capped(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
    threshold: float,
    uncertainty_weight: float,
    threshold_weight: float,
    diversity_weight: float,
    density_weight: float,
    threshold_beta: float,
    uncertainty_cap_quantile: float,
    random_state: int,
    quantile_min: float = 0.0,
    quantile_max: float = 1.0,
) -> np.ndarray:
    """Greedy acquisition with capped uncertainty and representativeness."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)

    bundle = model.predict_distribution(candidate_df)
    keep = _trim_by_prediction_quantiles(
        bundle.mean,
        budget=budget,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
    )
    features = model.transform(candidate_df)
    source_centroid = model.source_feature_centroid_
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")

    cap_q = float(np.clip(uncertainty_cap_quantile, 0.0, 1.0))
    cap_value = float(np.quantile(bundle.std[keep], cap_q)) if keep.size else 0.0
    uncertainty = np.zeros(len(candidate_df), dtype=float)
    clipped = np.minimum(bundle.std[keep], cap_value) if keep.size else np.array([], dtype=float)
    uncertainty[keep] = _minmax_scale(clipped)
    threshold_score = np.exp(-np.abs(bundle.mean - threshold) / max(threshold_beta, 1e-6))
    threshold_score = _minmax_scale(threshold_score)

    n_neighbors = min(10, max(2, int(np.sqrt(max(len(candidate_df), 1)))))
    density_raw = np.zeros(len(candidate_df), dtype=float)
    for idx in range(len(candidate_df)):
        dist = np.linalg.norm(features - features[idx], axis=1)
        nearest = np.partition(dist, min(n_neighbors, len(dist) - 1))[:n_neighbors]
        density_raw[idx] = float(np.mean(nearest))
    representativeness = 1.0 - _minmax_scale(density_raw)

    rng = np.random.default_rng(random_state)
    remaining = list(keep.tolist())
    selected: list[int] = []
    base_diversity = np.linalg.norm(features - source_centroid.reshape(1, -1), axis=1)
    base_diversity = _minmax_scale(base_diversity)

    while remaining and len(selected) < budget:
        if selected:
            raw_diversity = []
            for idx in remaining:
                dist_to_selected = np.linalg.norm(features[idx] - features[selected], axis=1)
                raw_diversity.append(float(dist_to_selected.min()))
            scaled_diversity = _minmax_scale(np.asarray(raw_diversity, dtype=float))
            diversity_lookup = {idx: float(score) for idx, score in zip(remaining, scaled_diversity)}
        else:
            diversity_lookup = {idx: float(base_diversity[idx]) for idx in remaining}

        candidate_scores: list[tuple[int, float]] = []
        for idx in remaining:
            diversity = diversity_lookup[idx]
            score = (
                uncertainty_weight * float(uncertainty[idx])
                + threshold_weight * float(threshold_score[idx])
                + diversity_weight * float(diversity)
                + density_weight * float(representativeness[idx])
                + float(rng.uniform(0.0, 1e-9))
            )
            candidate_scores.append((idx, score))
        best_idx = max(candidate_scores, key=lambda pair: pair[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(sorted(selected), dtype=int)


def select_oracle_residual(
    candidate_df: pd.DataFrame,
    *,
    model: TreeEnsembleRegressor,
    budget: int,
) -> np.ndarray:
    """Diagnostic upper-bound selector using true residuals (oracle control)."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return np.array([], dtype=int)
    if "error_rate" not in candidate_df.columns:
        raise ValueError("Oracle acquisition requires candidate_df to contain 'error_rate'.")
    bundle = model.predict_distribution(candidate_df)
    y_true = candidate_df["error_rate"].to_numpy(dtype=float)
    residual = np.abs(y_true - bundle.mean)
    ordering = np.argsort(-residual)
    return np.sort(ordering[:budget].astype(int))
