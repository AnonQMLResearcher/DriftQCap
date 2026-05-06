"""Model implementations for the DriftQCap research scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ModelConfig
from .features import CATEGORICAL_FEATURES, COARSE_NUMERIC_FEATURES, FULL_NUMERIC_FEATURES
import hashlib

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except ImportError:  # pragma: no cover
    TabNetRegressor = None

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    Data = None
    DataLoader = None
    GCNConv = None
    global_mean_pool = None

FeatureMode = Literal["full", "coarse"]


@dataclass
class PredictionBundle:
    """Predictions and uncertainty estimates from a model."""

    mean: np.ndarray
    std: np.ndarray
    member_predictions: np.ndarray
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    pass_probability_raw: np.ndarray | None = None
    pass_probability_calibrated: np.ndarray | None = None


class BaseCapabilityRegressor(ABC):
    """Abstract interface for capability predictors."""

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
    ) -> "BaseCapabilityRegressor":
        raise NotImplementedError

    @abstractmethod
    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {}



def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class TreeEnsembleRegressor(BaseCapabilityRegressor):
    """Tree-ensemble capability regressor using per-tree predictions as an ensemble."""
    model_family = "rf"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        feature_mode: FeatureMode = "full",
        numeric_features: Iterable[str] | None = None,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ) -> None:
        self.config = model_config
        self.feature_mode: FeatureMode = feature_mode
        if numeric_features is None:
            numeric_features = FULL_NUMERIC_FEATURES if feature_mode == "full" else COARSE_NUMERIC_FEATURES
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features if feature_mode == "full" else ["family"])
        self.feature_columns = [*self.numeric_features, *self.categorical_features]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", _make_one_hot_encoder(), self.categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self.forest = RandomForestRegressor(
            n_estimators=self.config.n_members,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=True,
            max_samples=self.config.bootstrap_fraction,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_seed,
        )
        self.source_feature_centroid_: np.ndarray | None = None
        self.source_feature_variance_: np.ndarray | None = None
        self.source_feature_rank_: np.ndarray | None = None
        self.source_feature_reference_: np.ndarray | None = None
        self._fitted = False

    def new_like(self, *, random_seed: int | None = None, feature_mode: FeatureMode | None = None) -> "TreeEnsembleRegressor":
        chosen_mode = feature_mode or self.feature_mode
        config = replace(self.config, random_seed=self.config.random_seed if random_seed is None else random_seed)
        return TreeEnsembleRegressor(model_config=config, feature_mode=chosen_mode)

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
    ) -> "TreeEnsembleRegressor":
        X = df[self.feature_columns].copy()
        y = df[target_col].to_numpy(dtype=float)
        transformed = self.preprocessor.fit_transform(X)
        self.source_feature_centroid_ = transformed.mean(axis=0)
        self.source_feature_variance_ = transformed.var(axis=0)
        self.source_feature_rank_ = np.argsort(-self.source_feature_variance_)
        self.source_feature_reference_ = transformed.copy()
        self.forest.fit(transformed, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self.preprocessor.transform(df[self.feature_columns].copy())

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        self._check_is_fitted()
        X = self.transform(df)
        member_predictions = np.vstack([tree.predict(X) for tree in self.forest.estimators_])
        mean = member_predictions.mean(axis=0)
        std = member_predictions.std(axis=0, ddof=1) if member_predictions.shape[0] > 1 else np.zeros_like(mean)
        return PredictionBundle(mean=mean, std=std, member_predictions=member_predictions)

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {"model_family": "rf", "ensemble_members": int(len(getattr(self.forest, "estimators_", [])))}


class _BaggedTabularRegressor(BaseCapabilityRegressor):
    """Shared bagged tabular regressor for non-RF estimator families."""

    model_family: str = "unknown"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        feature_mode: FeatureMode = "full",
        numeric_features: Iterable[str] | None = None,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ) -> None:
        self.config = model_config
        self.feature_mode = feature_mode
        if numeric_features is None:
            numeric_features = FULL_NUMERIC_FEATURES if feature_mode == "full" else COARSE_NUMERIC_FEATURES
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features if feature_mode == "full" else ["family"])
        self.feature_columns = [*self.numeric_features, *self.categorical_features]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", _make_one_hot_encoder(), self.categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self.members_: list[object] = []
        self.source_feature_centroid_: np.ndarray | None = None
        self.source_feature_variance_: np.ndarray | None = None
        self.source_feature_rank_: np.ndarray | None = None
        self.source_feature_reference_: np.ndarray | None = None
        self._fitted = False

    def _member_count(self) -> int:
        return max(1, int(self.config.non_rf_members))

    @abstractmethod
    def _build_member(self, *, random_seed: int) -> object:
        raise NotImplementedError

    def _fit_member(
        self,
        estimator: object,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
    ) -> None:
        if sample_weight is None:
            estimator.fit(X, y)
            return
        try:
            estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            estimator.fit(X, y)

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
    ) -> "_BaggedTabularRegressor":
        X = df[self.feature_columns].copy()
        y = df[target_col].to_numpy(dtype=float)
        transformed = self.preprocessor.fit_transform(X)
        self.source_feature_centroid_ = transformed.mean(axis=0)
        self.source_feature_variance_ = transformed.var(axis=0)
        self.source_feature_rank_ = np.argsort(-self.source_feature_variance_)
        self.source_feature_reference_ = transformed.copy()

        n_rows = transformed.shape[0]
        if n_rows <= 0:
            raise ValueError("Model cannot be fitted on an empty dataframe.")
        rng = np.random.default_rng(self.config.random_seed)
        self.members_ = []
        sample_weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        for idx in range(self._member_count()):
            estimator = self._build_member(random_seed=self.config.random_seed + 10_000 + idx)
            if n_rows > 1:
                boot_size = max(1, int(round(n_rows * float(np.clip(self.config.bootstrap_fraction, 0.1, 1.0)))))
                boot_idx = rng.choice(n_rows, size=boot_size, replace=True).astype(int)
            else:
                boot_idx = np.array([0], dtype=int)
            sw = None if sample_weights is None else sample_weights[boot_idx]
            self._fit_member(estimator, transformed[boot_idx], y[boot_idx], sample_weight=sw)
            self.members_.append(estimator)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self.preprocessor.transform(df[self.feature_columns].copy())

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        self._check_is_fitted()
        X = self.transform(df)
        member_predictions = np.vstack([np.asarray(member.predict(X), dtype=float) for member in self.members_])
        mean = member_predictions.mean(axis=0)
        std = member_predictions.std(axis=0, ddof=1) if member_predictions.shape[0] > 1 else np.zeros_like(mean)
        return PredictionBundle(mean=mean, std=std, member_predictions=member_predictions)

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {"model_family": self.model_family, "ensemble_members": int(len(self.members_))}


class XGBoostRegressorFamily(_BaggedTabularRegressor):
    """Bagged XGBoost family for tabular baseline comparisons."""

    model_family = "xgboost"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        feature_mode: FeatureMode = "full",
        numeric_features: Iterable[str] | None = None,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ) -> None:
        if XGBRegressor is None:
            raise ImportError("xgboost is required for base_estimator='xgboost'.")
        super().__init__(
            model_config=model_config,
            feature_mode=feature_mode,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

    def _build_member(self, *, random_seed: int) -> object:
        return XGBRegressor(
            n_estimators=max(200, int(self.config.n_members)),
            max_depth=max(2, int(self.config.max_depth)),
            learning_rate=float(max(self.config.xgb_learning_rate, 1e-3)),
            subsample=float(np.clip(self.config.xgb_subsample, 0.3, 1.0)),
            colsample_bytree=float(np.clip(self.config.xgb_colsample_bytree, 0.3, 1.0)),
            reg_lambda=float(max(self.config.xgb_reg_lambda, 0.0)),
            objective="reg:squarederror",
            n_jobs=int(max(self.config.n_jobs, 1)),
            random_state=int(random_seed),
        )


class LightGBMRegressorFamily(_BaggedTabularRegressor):
    """Bagged LightGBM family for fast, strong tabular baselines."""

    model_family = "lightgbm"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        feature_mode: FeatureMode = "full",
        numeric_features: Iterable[str] | None = None,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ) -> None:
        if LGBMRegressor is None:
            raise ImportError("lightgbm is required for base_estimator='lightgbm'.")
        super().__init__(
            model_config=model_config,
            feature_mode=feature_mode,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

    def _build_member(self, *, random_seed: int) -> object:
        return LGBMRegressor(
            n_estimators=max(200, int(self.config.n_members)),
            learning_rate=float(max(self.config.lgb_learning_rate, 1e-3)),
            subsample=float(np.clip(self.config.lgb_subsample, 0.3, 1.0)),
            colsample_bytree=float(np.clip(self.config.lgb_colsample_bytree, 0.3, 1.0)),
            reg_lambda=float(max(self.config.lgb_reg_lambda, 0.0)),
            random_state=int(random_seed),
            n_jobs=int(max(self.config.n_jobs, 1)),
            objective="regression",
            verbose=-1,
        )


class MLPRegressorFamily(_BaggedTabularRegressor):
    """Bagged MLP family for tabular baseline comparisons."""

    model_family = "mlp"

    def _build_member(self, *, random_seed: int) -> object:
        return MLPRegressor(
            hidden_layer_sizes=tuple(int(v) for v in self.config.mlp_hidden_layers),
            activation="relu",
            solver="adam",
            alpha=float(max(self.config.mlp_alpha, 1e-8)),
            learning_rate_init=float(max(self.config.mlp_learning_rate_init, 1e-6)),
            max_iter=int(max(self.config.mlp_max_iter, 50)),
            tol=float(max(self.config.mlp_tol, 1e-8)),
            n_iter_no_change=int(max(self.config.mlp_n_iter_no_change, 5)),
            random_state=int(random_seed),
            early_stopping=True,
            validation_fraction=0.15,
        )


if nn is not None:  # pragma: no cover
    class _QPAGraphNet(nn.Module):
        def __init__(self, node_dim: int, global_dim: int, hidden: tuple[int, ...]) -> None:
            super().__init__()
            h0 = int(hidden[0]) if hidden else 128
            h1 = int(hidden[1]) if len(hidden) > 1 else h0
            h2 = int(hidden[2]) if len(hidden) > 2 else h1
            self.conv1 = GCNConv(node_dim, h0)
            self.conv2 = GCNConv(h0, h1)
            self.global_proj = nn.Sequential(
                nn.Linear(global_dim, h1),
                nn.ReLU(),
                nn.Linear(h1, h1),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(h1 + h1, h2),
                nn.ReLU(),
                nn.Linear(h2, 1),
            )

        def forward(self, data: Data) -> torch.Tensor:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            g = global_mean_pool(x, batch)
            gg = self.global_proj(data.global_x)
            out = self.head(torch.cat([g, gg], dim=1)).squeeze(-1)
            return out
else:  # pragma: no cover
    class _QPAGraphNet:
        pass


class QPANNRegressorFamily(BaseCapabilityRegressor):
    """Graph-based qpa_nn estimator with CUDA support and ensemble uncertainty."""

    model_family = "qpa_nn"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        feature_mode: FeatureMode = "full",
        numeric_features: Iterable[str] | None = None,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ) -> None:
        if torch is None or Data is None or GCNConv is None:
            raise ImportError("qpa_nn requires torch and torch_geometric.")
        self.config = model_config
        self.feature_mode = feature_mode
        if numeric_features is None:
            numeric_features = FULL_NUMERIC_FEATURES if feature_mode == "full" else COARSE_NUMERIC_FEATURES
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features if feature_mode == "full" else ["family"])
        self.feature_columns = [*self.numeric_features, *self.categorical_features]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", _make_one_hot_encoder(), self.categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members_: list[_QPAGraphNet] = []
        self._fitted = False
        self.source_feature_centroid_: np.ndarray | None = None
        self.source_feature_variance_: np.ndarray | None = None
        self.source_feature_rank_: np.ndarray | None = None
        self.source_feature_reference_: np.ndarray | None = None
        self._node_dim = 6
        self._global_dim = 0

    def _member_count(self) -> int:
        return max(1, int(self.config.non_rf_members))

    @staticmethod
    def _family_id(v: object) -> float:
        s = str(v)
        mapping = {"ghz": 0.0, "mirror": 1.0, "qaoa_like": 2.0, "random_clifford": 3.0}
        return mapping.get(s, 4.0)

    def _build_edges(self, n_qubits: int, edge_density: float, circuit_id: str) -> np.ndarray:
        n = max(2, int(n_qubits))
        edges: list[tuple[int, int]] = []
        for i in range(n - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        target_undirected = int(max(n - 1, round(edge_density * n * (n - 1) / 2.0)))
        current_undirected = n - 1
        if target_undirected > current_undirected:
            seed = int(hashlib.md5(str(circuit_id).encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            attempts = 0
            present = {(min(a, b), max(a, b)) for a, b in edges if a < b}
            while current_undirected < target_undirected and attempts < 8 * n * n:
                a = int(rng.integers(0, n))
                b = int(rng.integers(0, n))
                attempts += 1
                if a == b:
                    continue
                u, v = min(a, b), max(a, b)
                if (u, v) in present:
                    continue
                present.add((u, v))
                edges.append((u, v))
                edges.append((v, u))
                current_undirected += 1
        return np.asarray(edges, dtype=np.int64).T

    def _row_to_graph(self, row: pd.Series, global_x: np.ndarray, y: float | None = None) -> Data:
        n_qubits = max(2, int(float(row.get("qubit_count", 2))))
        depth = float(row.get("depth", 0.0))
        num_2q = float(row.get("num_2q_gates", 0.0))
        readout = float(row.get("readout_error", 0.0))
        oneq = float(row.get("oneq_epg", 0.0))
        twoq = float(row.get("twoq_epg", 0.0))
        family = self._family_id(row.get("family", "unknown"))
        gate_load = num_2q / max(1.0, n_qubits)
        node_feats = np.zeros((n_qubits, self._node_dim), dtype=np.float32)
        for i in range(n_qubits):
            pos = i / max(1.0, n_qubits - 1.0)
            node_feats[i, :] = np.asarray([pos, depth, gate_load, readout, oneq, twoq], dtype=np.float32)
        node_feats[:, 2] = node_feats[:, 2] * (1.0 + 0.05 * family)
        edge_index = self._build_edges(
            n_qubits=n_qubits,
            edge_density=float(row.get("edge_density", 0.2)),
            circuit_id=str(row.get("circuit_id", "")),
        )
        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            global_x=torch.tensor(global_x.reshape(1, -1), dtype=torch.float32),
        )
        if y is not None:
            data.y = torch.tensor([float(y)], dtype=torch.float32)
        return data

    def _to_graph_dataset(self, df: pd.DataFrame, global_features: np.ndarray, y: np.ndarray | None = None) -> list[Data]:
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            yy = None if y is None else float(y[i])
            rows.append(self._row_to_graph(row, global_features[i], yy))
        return rows

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
    ) -> "QPANNRegressorFamily":
        X_df = df[self.feature_columns].copy()
        y = df[target_col].to_numpy(dtype=float)
        global_x = self.preprocessor.fit_transform(X_df).astype(np.float32)
        self._global_dim = int(global_x.shape[1])
        self.source_feature_centroid_ = global_x.mean(axis=0)
        self.source_feature_variance_ = global_x.var(axis=0)
        self.source_feature_rank_ = np.argsort(-self.source_feature_variance_)
        self.source_feature_reference_ = global_x.copy()
        dataset = self._to_graph_dataset(df, global_x, y=y)

        self.members_ = []
        n = len(dataset)
        rng = np.random.default_rng(self.config.random_seed)
        for m in range(self._member_count()):
            model = _QPAGraphNet(
                node_dim=self._node_dim,
                global_dim=self._global_dim,
                hidden=tuple(int(v) for v in self.config.qpa_hidden_layers),
            ).to(self.device)
            opt = torch.optim.Adam(
                model.parameters(),
                lr=float(max(self.config.qpa_learning_rate_init, 1e-6)),
                weight_decay=float(max(self.config.qpa_alpha, 0.0)),
            )
            model.train()
            idx = rng.choice(n, size=max(1, int(round(n * float(np.clip(self.config.bootstrap_fraction, 0.1, 1.0))))), replace=True)
            boot = [dataset[int(i)] for i in idx]
            loader = DataLoader(boot, batch_size=min(64, max(8, len(boot))), shuffle=True)
            epochs = int(max(self.config.qpa_max_iter, 50))
            best_epoch_loss = float("inf")
            stale_epochs = 0
            for _ in range(epochs):
                epoch_loss = 0.0
                seen = 0
                for batch in loader:
                    batch = batch.to(self.device)
                    pred = model(batch)
                    loss = F.l1_loss(pred, batch.y.view(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    bs = int(batch.y.numel())
                    epoch_loss += float(loss.detach().cpu()) * bs
                    seen += bs
                if seen > 0:
                    epoch_loss /= float(seen)
                    if best_epoch_loss - epoch_loss > float(self.config.qpa_tol):
                        best_epoch_loss = epoch_loss
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= int(max(self.config.qpa_n_iter_no_change, 1)):
                            break
            self.members_.append(model.eval())

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.preprocessor.transform(df[self.feature_columns].copy())

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        global_x = self.preprocessor.transform(df[self.feature_columns].copy()).astype(np.float32)
        dataset = self._to_graph_dataset(df, global_x, y=None)
        loader = DataLoader(dataset, batch_size=min(128, max(8, len(dataset))), shuffle=False)
        all_member_preds: list[np.ndarray] = []
        for model in self.members_:
            preds: list[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    out = model(batch).detach().cpu().numpy()
                    preds.append(out)
            all_member_preds.append(np.clip(np.concatenate(preds, axis=0), 0.0, 1.0))
        member_predictions = np.vstack(all_member_preds)
        mean = member_predictions.mean(axis=0)
        std = member_predictions.std(axis=0, ddof=1) if member_predictions.shape[0] > 1 else np.zeros_like(mean)
        return PredictionBundle(mean=mean, std=std, member_predictions=member_predictions)

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {
            "model_family": self.model_family,
            "ensemble_members": int(len(self.members_)),
            "device": str(self.device),
        }


class BayesianRidgeRegressorFamily(BaseCapabilityRegressor):
    """Bayesian ridge baseline for tiny-budget uncertainty-focused evaluation."""

    model_family = "bayesian_ridge"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        feature_mode: FeatureMode = "full",
        numeric_features: Iterable[str] | None = None,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ) -> None:
        self.config = model_config
        self.feature_mode = feature_mode
        if numeric_features is None:
            numeric_features = FULL_NUMERIC_FEATURES if feature_mode == "full" else COARSE_NUMERIC_FEATURES
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features if feature_mode == "full" else ["family"])
        self.feature_columns = [*self.numeric_features, *self.categorical_features]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", _make_one_hot_encoder(), self.categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self.model = BayesianRidge(
            alpha_1=float(max(self.config.bayesian_alpha_1, 1e-12)),
            alpha_2=float(max(self.config.bayesian_alpha_2, 1e-12)),
            lambda_1=float(max(self.config.bayesian_lambda_1, 1e-12)),
            lambda_2=float(max(self.config.bayesian_lambda_2, 1e-12)),
        )
        self.source_feature_centroid_: np.ndarray | None = None
        self.source_feature_variance_: np.ndarray | None = None
        self.source_feature_rank_: np.ndarray | None = None
        self.source_feature_reference_: np.ndarray | None = None
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
    ) -> "BayesianRidgeRegressorFamily":
        X = self.preprocessor.fit_transform(df[self.feature_columns].copy())
        y = df[target_col].to_numpy(dtype=float)
        self.source_feature_centroid_ = X.mean(axis=0)
        self.source_feature_variance_ = X.var(axis=0)
        self.source_feature_rank_ = np.argsort(-self.source_feature_variance_)
        self.source_feature_reference_ = X.copy()
        try:
            self.model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.model.fit(X, y)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self.preprocessor.transform(df[self.feature_columns].copy())

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        self._check_is_fitted()
        X = self.transform(df)
        mean, std = self.model.predict(X, return_std=True)
        n_members = max(4, int(self.config.non_rf_members))
        rng = np.random.default_rng(self.config.random_seed + 201)
        member_predictions = np.vstack([rng.normal(loc=mean, scale=np.maximum(std, 1e-6)) for _ in range(n_members)])
        return PredictionBundle(mean=np.asarray(mean, dtype=float), std=np.asarray(std, dtype=float), member_predictions=member_predictions)

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {"model_family": self.model_family, "ensemble_members": int(max(4, self.config.non_rf_members))}


class MonotonicXGBoostRegressorFamily(_BaggedTabularRegressor):
    """XGBoost with monotonic constraints for physics-consistent extrapolation."""

    model_family = "xgboost_monotonic"

    def __init__(self, *, model_config: ModelConfig, feature_mode: FeatureMode = "full") -> None:
        if XGBRegressor is None:
            raise ImportError("xgboost is required for base_estimator='xgboost_monotonic'.")
        numeric_features = FULL_NUMERIC_FEATURES if feature_mode == "full" else COARSE_NUMERIC_FEATURES
        super().__init__(
            model_config=model_config,
            feature_mode=feature_mode,
            numeric_features=numeric_features,
            categorical_features=(),
        )
        self.categorical_features = []
        self.feature_columns = list(self.numeric_features)
        self.preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), self.numeric_features)],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self._constraint_map = {str(name): int(value) for name, value in self.config.xgb_monotone_constraints}

    def _build_member(self, *, random_seed: int) -> object:
        constraint_vector = [int(self._constraint_map.get(name, 0)) for name in self.numeric_features]
        return XGBRegressor(
            n_estimators=max(200, int(self.config.n_members)),
            max_depth=max(2, int(self.config.xgb_monotone_depth)),
            learning_rate=float(max(self.config.xgb_learning_rate, 1e-3)),
            subsample=float(np.clip(self.config.xgb_subsample, 0.3, 1.0)),
            colsample_bytree=float(np.clip(self.config.xgb_colsample_bytree, 0.3, 1.0)),
            reg_lambda=float(max(self.config.xgb_reg_lambda, 0.0)),
            monotone_constraints=tuple(constraint_vector),
            objective="reg:squarederror",
            n_jobs=int(max(self.config.n_jobs, 1)),
            random_state=int(random_seed),
        )

    def diagnostics(self) -> dict[str, float | str | int | None]:
        out = super().diagnostics()
        out["constraint_mode"] = "physics_monotonic"
        return out


class TabNetRegressorFamily(BaseCapabilityRegressor):
    """TabNet deep tabular baseline for neural comparison without strawman MLP claims."""

    model_family = "tabnet"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        feature_mode: FeatureMode = "full",
        numeric_features: Iterable[str] | None = None,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    ) -> None:
        if TabNetRegressor is None:
            raise ImportError("pytorch-tabnet is required for base_estimator='tabnet'.")
        self.config = model_config
        self.feature_mode = feature_mode
        if numeric_features is None:
            numeric_features = FULL_NUMERIC_FEATURES if feature_mode == "full" else COARSE_NUMERIC_FEATURES
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features if feature_mode == "full" else ["family"])
        self.feature_columns = [*self.numeric_features, *self.categorical_features]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", _make_one_hot_encoder(), self.categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self.model = TabNetRegressor(
            n_d=int(max(self.config.tabnet_n_d, 4)),
            n_a=int(max(self.config.tabnet_n_a, 4)),
            n_steps=int(max(self.config.tabnet_n_steps, 2)),
            gamma=float(max(self.config.tabnet_gamma, 1.0)),
            lambda_sparse=float(max(self.config.tabnet_lambda_sparse, 0.0)),
            seed=int(self.config.random_seed),
            verbose=0,
        )
        self.source_feature_centroid_: np.ndarray | None = None
        self.source_feature_variance_: np.ndarray | None = None
        self.source_feature_rank_: np.ndarray | None = None
        self.source_feature_reference_: np.ndarray | None = None
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
    ) -> "TabNetRegressorFamily":
        X = self.preprocessor.fit_transform(df[self.feature_columns].copy()).astype(np.float32)
        y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.source_feature_centroid_ = X.mean(axis=0)
        self.source_feature_variance_ = X.var(axis=0)
        self.source_feature_rank_ = np.argsort(-self.source_feature_variance_)
        self.source_feature_reference_ = X.copy()
        fit_kwargs = {
            "X_train": X,
            "y_train": y,
            "max_epochs": int(max(self.config.tabnet_max_epochs, 20)),
            "patience": 20,
            "batch_size": min(1024, max(64, len(X))),
            "virtual_batch_size": min(256, max(16, len(X) // 2)),
        }
        if sample_weight is not None:
            fit_kwargs["weights"] = sample_weight
        try:
            self.model.fit(**fit_kwargs)
        except TypeError:
            fit_kwargs.pop("weights", None)
            self.model.fit(**fit_kwargs)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self.preprocessor.transform(df[self.feature_columns].copy())

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        self._check_is_fitted()
        X = self.transform(df).astype(np.float32)
        mean = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        source_var = self.source_feature_variance_ if self.source_feature_variance_ is not None else np.array([1e-4], dtype=float)
        proxy_std = float(np.sqrt(np.maximum(np.mean(source_var), 1e-6)))
        std = np.full_like(mean, fill_value=proxy_std, dtype=float)
        n_members = max(4, int(self.config.non_rf_members))
        rng = np.random.default_rng(self.config.random_seed + 401)
        member_predictions = np.vstack([rng.normal(loc=mean, scale=np.maximum(std, 1e-6)) for _ in range(n_members)])
        return PredictionBundle(mean=mean, std=std, member_predictions=member_predictions)

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {"model_family": self.model_family, "ensemble_members": int(max(4, self.config.non_rf_members))}

_SUMMARY_FEATURE_COLUMNS = [
    "qubit_count",
    "depth",
    "two_qubit_density",
    "avg_degree",
    "readout_error",
    "oneq_epg",
    "twoq_epg",
    "time_index",
]


def _source_distance(features: np.ndarray, source_centroid: np.ndarray | None) -> np.ndarray:
    if source_centroid is None:
        raise RuntimeError("Source centroid unavailable. Fit the base model first.")
    return np.linalg.norm(features - source_centroid.reshape(1, -1), axis=1).reshape(-1, 1)


class _ResidualAdapterBase(BaseCapabilityRegressor):
    """Shared design-matrix and calibration logic for residual adapters."""

    def __init__(
        self,
        base_model: BaseCapabilityRegressor,
        *,
        random_seed: int,
        std_temperature_quantile: float,
        std_temperature_min: float,
        std_temperature_max: float,
        std_floor: float,
    ) -> None:
        self.base_model = base_model
        self.random_seed = random_seed
        self.std_temperature_quantile = float(np.clip(std_temperature_quantile, 0.5, 0.99))
        self.std_temperature_min = float(max(std_temperature_min, 0.05))
        self.std_temperature_max = float(max(std_temperature_max, self.std_temperature_min))
        self.std_floor = float(max(std_floor, 1e-8))
        self._std_temperature: float = 1.0
        self._fitted = False

    def _finalize_temperature(self, *, labels: np.ndarray, adapted_mean: np.ndarray, base_std: np.ndarray) -> None:
        abs_error = np.abs(labels - adapted_mean)
        safe_base_std = np.maximum(base_std, self.std_floor)
        ratio = abs_error / safe_base_std
        estimated_temp = float(np.quantile(ratio, self.std_temperature_quantile))
        self._std_temperature = float(np.clip(estimated_temp, self.std_temperature_min, self.std_temperature_max))
        self._fitted = True

    def _prediction_bundle(self, *, base_bundle: PredictionBundle, residual_correction: np.ndarray) -> PredictionBundle:
        mean = np.clip(base_bundle.mean + residual_correction, 0.0, 1.0)
        std = np.maximum(base_bundle.std * self._std_temperature, self.std_floor)
        member_predictions = np.vstack(
            [np.clip(base_bundle.member_predictions[i] + residual_correction, 0.0, 1.0) for i in range(base_bundle.member_predictions.shape[0])]
        )
        return PredictionBundle(mean=mean, std=std, member_predictions=member_predictions)

    def diagnostics(self) -> dict[str, float | str | int | None]:
        family = str(getattr(self.base_model, "model_family", "unknown"))
        return {"std_temperature": self._std_temperature, "model_family": family}

    def _adapter_features(
        self,
        df: pd.DataFrame,
        bundle: PredictionBundle,
        *,
        feature_mode: Literal["full", "summary", "hybrid"],
        use_base_mean: bool,
        use_base_std: bool,
        max_feature_dims: int | None,
    ) -> np.ndarray:
        features = self.base_model.transform(df)
        source_centroid = getattr(self.base_model, "source_feature_centroid_", None)
        source_distance = _source_distance(features, source_centroid)
        columns: list[np.ndarray] = []
        if feature_mode == "summary":
            summary_cols = [df[col].to_numpy(dtype=float).reshape(-1, 1) for col in _SUMMARY_FEATURE_COLUMNS if col in df.columns]
            columns.extend(summary_cols)
        elif feature_mode == "hybrid":
            rank = getattr(self.base_model, "source_feature_rank_", None)
            if rank is None:
                raise RuntimeError("Source feature ranking unavailable. Fit the base model first.")
            if max_feature_dims is None:
                reduced_features = features
            else:
                keep = rank[: min(max_feature_dims, len(rank))]
                reduced_features = features[:, keep]
            columns.append(reduced_features)
        else:
            columns.append(features)
        columns.append(source_distance)
        if use_base_mean:
            columns.append(bundle.mean.reshape(-1, 1))
        if use_base_std:
            columns.append(bundle.std.reshape(-1, 1))
        return np.hstack(columns)

    def _weighted_ridge_closed_form(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        penalty_diag: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(residuals, dtype=float).reshape(-1)
        if sample_weight is None:
            weights = np.ones(X.shape[0], dtype=float)
        else:
            weights = np.clip(np.asarray(sample_weight, dtype=float).reshape(-1), 1e-8, None)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("Sample weights must sum to a positive value.")

        x_mean = np.average(X, axis=0, weights=weights)
        y_mean = float(np.average(y, weights=weights))
        X_centered = X - x_mean.reshape(1, -1)
        y_centered = y - y_mean
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        Xw = X_centered * sqrt_w
        yw = y_centered * sqrt_w.reshape(-1)
        lhs = Xw.T @ Xw + np.diag(np.asarray(penalty_diag, dtype=float).reshape(-1))
        rhs = Xw.T @ yw
        coef = np.linalg.solve(lhs, rhs)
        intercept = float(y_mean - x_mean @ coef)
        return coef, intercept


class FewShotResidualAdapter(_ResidualAdapterBase):
    """Residual adapter fitted on a small labeled target set."""

    def __init__(
        self,
        base_model: BaseCapabilityRegressor,
        *,
        ridge_alpha: float = 1.0,
        residual_feature_mode: Literal["full", "summary", "hybrid"] = "hybrid",
        use_base_mean: bool = True,
        use_base_std: bool = True,
        max_feature_dims: int | None = 64,
        alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        cv_folds: int = 3,
        selection_metric: Literal["mae"] = "mae",
        random_seed: int = 20260308,
        std_temperature_quantile: float = 0.90,
        std_temperature_min: float = 0.60,
        std_temperature_max: float = 1.00,
        std_floor: float = 1e-4,
    ) -> None:
        super().__init__(
            base_model,
            random_seed=random_seed,
            std_temperature_quantile=std_temperature_quantile,
            std_temperature_min=std_temperature_min,
            std_temperature_max=std_temperature_max,
            std_floor=std_floor,
        )
        self.ridge_alpha = ridge_alpha
        self.residual_feature_mode = residual_feature_mode
        self.use_base_mean = bool(use_base_mean)
        self.use_base_std = bool(use_base_std)
        self.max_feature_dims = None if max_feature_dims is None else max(int(max_feature_dims), 1)
        self.alpha_grid = tuple(float(alpha) for alpha in alpha_grid) if alpha_grid else (float(ridge_alpha),)
        self.cv_folds = int(max(cv_folds, 2))
        self.selection_metric = selection_metric
        self.adapter: Ridge | None = None
        self.selected_alpha_: float = float(ridge_alpha)
        self.design_dim_: int = 0

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
        shuffle_labels: bool = False,
    ) -> "FewShotResidualAdapter":
        if df.empty:
            raise ValueError("Adapter cannot be fitted on an empty dataframe.")
        labels = df[target_col].to_numpy(dtype=float)
        if shuffle_labels:
            rng = np.random.default_rng(self.random_seed)
            labels = labels.copy()
            rng.shuffle(labels)
        base_bundle = self.base_model.predict_distribution(df)
        X = self._adapter_design_matrix(df, base_bundle)
        self.design_dim_ = int(X.shape[1])
        residuals = labels - base_bundle.mean
        alpha = self._select_alpha(
            X=X,
            residuals=residuals,
            base_mean=base_bundle.mean,
            sample_weight=sample_weight,
        )
        self.selected_alpha_ = float(alpha)
        self.adapter = Ridge(alpha=self.selected_alpha_, fit_intercept=True)
        self.adapter.fit(X, residuals, sample_weight=sample_weight)
        fitted_residuals = self.adapter.predict(X)
        adapted_mean = np.clip(base_bundle.mean + fitted_residuals, 0.0, 1.0)
        self._finalize_temperature(labels=labels, adapted_mean=adapted_mean, base_std=base_bundle.std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self.base_model._check_is_fitted()
        bundle = self.base_model.predict_distribution(df)
        return self._adapter_design_matrix(df, bundle)

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        if not self._fitted or self.adapter is None:
            raise RuntimeError("Adapter must be fitted before prediction.")
        base_bundle = self.base_model.predict_distribution(df)
        X = self._adapter_design_matrix(df, base_bundle)
        residual_correction = self.adapter.predict(X)
        return self._prediction_bundle(base_bundle=base_bundle, residual_correction=residual_correction)

    def _adapter_design_matrix(self, df: pd.DataFrame, bundle: PredictionBundle) -> np.ndarray:
        return self._adapter_features(
            df,
            bundle,
            feature_mode=self.residual_feature_mode,
            use_base_mean=self.use_base_mean,
            use_base_std=self.use_base_std,
            max_feature_dims=self.max_feature_dims,
        )

    def _select_alpha(
        self,
        *,
        X: np.ndarray,
        residuals: np.ndarray,
        base_mean: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        if X.shape[0] < 6 or len(self.alpha_grid) == 1:
            return float(self.ridge_alpha)
        n_folds = min(self.cv_folds, X.shape[0])
        if n_folds < 2:
            return float(self.ridge_alpha)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        best_alpha = float(self.ridge_alpha)
        best_score = float("inf")
        for alpha in self.alpha_grid:
            fold_scores: list[float] = []
            for train_idx, val_idx in kfold.split(X):
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue
                model = Ridge(alpha=float(alpha), fit_intercept=True)
                sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[train_idx]
                model.fit(X[train_idx], residuals[train_idx], sample_weight=sw)
                pred_resid = model.predict(X[val_idx])
                pred_mean = np.clip(base_mean[val_idx] + pred_resid, 0.0, 1.0)
                true_mean = np.clip(base_mean[val_idx] + residuals[val_idx], 0.0, 1.0)
                fold_scores.append(float(np.mean(np.abs(true_mean - pred_mean))))
            if fold_scores:
                score = float(np.mean(fold_scores))
                if score < best_score:
                    best_score = score
                    best_alpha = float(alpha)
        return best_alpha

    def diagnostics(self) -> dict[str, float | str | int | None]:
        out = super().diagnostics()
        out.update(
            {
                "selected_alpha": self.selected_alpha_,
                "design_dim": self.design_dim_,
                "feature_mode": self.residual_feature_mode,
            }
        )
        return out


class MeanShiftAdapter(_ResidualAdapterBase):
    """Adapt by shifting all predictions by the mean residual on the labeled target subset."""

    def __init__(
        self,
        base_model: BaseCapabilityRegressor,
        *,
        random_seed: int = 20260308,
        std_temperature_quantile: float = 0.90,
        std_temperature_min: float = 0.60,
        std_temperature_max: float = 1.00,
        std_floor: float = 1e-4,
    ) -> None:
        super().__init__(
            base_model,
            random_seed=random_seed,
            std_temperature_quantile=std_temperature_quantile,
            std_temperature_min=std_temperature_min,
            std_temperature_max=std_temperature_max,
            std_floor=std_floor,
        )
        self._shift = 0.0

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
        shuffle_labels: bool = False,
    ) -> "MeanShiftAdapter":
        if df.empty:
            raise ValueError("Adapter cannot be fitted on an empty dataframe.")
        labels = df[target_col].to_numpy(dtype=float)
        if shuffle_labels:
            rng = np.random.default_rng(self.random_seed)
            labels = labels.copy()
            rng.shuffle(labels)
        base_bundle = self.base_model.predict_distribution(df)
        residuals = labels - base_bundle.mean
        if sample_weight is None:
            self._shift = float(residuals.mean())
        else:
            weights = np.asarray(sample_weight, dtype=float)
            self._shift = float(np.average(residuals, weights=weights))
        adapted_mean = np.clip(base_bundle.mean + self._shift, 0.0, 1.0)
        self._finalize_temperature(labels=labels, adapted_mean=adapted_mean, base_std=base_bundle.std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self.base_model._check_is_fitted()
        return self.base_model.transform(df)

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        if not self._fitted:
            raise RuntimeError("Adapter must be fitted before prediction.")
        base_bundle = self.base_model.predict_distribution(df)
        residual_correction = np.full_like(base_bundle.mean, self._shift, dtype=float)
        return self._prediction_bundle(base_bundle=base_bundle, residual_correction=residual_correction)

    def diagnostics(self) -> dict[str, float | str | int | None]:
        out = super().diagnostics()
        out["mean_shift"] = self._shift
        return out


class EWCResidualAdapter(_ResidualAdapterBase):
    """Residual adapter with a diagonal Fisher-style penalty on feature shifts."""

    def __init__(
        self,
        base_model: BaseCapabilityRegressor,
        *,
        ridge_alpha: float = 1.0,
        ewc_lambda: float = 2.0,
        random_seed: int = 20260308,
        std_temperature_quantile: float = 0.90,
        std_temperature_min: float = 0.60,
        std_temperature_max: float = 1.00,
        std_floor: float = 1e-4,
    ) -> None:
        super().__init__(
            base_model,
            random_seed=random_seed,
            std_temperature_quantile=std_temperature_quantile,
            std_temperature_min=std_temperature_min,
            std_temperature_max=std_temperature_max,
            std_floor=std_floor,
        )
        self.ridge_alpha = float(max(ridge_alpha, 1e-8))
        self.ewc_lambda = float(max(ewc_lambda, 0.0))
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.design_dim_: int = 0

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
        shuffle_labels: bool = False,
    ) -> "EWCResidualAdapter":
        if df.empty:
            raise ValueError("Adapter cannot be fitted on an empty dataframe.")
        labels = df[target_col].to_numpy(dtype=float)
        if shuffle_labels:
            rng = np.random.default_rng(self.random_seed)
            labels = labels.copy()
            rng.shuffle(labels)
        base_bundle = self.base_model.predict_distribution(df)
        X = self._adapter_design_matrix(df, base_bundle)
        self.design_dim_ = int(X.shape[1])
        residuals = labels - base_bundle.mean
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
        self.coef_, self.intercept_ = self._solve_diagonal_penalty_ridge(X, residuals, sample_weight=weights)
        fitted_residuals = X @ self.coef_ + self.intercept_
        adapted_mean = np.clip(base_bundle.mean + fitted_residuals, 0.0, 1.0)
        self._finalize_temperature(labels=labels, adapted_mean=adapted_mean, base_std=base_bundle.std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self.base_model._check_is_fitted()
        bundle = self.base_model.predict_distribution(df)
        return self._adapter_design_matrix(df, bundle)

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        if not self._fitted or self.coef_ is None:
            raise RuntimeError("Adapter must be fitted before prediction.")
        base_bundle = self.base_model.predict_distribution(df)
        X = self._adapter_design_matrix(df, base_bundle)
        residual_correction = X @ self.coef_ + self.intercept_
        return self._prediction_bundle(base_bundle=base_bundle, residual_correction=residual_correction)

    def _adapter_design_matrix(self, df: pd.DataFrame, bundle: PredictionBundle) -> np.ndarray:
        features = self.base_model.transform(df)
        distance_col = _source_distance(features, getattr(self.base_model, "source_feature_centroid_", None))
        mean_col = bundle.mean.reshape(-1, 1)
        std_col = bundle.std.reshape(-1, 1)
        return np.hstack([features, distance_col, mean_col, std_col])

    def _solve_diagonal_penalty_ridge(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        X = np.asarray(X, dtype=float)
        if sample_weight is None:
            weights = np.ones(X.shape[0], dtype=float)
        else:
            weights = np.clip(np.asarray(sample_weight, dtype=float).reshape(-1), 1e-8, None)
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        X_centered = X - np.average(X, axis=0, weights=weights).reshape(1, -1)
        Xw = X_centered * sqrt_w
        fisher_diag = np.mean(np.square(Xw), axis=0)
        penalty_diag = self.ridge_alpha + self.ewc_lambda * fisher_diag
        return self._weighted_ridge_closed_form(
            X,
            residuals,
            sample_weight=weights,
            penalty_diag=penalty_diag,
        )

    def diagnostics(self) -> dict[str, float | str | int | None]:
        out = super().diagnostics()
        out.update({"ewc_lambda": self.ewc_lambda, "design_dim": self.design_dim_})
        return out


class ElasticResidualAdapter(_ResidualAdapterBase):
    """Residual adapter with CV-selected isotropic and Fisher-weighted regularization."""

    def __init__(
        self,
        base_model: BaseCapabilityRegressor,
        *,
        ridge_alpha: float = 1.0,
        ewc_lambda: float = 0.0,
        feature_mode: Literal["full", "summary", "hybrid"] = "hybrid",
        use_base_mean: bool = True,
        use_base_std: bool = True,
        max_feature_dims: int | None = 64,
        alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        ewc_lambda_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
        cv_folds: int = 3,
        selection_metric: Literal["mae"] = "mae",
        random_seed: int = 20260308,
        std_temperature_quantile: float = 0.90,
        std_temperature_min: float = 0.60,
        std_temperature_max: float = 1.00,
        std_floor: float = 1e-4,
    ) -> None:
        super().__init__(
            base_model,
            random_seed=random_seed,
            std_temperature_quantile=std_temperature_quantile,
            std_temperature_min=std_temperature_min,
            std_temperature_max=std_temperature_max,
            std_floor=std_floor,
        )
        self.ridge_alpha = float(max(ridge_alpha, 1e-8))
        self.ewc_lambda = float(max(ewc_lambda, 0.0))
        self.feature_mode = feature_mode
        self.use_base_mean = bool(use_base_mean)
        self.use_base_std = bool(use_base_std)
        self.max_feature_dims = None if max_feature_dims is None else max(int(max_feature_dims), 1)
        self.alpha_grid = tuple(float(alpha) for alpha in alpha_grid) if alpha_grid else (self.ridge_alpha,)
        self.ewc_lambda_grid = tuple(float(value) for value in ewc_lambda_grid) if ewc_lambda_grid else (self.ewc_lambda,)
        self.cv_folds = int(max(cv_folds, 2))
        self.selection_metric = selection_metric
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.design_dim_: int = 0
        self.selected_alpha_: float = self.ridge_alpha
        self.selected_ewc_lambda_: float = self.ewc_lambda

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "error_rate",
        sample_weight: np.ndarray | None = None,
        shuffle_labels: bool = False,
    ) -> "ElasticResidualAdapter":
        if df.empty:
            raise ValueError("Adapter cannot be fitted on an empty dataframe.")
        labels = df[target_col].to_numpy(dtype=float)
        if shuffle_labels:
            rng = np.random.default_rng(self.random_seed)
            labels = labels.copy()
            rng.shuffle(labels)
        base_bundle = self.base_model.predict_distribution(df)
        X = self._adapter_design_matrix(df, base_bundle)
        residuals = labels - base_bundle.mean
        self.design_dim_ = int(X.shape[1])
        alpha, ewc_lambda = self._select_hyperparameters(
            X=X,
            residuals=residuals,
            base_mean=base_bundle.mean,
            sample_weight=sample_weight,
        )
        self.selected_alpha_ = float(alpha)
        self.selected_ewc_lambda_ = float(ewc_lambda)
        self.coef_, self.intercept_ = self._solve_diagonal_penalty_ridge(
            X,
            residuals,
            sample_weight=sample_weight,
            alpha=self.selected_alpha_,
            ewc_lambda=self.selected_ewc_lambda_,
        )
        fitted_residuals = X @ self.coef_ + self.intercept_
        adapted_mean = np.clip(base_bundle.mean + fitted_residuals, 0.0, 1.0)
        self._finalize_temperature(labels=labels, adapted_mean=adapted_mean, base_std=base_bundle.std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self.base_model._check_is_fitted()
        bundle = self.base_model.predict_distribution(df)
        return self._adapter_design_matrix(df, bundle)

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        if not self._fitted or self.coef_ is None:
            raise RuntimeError("Adapter must be fitted before prediction.")
        base_bundle = self.base_model.predict_distribution(df)
        X = self._adapter_design_matrix(df, base_bundle)
        residual_correction = X @ self.coef_ + self.intercept_
        return self._prediction_bundle(base_bundle=base_bundle, residual_correction=residual_correction)

    def _adapter_design_matrix(self, df: pd.DataFrame, bundle: PredictionBundle) -> np.ndarray:
        return self._adapter_features(
            df,
            bundle,
            feature_mode=self.feature_mode,
            use_base_mean=self.use_base_mean,
            use_base_std=self.use_base_std,
            max_feature_dims=self.max_feature_dims,
        )

    def _solve_diagonal_penalty_ridge(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        alpha: float,
        ewc_lambda: float,
    ) -> tuple[np.ndarray, float]:
        X = np.asarray(X, dtype=float)
        if sample_weight is None:
            weights = np.ones(X.shape[0], dtype=float)
        else:
            weights = np.clip(np.asarray(sample_weight, dtype=float).reshape(-1), 1e-8, None)
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        X_centered = X - np.average(X, axis=0, weights=weights).reshape(1, -1)
        Xw = X_centered * sqrt_w
        fisher_diag = np.mean(np.square(Xw), axis=0)
        penalty_diag = float(alpha) + float(ewc_lambda) * fisher_diag
        return self._weighted_ridge_closed_form(
            X,
            residuals,
            sample_weight=weights,
            penalty_diag=penalty_diag,
        )

    def _select_hyperparameters(
        self,
        *,
        X: np.ndarray,
        residuals: np.ndarray,
        base_mean: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[float, float]:
        if X.shape[0] < 6 or (len(self.alpha_grid) == 1 and len(self.ewc_lambda_grid) == 1):
            return float(self.ridge_alpha), float(self.ewc_lambda)
        n_folds = min(self.cv_folds, X.shape[0])
        if n_folds < 2:
            return float(self.ridge_alpha), float(self.ewc_lambda)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        best_pair = (float(self.ridge_alpha), float(self.ewc_lambda))
        best_score = float("inf")
        for alpha in self.alpha_grid:
            for ewc_lambda in self.ewc_lambda_grid:
                fold_scores: list[float] = []
                for train_idx, val_idx in kfold.split(X):
                    if len(train_idx) == 0 or len(val_idx) == 0:
                        continue
                    fold_weights = None if weights is None else weights[train_idx]
                    coef, intercept = self._solve_diagonal_penalty_ridge(
                        X[train_idx],
                        residuals[train_idx],
                        sample_weight=fold_weights,
                        alpha=float(alpha),
                        ewc_lambda=float(ewc_lambda),
                    )
                    pred_resid = X[val_idx] @ coef + intercept
                    pred_mean = np.clip(base_mean[val_idx] + pred_resid, 0.0, 1.0)
                    true_mean = np.clip(base_mean[val_idx] + residuals[val_idx], 0.0, 1.0)
                    fold_scores.append(float(np.mean(np.abs(true_mean - pred_mean))))
                if fold_scores:
                    score = float(np.mean(fold_scores))
                    if score < best_score:
                        best_score = score
                        best_pair = (float(alpha), float(ewc_lambda))
        return best_pair

    def diagnostics(self) -> dict[str, float | str | int | None]:
        out = super().diagnostics()
        out.update(
            {
                "selected_alpha": self.selected_alpha_,
                "selected_ewc_lambda": self.selected_ewc_lambda_,
                "ewc_lambda": self.selected_ewc_lambda_,
                "design_dim": self.design_dim_,
                "feature_mode": self.feature_mode,
            }
        )
        return out



def make_base_model(
    *,
    model_config: ModelConfig,
    feature_mode: FeatureMode = "full",
    random_seed: int | None = None,
    force_estimator: Literal["rf", "xgboost", "lightgbm", "mlp", "bayesian_ridge", "xgboost_monotonic", "tabnet", "qpa_nn"] | None = None,
) -> BaseCapabilityRegressor:
    estimator = force_estimator or model_config.base_estimator
    cfg = model_config if random_seed is None else replace(model_config, random_seed=int(random_seed))
    if estimator == "rf":
        return TreeEnsembleRegressor(model_config=cfg, feature_mode=feature_mode)
    if estimator == "xgboost":
        return XGBoostRegressorFamily(model_config=cfg, feature_mode=feature_mode)
    if estimator == "lightgbm":
        return LightGBMRegressorFamily(model_config=cfg, feature_mode=feature_mode)
    if estimator == "mlp":
        return MLPRegressorFamily(model_config=cfg, feature_mode=feature_mode)
    if estimator == "qpa_nn":
        return QPANNRegressorFamily(model_config=cfg, feature_mode=feature_mode)
    if estimator == "bayesian_ridge":
        return BayesianRidgeRegressorFamily(model_config=cfg, feature_mode=feature_mode)
    if estimator == "xgboost_monotonic":
        return MonotonicXGBoostRegressorFamily(model_config=cfg, feature_mode=feature_mode)
    if estimator == "tabnet":
        return TabNetRegressorFamily(model_config=cfg, feature_mode=feature_mode)
    raise ValueError(f"Unknown base_estimator: {estimator}")


def fit_target_only_model(
    target_df: pd.DataFrame,
    *,
    model_config: ModelConfig,
    random_seed: int,
    base_estimator: Literal["rf", "xgboost", "lightgbm", "mlp", "bayesian_ridge", "xgboost_monotonic", "tabnet", "qpa_nn"] | None = None,
) -> BaseCapabilityRegressor:
    model = make_base_model(
        model_config=model_config,
        feature_mode="full",
        random_seed=random_seed,
        force_estimator=base_estimator,
    )
    return model.fit(target_df)



def fit_pooled_retrain_model(
    source_train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    model_config: ModelConfig,
    target_upweight: float,
    random_seed: int,
    base_estimator: Literal["rf", "xgboost", "lightgbm", "mlp", "bayesian_ridge", "xgboost_monotonic", "tabnet", "qpa_nn"] | None = None,
) -> BaseCapabilityRegressor:
    combined = pd.concat([source_train_df, target_df], ignore_index=True)
    sample_weight = np.concatenate(
        [
            np.ones(len(source_train_df), dtype=float),
            np.full(len(target_df), float(target_upweight), dtype=float),
        ]
    )
    model = make_base_model(
        model_config=model_config,
        feature_mode="full",
        random_seed=random_seed,
        force_estimator=base_estimator,
    )
    return model.fit(combined, sample_weight=sample_weight)



def attach_pass_probability(bundle: PredictionBundle, *, threshold: float) -> PredictionBundle:
    pass_probability = (bundle.member_predictions <= threshold).mean(axis=0)
    bundle.pass_probability_raw = pass_probability
    return bundle
