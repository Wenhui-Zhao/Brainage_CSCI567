#!/usr/bin/env python3
"""
Workflow:
  1. Load the ADNI T1/ASL feature CSV.
  2. Apply the six fixed final models:
       - T1 ElasticNet
       - T1 RandomForestRegressor
       - ASL ElasticNet
       - ASL RandomForestRegressor
       - T1+ASL ElasticNet
       - T1+ASL RandomForestRegressor
  3. Split the normative pool 80/20 by RID.
  4. Fit models on the 80% normative training split.
  5. Recompute train, GroupKFold out-of-fold CV, and held-out test metrics.
  6. Predict age, compute PAD = predicted age - chronological age, and fit
     PAD ~ diagnosis + age + sex with CN as the reference group.
  7. Generate only the figures and CSV tables used in the final report.

Expected input columns:
  rid, age_scan, normative_pool_primary, dx_model or dx, sex,
  t1vol_st*, aslcbf_st*, and icv.

Report outputs:
  outdir/
    figures/
      predicted_vs_chronological_all_models.png
      pad_boxplots_age_sex_adjusted_all_models.png
      top_regions_feature_T1.png
      top_regions_feature_ASL.png
      top_regions_feature_Fusion.png
    tables/
      model_performance.csv
      pad_contrasts_age_sex_adjusted.csv
      pad_group_summary.csv
      top_regions_by_feature_set.csv

"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection


# -----------------------------------------------------------------------------
# Fixed project design and final locked model settings
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
HEALTHY_POOL_COLUMN = "normative_pool_primary"
GROUP_COLUMN = "rid"
TARGET_COLUMN = "age_scan"
SEX_COLUMN = "sex"

MODEL_ORDER = [
    "T1_Linear_ElasticNet",
    "T1_Nonlinear_RandomForest",
    "ASL_Linear_ElasticNet",
    "ASL_Nonlinear_RandomForest",
    "Fusion_Linear_ElasticNet",
    "Fusion_Nonlinear_RandomForest",
]

GROUP_ORDER = ["CN", "MCI", "AD"]
FEATURE_LABELS = {"t1": "T1", "asl": "ASL", "fusion": "Fusion"}

# Final locked parameters from the selected-best run. Do not edit unless the
# final selected model settings change.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "T1_Linear_ElasticNet": {
        "feature_set": "t1",
        "algorithm": "elasticnet",
        "params": {
            "alpha": 0.5,
            "l1_ratio": 0.5,
            "max_iter": 50000,
            "selection": "cyclic",
            "tol": 1e-4,
        },
    },
    "T1_Nonlinear_RandomForest": {
        "feature_set": "t1",
        "algorithm": "randomforest",
        "params": {
            "bootstrap": True,
            "max_depth": None,
            "max_features": 0.2,
            "max_samples": 0.6,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "n_estimators": 120,
            "n_jobs": 1,
        },
    },
    "ASL_Linear_ElasticNet": {
        "feature_set": "asl",
        "algorithm": "elasticnet",
        "params": {
            "alpha": 0.5,
            "l1_ratio": 0.25,
            "max_iter": 50000,
            "selection": "cyclic",
            "tol": 1e-4,
        },
    },
    "ASL_Nonlinear_RandomForest": {
        "feature_set": "asl",
        "algorithm": "randomforest",
        "params": {
            "bootstrap": True,
            "max_depth": 5,
            "max_features": 0.5,
            "max_samples": None,
            "min_samples_leaf": 1,
            "min_samples_split": 8,
            "n_estimators": 200,
            "n_jobs": 1,
        },
    },
    "Fusion_Linear_ElasticNet": {
        "feature_set": "fusion",
        "algorithm": "elasticnet",
        "params": {
            "alpha": 0.5,
            "l1_ratio": 0.25,
            "max_iter": 50000,
            "selection": "cyclic",
            "tol": 1e-4,
        },
    },
    "Fusion_Nonlinear_RandomForest": {
        "feature_set": "fusion",
        "algorithm": "randomforest",
        "params": {
            "bootstrap": True,
            "max_depth": None,
            "max_features": 0.35,
            "max_samples": 0.8,
            "min_samples_leaf": 1,
            "min_samples_split": 8,
            "n_estimators": 200,
            "n_jobs": 1,
        },
    },
}


# -----------------------------------------------------------------------------
# Basic data utilities
# -----------------------------------------------------------------------------
def coerce_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False, "T": True, "F": False})
        .fillna(False)
    )


def map_group3(x: Any) -> Any:
    if pd.isna(x):
        return np.nan
    text = str(x).strip()
    if text == "CN":
        return "CN"
    if text in {"MCI", "LMCI", "EMCI"}:
        return "MCI"
    if text in {"AD", "Dementia"}:
        return "AD"
    return np.nan


def clean_sex_value(x: Any) -> Any:
    if pd.isna(x):
        return np.nan
    text = str(x).strip().upper()
    if text in {"M", "MALE", "1"}:
        return "Male"
    if text in {"F", "FEMALE", "0"}:
        return "Female"
    return str(x).strip()


def sorted_st_feature_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]

    def st_num(col: str) -> int:
        match = re.search(r"st(\d+)", col, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 10**9

    return sorted(cols, key=st_num)


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    t1_cols = sorted_st_feature_cols(df, "t1vol_st")
    asl_cols = sorted_st_feature_cols(df, "aslcbf_st")
    icv_cols = ["icv"] if "icv" in df.columns else []

    if feature_set == "t1":
        cols = t1_cols + icv_cols
    elif feature_set == "asl":
        cols = asl_cols + icv_cols
    elif feature_set == "fusion":
        cols = t1_cols + asl_cols + icv_cols
    else:
        raise ValueError(f"Unknown feature_set={feature_set!r}")

    if not cols:
        raise ValueError(f"No feature columns found for feature_set={feature_set!r}")
    return cols


def first_scan_per_rid(df: pd.DataFrame, group_col: str, age_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_age_sort"] = pd.to_numeric(out[age_col], errors="coerce") if age_col in out.columns else np.nan
    if "t1_rundate" in out.columns:
        out["_date_sort"] = pd.to_datetime(out["t1_rundate"], errors="coerce")
        sort_cols = [group_col, "_age_sort", "_date_sort"]
    else:
        sort_cols = [group_col, "_age_sort"]
    out = out.sort_values(sort_cols, na_position="last")
    out = out.drop_duplicates(group_col, keep="first")
    return out.drop(columns=["_age_sort", "_date_sort"], errors="ignore")


def make_train_test_split(
    healthy_df: pd.DataFrame,
    group_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_rids = healthy_df[group_col].dropna().drop_duplicates().to_numpy()
    train_rids, test_rids = train_test_split(
        unique_rids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = healthy_df[healthy_df[group_col].isin(train_rids)].copy().reset_index(drop=True)
    test_df = healthy_df[healthy_df[group_col].isin(test_rids)].copy().reset_index(drop=True)
    return train_df, test_df


# -----------------------------------------------------------------------------
# Metrics and model fitting
# -----------------------------------------------------------------------------
def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def safe_r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    if len(y) < 2:
        return float("nan")
    return float(r2_score(y, p))


def eval_predictions(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    return {
        "r2": safe_r2(y, p),
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "n_rows": int(len(y)),
    }


def make_model(algorithm: str, params: Dict[str, Any], random_state: int) -> Pipeline:
    if algorithm == "elasticnet":
        estimator = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=int(params.get("max_iter", 50000)),
            tol=float(params.get("tol", 1e-4)),
            selection=str(params.get("selection", "cyclic")),
            random_state=random_state,
        )
    elif algorithm == "randomforest":
        max_depth = params.get("max_depth", None)
        max_samples = params.get("max_samples", None)
        estimator = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=None if max_depth is None or pd.isna(max_depth) else int(max_depth),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=bool(params.get("bootstrap", True)),
            max_samples=None if max_samples is None or pd.isna(max_samples) else max_samples,
            random_state=random_state,
            n_jobs=int(params.get("n_jobs", 1)),
        )
    else:
        raise ValueError(f"Unknown algorithm={algorithm!r}")

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("zscore", StandardScaler()),
            ("model", estimator),
        ]
    )


def cross_validate_oof(
    model_name: str,
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    target_col: str,
    cv_folds: int,
    random_state: int,
) -> Dict[str, float]:
    x = train_df[feature_cols]
    y = train_df[target_col].astype(float).to_numpy()
    groups = train_df[group_col].to_numpy()
    n_splits = min(cv_folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError(f"Need at least two unique groups for {model_name} cross-validation.")

    cv = GroupKFold(n_splits=n_splits)
    oof = np.full(len(train_df), np.nan, dtype=float)
    fold_rows: List[Dict[str, float]] = []

    for train_idx, val_idx in cv.split(x, y, groups=groups):
        model = make_model(config["algorithm"], config["params"], random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x.iloc[train_idx], y[train_idx])
        pred_train = np.asarray(model.predict(x.iloc[train_idx]), dtype=float)
        pred_val = np.asarray(model.predict(x.iloc[val_idx]), dtype=float)
        oof[val_idx] = pred_val
        train_perf = eval_predictions(y[train_idx], pred_train)
        val_perf = eval_predictions(y[val_idx], pred_val)
        fold_rows.append(
            {
                "train_r2": train_perf["r2"],
                "val_r2": val_perf["r2"],
            }
        )

    valid = ~np.isnan(oof)
    oof_perf = eval_predictions(y[valid], oof[valid])
    fold_df = pd.DataFrame(fold_rows)
    return {
        "cv_oof_r2": oof_perf["r2"],
        "cv_oof_rmse": oof_perf["rmse"],
        "cv_oof_mae": oof_perf["mae"],
        "cv_gap_r2": float(fold_df["train_r2"].mean() - fold_df["val_r2"].mean()),
    }


# -----------------------------------------------------------------------------
# Region mapping and model explanation
# -----------------------------------------------------------------------------
def load_region_mapping(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["st_num", "region_name", "hemisphere", "tissue_class"])
    if not path.exists():
        raise FileNotFoundError(f"Region mapping file not found: {path}")

    mapping = pd.read_csv(path)
    lower_to_original = {c.lower(): c for c in mapping.columns}

    if "st_num" in lower_to_original:
        mapping = mapping.rename(columns={lower_to_original["st_num"]: "st_num"})
    else:
        candidate = None
        for c in mapping.columns:
            if c.lower() in {"st", "stnum", "st_number", "roi", "roi_num"}:
                candidate = c
                break
        if candidate is None:
            raise ValueError("Region mapping must contain st_num or an equivalent ST-number column.")
        mapping = mapping.rename(columns={candidate: "st_num"})

    if "region_name" in lower_to_original:
        mapping = mapping.rename(columns={lower_to_original["region_name"]: "region_name"})
    else:
        candidate = None
        for c in mapping.columns:
            if c.lower() in {"region", "name", "label", "roi_name", "structure"}:
                candidate = c
                break
        if candidate is None:
            mapping["region_name"] = mapping["st_num"].apply(lambda x: f"ST{x}")
        else:
            mapping = mapping.rename(columns={candidate: "region_name"})

    for c in ["hemisphere", "tissue_class"]:
        if c not in mapping.columns:
            mapping[c] = ""

    mapping["st_num"] = pd.to_numeric(mapping["st_num"], errors="coerce").astype("Int64")
    return mapping[["st_num", "region_name", "hemisphere", "tissue_class"]].drop_duplicates("st_num")


def parse_feature_metadata(feature: str, region_map: pd.DataFrame) -> Dict[str, Any]:
    if feature == "icv":
        return {
            "feature": feature,
            "st_num": np.nan,
            "region_name": "Intracranial volume",
            "hemisphere": "Global",
            "tissue_class": "Global",
            "feature_modality": "ICV",
            "region_label": "ICV",
        }

    match = re.search(r"st(\d+)", feature, flags=re.IGNORECASE)
    st_num = int(match.group(1)) if match else np.nan
    if feature.startswith("t1vol_"):
        modality = "T1 volume"
    elif feature.startswith("aslcbf_"):
        modality = "ASL CBF"
    else:
        modality = "Other"

    region_name = f"ST{st_num}" if not pd.isna(st_num) else feature
    hemisphere = ""
    tissue_class = ""
    if not pd.isna(st_num) and len(region_map):
        row = region_map[region_map["st_num"].astype("Int64") == int(st_num)]
        if len(row):
            region_name = str(row.iloc[0]["region_name"])
            hemisphere = str(row.iloc[0].get("hemisphere", ""))
            tissue_class = str(row.iloc[0].get("tissue_class", ""))

    return {
        "feature": feature,
        "st_num": st_num,
        "region_name": region_name,
        "hemisphere": hemisphere,
        "tissue_class": tissue_class,
        "feature_modality": modality,
        "region_label": f"{modality}: {region_name}",
    }


def model_importance_table(
    model_name: str,
    model: Pipeline,
    feature_cols: List[str],
    config: Dict[str, Any],
    region_map: pd.DataFrame,
) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        importance = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        importance_type = "standardized ElasticNet coefficient"
        importance_abs = np.abs(importance)
    elif hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
        importance_type = "Random Forest impurity importance"
        importance_abs = importance
    else:
        importance = np.full(len(feature_cols), np.nan)
        importance_type = "unknown"
        importance_abs = np.full(len(feature_cols), np.nan)

    rows: List[Dict[str, Any]] = []
    for i, feat in enumerate(feature_cols):
        rows.append(
            {
                "model": model_name,
                "feature_set": config["feature_set"],
                "algorithm": config["algorithm"],
                "importance_type": importance_type,
                "importance": float(importance[i]) if i < len(importance) else np.nan,
                "importance_abs": float(importance_abs[i]) if i < len(importance_abs) else np.nan,
                **parse_feature_metadata(feat, region_map),
            }
        )

    out = pd.DataFrame(rows).sort_values("importance_abs", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# -----------------------------------------------------------------------------
# PAD analysis
# -----------------------------------------------------------------------------
def p_text(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def residualize_pad_for_age_sex(df: pd.DataFrame, pad_col: str, age_col: str, sex_col: str) -> pd.Series:
    tmp = df[[pad_col, age_col, sex_col]].rename(columns={pad_col: "pad", age_col: "age", sex_col: "sex_clean"}).copy()
    tmp = tmp.dropna(subset=["pad", "age"])
    if len(tmp) < 5:
        return pd.Series(np.nan, index=df.index)

    if tmp["sex_clean"].notna().sum() >= 5 and tmp["sex_clean"].nunique(dropna=True) >= 2:
        tmp = tmp.dropna(subset=["sex_clean"])
        formula = "pad ~ age + C(sex_clean)"
    else:
        formula = "pad ~ age"

    try:
        fit = smf.ols(formula, data=tmp).fit()
        adjusted = fit.resid + tmp["pad"].mean()
        out = pd.Series(np.nan, index=df.index)
        out.loc[tmp.index] = adjusted
        return out
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(f"PAD residualization failed for {pad_col}: {exc}")
        return df[pad_col]


def run_pad_regression(
    df: pd.DataFrame,
    model_name: str,
    pad_col: str,
    age_col: str,
    sex_col: str,
) -> pd.DataFrame:
    tmp = df[[pad_col, age_col, sex_col, "group3"]].rename(
        columns={pad_col: "pad", age_col: "age", sex_col: "sex_clean"}
    ).copy()
    tmp = tmp[tmp["group3"].isin(GROUP_ORDER)].dropna(subset=["pad", "age", "group3"])
    if len(tmp) < 10 or "CN" not in set(tmp["group3"]):
        return pd.DataFrame()

    if tmp["sex_clean"].notna().sum() >= 5 and tmp["sex_clean"].nunique(dropna=True) >= 2:
        tmp = tmp.dropna(subset=["sex_clean"])
        formula = 'pad ~ C(group3, Treatment(reference="CN")) + age + C(sex_clean)'
    else:
        formula = 'pad ~ C(group3, Treatment(reference="CN")) + age'

    fit = smf.ols(formula, data=tmp).fit()
    rows: List[Dict[str, Any]] = []
    for group in ["MCI", "AD"]:
        param_name = None
        for name in fit.params.index:
            if "group3" in name and f"[T.{group}]" in name:
                param_name = name
                break
        rows.append(
            {
                "model": model_name,
                "contrast": f"{group} vs CN",
                "delta_pad": float(fit.params[param_name]) if param_name else np.nan,
                "t_value": float(fit.tvalues[param_name]) if param_name else np.nan,
                "p_value": float(fit.pvalues[param_name]) if param_name else np.nan,
                "n": int(len(tmp)),
                "formula": formula,
            }
        )
    return pd.DataFrame(rows)


def pad_group_summary(df: pd.DataFrame, model_name: str, pad_col: str, pad_adjusted_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group in GROUP_ORDER:
        sub = df[df["group3"] == group]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "group": group,
                "n": int(len(sub)),
                "raw_pad_mean": float(sub[pad_col].mean()),
                "raw_pad_median": float(sub[pad_col].median()),
                "raw_pad_sd": float(sub[pad_col].std(ddof=1)),
                "age_sex_adjusted_pad_mean": float(sub[pad_adjusted_col].mean()),
                "age_sex_adjusted_pad_median": float(sub[pad_adjusted_col].median()),
                "age_sex_adjusted_pad_sd": float(sub[pad_adjusted_col].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_predicted_vs_chronological(
    plot_df: pd.DataFrame,
    model_names: List[str],
    performance_df: pd.DataFrame,
    target_col: str,
    figdir: Path,
) -> None:
    pred_cols = [f"pred_age__{m}" for m in model_names]
    ages = pd.to_numeric(plot_df[target_col], errors="coerce")
    preds = pd.to_numeric(plot_df[pred_cols].stack(), errors="coerce") if pred_cols else pd.Series(dtype=float)
    lo = float(np.nanmin([ages.min(), preds.min()]))
    hi = float(np.nanmax([ages.max(), preds.max()]))
    padding = (hi - lo) * 0.05 if hi > lo else 1.0
    lo, hi = lo - padding, hi + padding

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, model_name in zip(axes.ravel(), model_names):
        pred_col = f"pred_age__{model_name}"
        sub = plot_df[[target_col, pred_col, "group3"]].dropna().copy()
        for group in GROUP_ORDER:
            g = sub[sub["group3"] == group]
            if len(g):
                ax.scatter(g[target_col], g[pred_col], s=13, alpha=0.75, label=group)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        perf = performance_df[performance_df["model"] == model_name]
        if len(perf):
            title = (
                f"{model_name.replace('_', ' ')}\n"
                f"test $R^2$={perf.iloc[0]['test_r2']:.3f}, RMSE={perf.iloc[0]['test_rmse']:.2f}"
            )
        else:
            title = model_name.replace("_", " ")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    for ax in axes[-1, :]:
        ax.set_xlabel("Chronological age")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted age")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(figdir / "predicted_vs_chronological_all_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pad_boxplots(
    plot_df: pd.DataFrame,
    model_names: List[str],
    pad_regression_df: pd.DataFrame,
    figdir: Path,
) -> None:
    rng = np.random.default_rng(42)

    def annotate(ax: plt.Axes, model_name: str) -> None:
        rows = pad_regression_df[pad_regression_df["model"] == model_name]
        text_lines = []
        for contrast in ["MCI vs CN", "AD vs CN"]:
            row = rows[rows["contrast"] == contrast]
            if len(row):
                r = row.iloc[0]
                text_lines.append(
                    f"{contrast}: ΔPAD={r['delta_pad']:.2f}, t={r['t_value']:.2f}, p={p_text(r['p_value'])}"
                )
        if text_lines:
            ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes, va="top", ha="left", fontsize=7)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharey=False)
    for ax, model_name in zip(axes.ravel(), model_names):
        col = f"pad_age_sex_adjusted__{model_name}"
        data = [plot_df.loc[plot_df["group3"] == g, col].dropna().to_numpy() for g in GROUP_ORDER]
        ax.boxplot(data, labels=GROUP_ORDER, showfliers=False)
        for i, vals in enumerate(data, start=1):
            if len(vals):
                ax.scatter(rng.normal(loc=i, scale=0.035, size=len(vals)), vals, s=8, alpha=0.45)
        ax.axhline(0, linewidth=0.8, linestyle="--")
        ax.set_title(model_name.replace("_", " "), fontsize=9)
        ax.set_ylabel("Age/sex-adjusted PAD")
        annotate(ax, model_name)

    fig.tight_layout()
    fig.savefig(figdir / "pad_boxplots_age_sex_adjusted_all_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_regions(top_df: pd.DataFrame, title: str, outpath: Path, n_top: int) -> None:
    sub = top_df.head(n_top).copy().iloc[::-1]
    if sub.empty:
        return
    labels = sub["region_label"].astype(str).tolist()
    values = sub["importance"].astype(float).to_numpy()
    algorithm = str(sub["algorithm"].iloc[0])

    fig_height = max(3.2, 0.32 * len(sub) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    ax.barh(np.arange(len(sub)), values)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels, fontsize=8)
    if algorithm == "elasticnet":
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel("Standardized coefficient")
    else:
        ax.set_xlabel("Feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply embedded locked six ADNI brain-age models and generate report-ready outputs."
    )
    parser.add_argument("--data", default="t1vol_asl_dataset.csv", help="Input dataset CSV")
    parser.add_argument("--region-map", default="adni_st_region_mapping (2).csv", help="CSV mapping ST regions to names")
    parser.add_argument("--outdir", default="report_outputs", help="Output directory")
    parser.add_argument("--healthy-col", default=HEALTHY_POOL_COLUMN)
    parser.add_argument("--group-col", default=GROUP_COLUMN)
    parser.add_argument("--target-col", default=TARGET_COLUMN)
    parser.add_argument("--sex-col", default=SEX_COLUMN)
    parser.add_argument("--dx-col", default="auto", help="Diagnosis column; auto uses dx_model, then dx")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--analysis-level", choices=["first_scan_per_rid", "all_samples"], default="first_scan_per_rid")
    parser.add_argument("--top-n-regions", type=int, default=12)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, na_values=["NA", "", "NaN", "nan"])
    for required in [args.group_col, args.target_col, args.healthy_col]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")
    if args.sex_col not in df.columns:
        warnings.warn(f"Sex column {args.sex_col!r} not found; PAD models will adjust for age only.")
        df[args.sex_col] = np.nan

    for col in ["normative_pool_primary", "normative_pool_cn_only", "is_screening_visit", args.healthy_col]:
        if col in df.columns:
            df[col] = coerce_bool(df[col])
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")

    if args.dx_col == "auto":
        dx_col = "dx_model" if "dx_model" in df.columns else ("dx" if "dx" in df.columns else None)
    else:
        dx_col = args.dx_col
    if dx_col is None or dx_col not in df.columns:
        raise ValueError("Could not find diagnosis column. Provide --dx-col explicitly.")
    df["group3"] = df[dx_col].map(map_group3)
    df["sex_clean"] = df[args.sex_col].map(clean_sex_value)

    if not sorted_st_feature_cols(df, "t1vol_st"):
        raise ValueError("No t1vol_st* feature columns found.")
    if not sorted_st_feature_cols(df, "aslcbf_st"):
        raise ValueError("No aslcbf_st* feature columns found.")

    region_map = load_region_mapping(Path(args.region_map) if args.region_map else None)

    healthy_df = df[df[args.healthy_col] & df[args.target_col].notna() & df[args.group_col].notna()].copy().reset_index(drop=True)
    if healthy_df[args.group_col].nunique() < 2:
        raise ValueError("Need at least two normative RIDs for train/test splitting.")
    train_df, test_df = make_train_test_split(healthy_df, args.group_col, args.test_size, args.random_state)

    predictions_df = df.copy()
    performance_rows: List[Dict[str, Any]] = []
    explanation_tables: Dict[str, pd.DataFrame] = {}

    print("Applying embedded locked six-model configuration")
    print(f"  data: {args.data}")
    print(f"  normative train: {len(train_df)} scans / {train_df[args.group_col].nunique()} RIDs")
    print(f"  normative test:  {len(test_df)} scans / {test_df[args.group_col].nunique()} RIDs")
    print("  parameter source: embedded MODEL_CONFIGS")

    for model_name in MODEL_ORDER:
        config = MODEL_CONFIGS[model_name]
        feature_cols = get_feature_columns(df, config["feature_set"])
        x_train = train_df[feature_cols]
        y_train = train_df[args.target_col].astype(float).to_numpy()
        x_test = test_df[feature_cols]
        y_test = test_df[args.target_col].astype(float).to_numpy()

        print(f"  fitting {model_name} ...", flush=True)
        cv = cross_validate_oof(
            model_name,
            config,
            train_df,
            feature_cols,
            args.group_col,
            args.target_col,
            args.cv_folds,
            args.random_state,
        )

        model = make_model(config["algorithm"], config["params"], random_state=args.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)

        pred_train = np.asarray(model.predict(x_train), dtype=float)
        pred_test = np.asarray(model.predict(x_test), dtype=float)
        train_perf = eval_predictions(y_train, pred_train)
        test_perf = eval_predictions(y_test, pred_test)

        pred_all = np.asarray(model.predict(df[feature_cols]), dtype=float)
        predictions_df[f"pred_age__{model_name}"] = pred_all
        predictions_df[f"pad__{model_name}"] = pred_all - df[args.target_col]

        performance_rows.append(
            {
                "model": model_name,
                "feature_set": config["feature_set"],
                "algorithm": config["algorithm"],
                "n_features": len(feature_cols),
                "params_json": json.dumps(config["params"], sort_keys=True),
                "train_r2": train_perf["r2"],
                "train_rmse": train_perf["rmse"],
                "train_mae": train_perf["mae"],
                "cv_oof_r2": cv["cv_oof_r2"],
                "cv_oof_rmse": cv["cv_oof_rmse"],
                "cv_oof_mae": cv["cv_oof_mae"],
                "cv_gap_r2": cv["cv_gap_r2"],
                "test_r2": test_perf["r2"],
                "test_rmse": test_perf["rmse"],
                "test_mae": test_perf["mae"],
                "n_train_scans": int(len(train_df)),
                "n_train_rids": int(train_df[args.group_col].nunique()),
                "n_test_scans": int(len(test_df)),
                "n_test_rids": int(test_df[args.group_col].nunique()),
            }
        )
        explanation_tables[model_name] = model_importance_table(model_name, model, feature_cols, config, region_map)

    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(tabdir / "model_performance.csv", index=False)

    if args.analysis_level == "first_scan_per_rid":
        analysis_df = first_scan_per_rid(predictions_df, args.group_col, args.target_col)
    else:
        analysis_df = predictions_df.copy()
    analysis_df = analysis_df[analysis_df["group3"].isin(GROUP_ORDER)].copy()

    plot_predicted_vs_chronological(analysis_df, MODEL_ORDER, performance_df, args.target_col, figdir)

    pad_regression_rows: List[pd.DataFrame] = []
    pad_summary_rows: List[pd.DataFrame] = []
    for model_name in MODEL_ORDER:
        raw_col = f"pad__{model_name}"
        adjusted_col = f"pad_age_sex_adjusted__{model_name}"
        analysis_df[adjusted_col] = residualize_pad_for_age_sex(analysis_df, raw_col, args.target_col, "sex_clean")
        reg = run_pad_regression(analysis_df, model_name, raw_col, args.target_col, "sex_clean")
        if not reg.empty:
            pad_regression_rows.append(reg)
        pad_summary_rows.append(pad_group_summary(analysis_df, model_name, raw_col, adjusted_col))

    pad_regression_df = pd.concat(pad_regression_rows, ignore_index=True) if pad_regression_rows else pd.DataFrame()
    if not pad_regression_df.empty:
        mask = pad_regression_df["p_value"].notna()
        pad_regression_df["fdr_q_value"] = np.nan
        if mask.any():
            _, qvals = fdrcorrection(pad_regression_df.loc[mask, "p_value"].to_numpy())
            pad_regression_df.loc[mask, "fdr_q_value"] = qvals
    pad_regression_df.to_csv(tabdir / "pad_contrasts_age_sex_adjusted.csv", index=False)

    pad_summary_df = pd.concat(pad_summary_rows, ignore_index=True) if pad_summary_rows else pd.DataFrame()
    pad_summary_df.to_csv(tabdir / "pad_group_summary.csv", index=False)

    plot_pad_boxplots(analysis_df, MODEL_ORDER, pad_regression_df, figdir)

    top_feature_rows: List[pd.DataFrame] = []
    for feature_set in ["t1", "asl", "fusion"]:
        perf_sub = performance_df[performance_df["feature_set"] == feature_set].sort_values("test_r2", ascending=False)
        if perf_sub.empty:
            continue
        selected_model = str(perf_sub.iloc[0]["model"])
        top = explanation_tables[selected_model].head(args.top_n_regions).copy()
        top.insert(0, "selected_for_feature_set", FEATURE_LABELS[feature_set])
        top_feature_rows.append(top)
        plot_top_regions(
            top,
            f"{FEATURE_LABELS[feature_set]} explanation ({selected_model.replace('_', ' ')})",
            figdir / f"top_regions_feature_{FEATURE_LABELS[feature_set]}.png",
            args.top_n_regions,
        )

    top_feature_df = pd.concat(top_feature_rows, ignore_index=True) if top_feature_rows else pd.DataFrame()
    keep_cols = [
        "selected_for_feature_set",
        "model",
        "rank",
        "feature_modality",
        "region_name",
        "hemisphere",
        "tissue_class",
        "importance_type",
        "importance",
        "importance_abs",
    ]
    if not top_feature_df.empty:
        top_feature_df[keep_cols].to_csv(tabdir / "top_regions_by_feature_set.csv", index=False)

    print("\nDone. Report-ready outputs:")
    print(f"  {figdir / 'predicted_vs_chronological_all_models.png'}")
    print(f"  {figdir / 'pad_boxplots_age_sex_adjusted_all_models.png'}")
    print(f"  {figdir / 'top_regions_feature_T1.png'}")
    print(f"  {figdir / 'top_regions_feature_ASL.png'}")
    print(f"  {figdir / 'top_regions_feature_Fusion.png'}")
    print(f"  {tabdir / 'model_performance.csv'}")
    print(f"  {tabdir / 'pad_contrasts_age_sex_adjusted.csv'}")
    print(f"  {tabdir / 'pad_group_summary.csv'}")
    print(f"  {tabdir / 'top_regions_by_feature_set.csv'}")


if __name__ == "__main__":
    main()
