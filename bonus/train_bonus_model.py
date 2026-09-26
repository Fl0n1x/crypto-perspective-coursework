# -*- coding: utf-8 -*-
"""
Бонусний модельний експеримент (поза межами оцінювання курсової, крок
"НАСТУПНИЙ КРОК" з TASKS.md). Мета - показати модель, яка справді
узагальнюється на різні режими ринку, а не лише крос-валідацію на
train, що складався переважно з bull.

Що змінено відносно розділу 5 курсової:
  1. Ознаки: до базового фінального набору додано RSI-14, MACD-гістограму
     (12/26/9), позицію в смугах Боллінджера (%B, 20d), прискорення
     моментуму і RSI-14 самого BTC (стан ринку тоншим сигналом, ніж
     3-категорійний market_regime). Рахує `build_features_v2.py`.
  2. Валідація: один статичний train/test-спліт замінено на purged
     walk-forward (експандне вікно, тестові блоки по 4 checkpoint,
     embargo 2 checkpoint перед кожним тестовим блоком, щоб forward-вікно
     таргету з train не заходило в тестовий період). Це дає кілька
     тестових блоків, які покривають bull->bear і bear->bull переходи,
     а не лише один блок, що випадково опинився суцільним bear/sideways.
  3. Гіперпараметри: підібрані один раз (RandomizedSearchCV,
     StratifiedGroupKFold(5, group=symbol)) на першому великому блоці
     історії - так, як це реально роблять перед деплоєм, а не одноразово
     на всьому датасеті заднім числом.

Результат - таблиця fold-ів (тестовий регiон, регiм, ROC-AUC базового і
підібраного конвеєра) і підсумкове порівняння.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

SEED = 42
TEST_BLOCK = 3
EMBARGO = 2
MIN_TUNING_CHECKPOINTS = 12
MIN_FOLD_TRAIN_CHECKPOINTS = 8

ROOT = Path(__file__).parent
PANEL_PATH = ROOT / "panel_v2.csv"

NUM_FEATURES_BASE = [
    "trailing_return_30d", "trailing_return_90d", "trailing_volatility_30d",
    "trailing_volume_trend", "drawdown_from_high_180d", "avg_quote_volume_30d",
    "coin_age_days", "relative_strength_vs_btc_90d", "sector_avg_trailing_return_30d",
]
NUM_FEATURES_NEW = [
    "rsi_14", "macd_hist_12_26_9", "bollinger_pctb_20", "momentum_acceleration",
    "btc_rsi_14",
]
CAT_FEATURES = ["sector", "market_regime"]

# ознаки, чий рівень сильно залежить від того, який зараз режим ринку
# (наприклад +5% за 30д - це "слабко" в bull і "сильно" в bear) - їх
# кроссекційна (within-checkpoint) z-нормалізація прибирає цю залежність
# від абсолютного рівня і залишає лише відносний ранг монети того місяця
XS_NORMALIZE_COLS = [
    "trailing_return_30d", "trailing_return_90d", "trailing_volatility_30d",
    "trailing_volume_trend", "drawdown_from_high_180d", "rsi_14",
    "macd_hist_12_26_9", "bollinger_pctb_20", "momentum_acceleration",
]
NUM_FEATURES_XS = [f"{c}_xs" for c in XS_NORMALIZE_COLS] + [
    "relative_strength_vs_btc_90d", "coin_age_days", "avg_quote_volume_30d_log",
]


def load_panel():
    df = pd.read_csv(PANEL_PATH, parse_dates=["checkpoint"])
    df["sector"] = df["sector"].fillna("Other")
    df = df.sort_values(["symbol", "checkpoint"]).reset_index(drop=True)
    df["sector_avg_trailing_return_30d"] = (
        df.groupby(["checkpoint", "sector"])["trailing_return_30d"].transform("mean")
    )
    df["avg_quote_volume_30d_log"] = np.log1p(df["avg_quote_volume_30d"])

    def xs_zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std and std > 0 else s * 0

    for col in XS_NORMALIZE_COLS:
        df[f"{col}_xs"] = df.groupby("checkpoint")[col].transform(xs_zscore)
    return df


def make_pipeline(numeric_features, categorical_features, model):
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def tune_hyperparameters(df, numeric_features, categorical_features, tuning_checkpoints):
    tune_df = df[df["checkpoint"].isin(tuning_checkpoints)]
    X = tune_df[numeric_features + categorical_features]
    y = tune_df["outperform_market"]
    groups = tune_df["symbol"]

    base_model = HistGradientBoostingClassifier(random_state=SEED)
    pipe = make_pipeline(numeric_features, categorical_features, base_model)

    param_dist = {
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_iter": randint(100, 500),
        "model__max_depth": [3, 4, 5, 6, None],
        "model__max_leaf_nodes": randint(15, 63),
        "model__l2_regularization": loguniform(1e-3, 1.0),
        "model__min_samples_leaf": randint(10, 60),
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=40, scoring="roc_auc",
        cv=cv, random_state=SEED, n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    print(f"Підбір гіперпараметрів на {len(tuning_checkpoints)} checkpoint "
          f"({tuning_checkpoints[0].date()} - {tuning_checkpoints[-1].date()}), "
          f"{len(tune_df)} рядків")
    print("Найкращий ROC-AUC на тюнінговій крос-валідації:", round(search.best_score_, 4))
    print("Найкращі гіперпараметри:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    return search.best_params_


def walk_forward_folds(checkpoints, test_block, embargo, min_tuning_checkpoints,
                        min_fold_train_checkpoints):
    folds = []
    start = min_tuning_checkpoints
    while start < len(checkpoints):
        test_cps = checkpoints[start:start + test_block]
        train_end = start - embargo
        train_cps = checkpoints[:max(train_end, 0)]
        if len(train_cps) >= min_fold_train_checkpoints and test_cps:
            folds.append((train_cps, test_cps))
        start += test_block
    return folds


def evaluate_fold(df, train_cps, test_cps, numeric_features, categorical_features, model_params,
                   return_proba=False):
    train_fold = df[df["checkpoint"].isin(train_cps)]
    test_fold = df[df["checkpoint"].isin(test_cps)]

    X_train = train_fold[numeric_features + categorical_features]
    y_train = train_fold["outperform_market"]
    X_test = test_fold[numeric_features + categorical_features]
    y_test = test_fold["outperform_market"]

    model = HistGradientBoostingClassifier(random_state=SEED, **{
        k.replace("model__", ""): v for k, v in model_params.items()
    }) if model_params else HistGradientBoostingClassifier(
        random_state=SEED, max_iter=150, learning_rate=0.08, max_depth=4)

    pipe = make_pipeline(numeric_features, categorical_features, model)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)

    regime_counts = test_fold["market_regime"].value_counts(normalize=True)
    dominant_regime = regime_counts.idxmax()

    result = {
        "test_start": test_cps[0].date(),
        "test_end": test_cps[-1].date(),
        "dominant_regime": dominant_regime,
        "n_train": len(train_fold),
        "n_test": len(test_fold),
        "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
    }
    if return_proba:
        return result, proba, y_test.reset_index(drop=True)
    return result


def evaluate_ensemble_fold(df, train_cps, test_cps, params_v2, params_v3):
    """Блендинг: усереднення ймовірностей моделі на сирих ознаках (v2) і
    моделі на кроссекційно нормалізованих ознаках (v3). Обидві моделі
    бачать різне представлення того самого сигналу (абсолютний рівень
    проти відносного рангу того місяця), тому їхні помилки частково
    некорельовані - блендинг мав би згладжувати провальні фолди кожної
    окремо."""
    numeric_v2 = NUM_FEATURES_BASE + NUM_FEATURES_NEW
    r2, proba2, y_test = evaluate_fold(df, train_cps, test_cps, numeric_v2, CAT_FEATURES,
                                        params_v2, return_proba=True)
    r3, proba3, _ = evaluate_fold(df, train_cps, test_cps, NUM_FEATURES_XS, CAT_FEATURES,
                                   params_v3, return_proba=True)
    proba_ens = (proba2 + proba3) / 2
    pred_ens = (proba_ens >= 0.5).astype(int)
    return {
        "test_start": r2["test_start"],
        "test_end": r2["test_end"],
        "dominant_regime": r2["dominant_regime"],
        "n_test": r2["n_test"],
        "roc_auc": roc_auc_score(y_test, proba_ens) if y_test.nunique() > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_ens),
        "accuracy": accuracy_score(y_test, pred_ens),
    }


def main():
    df = load_panel()
    checkpoints = sorted(df["checkpoint"].unique())
    numeric_features = NUM_FEATURES_BASE + NUM_FEATURES_NEW
    categorical_features = CAT_FEATURES

    tuning_checkpoints = checkpoints[:MIN_TUNING_CHECKPOINTS]
    print("--- Тюнінг: сирі ознаки + нові індикатори (v2) ---")
    best_params_v2 = tune_hyperparameters(df, numeric_features, categorical_features, tuning_checkpoints)
    print()
    print("--- Тюнінг: кроссекційно нормалізовані ознаки (v3) ---")
    best_params_v3 = tune_hyperparameters(df, NUM_FEATURES_XS, categorical_features, tuning_checkpoints)
    print()

    folds = walk_forward_folds(checkpoints, TEST_BLOCK, EMBARGO,
                                MIN_TUNING_CHECKPOINTS, MIN_FOLD_TRAIN_CHECKPOINTS)
    print(f"Walk-forward фолдів: {len(folds)}")
    print()

    baseline_rows, v2_rows, v3_rows, ens_rows = [], [], [], []
    for train_cps, test_cps in folds:
        baseline_rows.append(evaluate_fold(df, train_cps, test_cps, numeric_features,
                                            categorical_features, model_params=None))
        v2_rows.append(evaluate_fold(df, train_cps, test_cps, numeric_features,
                                      categorical_features, model_params=best_params_v2))
        v3_rows.append(evaluate_fold(df, train_cps, test_cps, NUM_FEATURES_XS,
                                      categorical_features, model_params=best_params_v3))
        ens_rows.append(evaluate_ensemble_fold(df, train_cps, test_cps, best_params_v2, best_params_v3))

    baseline_df = pd.DataFrame(baseline_rows)
    v2_df = pd.DataFrame(v2_rows)
    v3_df = pd.DataFrame(v3_rows)
    ens_df = pd.DataFrame(ens_rows)

    report = pd.DataFrame({
        "test_start": baseline_df["test_start"],
        "test_end": baseline_df["test_end"],
        "dominant_regime": baseline_df["dominant_regime"],
        "n_test": baseline_df["n_test"],
        "roc_auc_baseline": baseline_df["roc_auc"].round(4),
        "roc_auc_tuned_v2": v2_df["roc_auc"].round(4),
        "roc_auc_xs_v3": v3_df["roc_auc"].round(4),
        "roc_auc_ensemble": ens_df["roc_auc"].round(4),
        "balanced_acc_baseline": baseline_df["balanced_accuracy"].round(4),
        "balanced_acc_tuned_v2": v2_df["balanced_accuracy"].round(4),
        "balanced_acc_xs_v3": v3_df["balanced_accuracy"].round(4),
        "balanced_acc_ensemble": ens_df["balanced_accuracy"].round(4),
    })
    print(report.to_string(index=False))
    print()
    print("Середній ROC-AUC по фолдах (базовий конвеєр курсової):     ",
          round(baseline_df["roc_auc"].mean(), 4), "std:", round(baseline_df["roc_auc"].std(), 4))
    print("Середній ROC-AUC по фолдах (тюнінг + нові ознаки, v2):     ",
          round(v2_df["roc_auc"].mean(), 4), "std:", round(v2_df["roc_auc"].std(), 4))
    print("Середній ROC-AUC по фолдах (кроссекційна нормалізація, v3):",
          round(v3_df["roc_auc"].mean(), 4), "std:", round(v3_df["roc_auc"].std(), 4))
    print("Середній ROC-AUC по фолдах (блендинг v2+v3, ensemble):     ",
          round(ens_df["roc_auc"].mean(), 4), "std:", round(ens_df["roc_auc"].std(), 4))
    print()
    by_regime = report.groupby("dominant_regime")[
        ["roc_auc_baseline", "roc_auc_tuned_v2", "roc_auc_xs_v3", "roc_auc_ensemble"]].mean().round(4)
    print("Середній ROC-AUC за домінантним режимом тестового блоку:")
    print(by_regime.to_string())

    report.to_csv(ROOT / "walk_forward_report.csv", index=False)
    print()
    print("Збережено:", ROOT / "walk_forward_report.csv")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(report))
        ax.plot(x, report["roc_auc_baseline"], marker="o", label="Базовий (курсова)", color="#9CA3AF")
        ax.plot(x, report["roc_auc_tuned_v2"], marker="o", label="Тюнінг + нові ознаки (v2)", color="#60A5FA")
        ax.plot(x, report["roc_auc_xs_v3"], marker="o", label="Кроссекційна нормалізація (v3)", color="#2563EB")
        ax.plot(x, report["roc_auc_ensemble"], marker="s", label="Блендинг v2+v3", color="#16A34A", linewidth=2)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Випадковий рівень")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.test_start}\n({r.dominant_regime})" for r in report.itertuples()],
                            fontsize=8)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Walk-forward ROC-AUC по тестових блоках")
        ax.legend()
        plt.tight_layout()
        fig.savefig(ROOT / "figures" / "walk_forward_roc_auc.png", dpi=150)
        print("Збережено графік:", ROOT / "figures" / "walk_forward_roc_auc.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
