# -*- coding: utf-8 -*-
"""
Візуальна перевірка "чи модель дійсно щось відрізняє": ROC-AUC - число,
яке треба вміти прочитати. Цей скрипт натомість будує графік, який можна
просто подивитись: сортує монети кожного тестового фолду за прогнозом
моделі, ділить на 5 однакових груп (квінтилів) і рахує, якою насправді
була їхня надлишкова дохідність за наступні 60 днів.

Якщо модель має реальний сигнал, бар для квінтиля 5 (найвищий прогноз)
має бути вище бару для квінтиля 1 (найнижчий прогноз), і бажано щоб
зростання було більш-менш монотонним. Це саме той графік, який можна
"просто подивитись" - без довіри до жодної окремої метрики.

Порівнюються дві моделі на тих самих семи walk-forward фолдах з
train_bonus_model.py: справжня базова модель курсової і блендинг
v2+v3, що дав найкращий середній ROC-AUC. Усі прогнози - out-of-fold:
кожен рядок передбачений моделлю, навченою тільки на даних до embargo
перед його checkpoint.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from train_bonus_model import (
    CAT_FEATURES, MIN_FOLD_TRAIN_CHECKPOINTS, MIN_TUNING_CHECKPOINTS,
    NUM_FEATURES_BASE, NUM_FEATURES_NEW, NUM_FEATURES_XS, EMBARGO, TEST_BLOCK,
    load_panel, make_pipeline, walk_forward_folds,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
N_BUCKETS = 5


def fit_predict(df, train_cps, test_cps, numeric_features, categorical_features, params):
    train_fold = df[df["checkpoint"].isin(train_cps)]
    test_fold = df[df["checkpoint"].isin(test_cps)]
    X_train = train_fold[numeric_features + categorical_features]
    y_train = train_fold["outperform_market"]
    X_test = test_fold[numeric_features + categorical_features]

    model = HistGradientBoostingClassifier(random_state=42, **{
        k.replace("model__", ""): v for k, v in params.items()
    }) if params else HistGradientBoostingClassifier(
        random_state=42, max_iter=150, learning_rate=0.08, max_depth=4)
    pipe = make_pipeline(numeric_features, categorical_features, model)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    return test_fold.assign(_proba=proba)


def main():
    with open(ROOT / "best_params.json", encoding="utf-8") as f:
        params = json.load(f)
    best_params_v2 = params["best_params_v2"]
    best_params_v3 = params["best_params_v3"]

    df = load_panel()
    checkpoints = sorted(df["checkpoint"].unique())
    numeric_v2 = NUM_FEATURES_BASE + NUM_FEATURES_NEW
    folds = walk_forward_folds(checkpoints, TEST_BLOCK, EMBARGO,
                                MIN_TUNING_CHECKPOINTS, MIN_FOLD_TRAIN_CHECKPOINTS)

    baseline_chunks, ensemble_chunks = [], []
    for train_cps, test_cps in folds:
        base_pred = fit_predict(df, train_cps, test_cps, NUM_FEATURES_BASE, CAT_FEATURES, None)
        baseline_chunks.append(base_pred[["checkpoint", "symbol", "outperform_market",
                                           "excess_return_60d", "_proba"]])

        v2_pred = fit_predict(df, train_cps, test_cps, numeric_v2, CAT_FEATURES, best_params_v2)
        v3_pred = fit_predict(df, train_cps, test_cps, NUM_FEATURES_XS, CAT_FEATURES, best_params_v3)
        rank2 = v2_pred["_proba"].rank(pct=True).values
        rank3 = v3_pred["_proba"].rank(pct=True).values
        blended = (rank2 + rank3) / 2
        ensemble_chunks.append(v2_pred.assign(_proba=blended)[
            ["checkpoint", "symbol", "outperform_market", "excess_return_60d", "_proba"]])

    baseline_oof = pd.concat(baseline_chunks, ignore_index=True)
    ensemble_oof = pd.concat(ensemble_chunks, ignore_index=True)
    print(f"Пул out-of-fold прогнозів: {len(baseline_oof)} рядків "
          f"({baseline_oof['checkpoint'].min().date()} - {baseline_oof['checkpoint'].max().date()})")

    def bucket_summary(oof_df, label):
        oof_df = oof_df.copy()
        oof_df["bucket"] = pd.qcut(oof_df["_proba"], N_BUCKETS, labels=False, duplicates="drop") + 1
        summary = oof_df.groupby("bucket").agg(
            mean_excess_return=("excess_return_60d", "mean"),
            outperform_rate=("outperform_market", "mean"),
            n=("symbol", "count"),
        )
        summary.columns = pd.MultiIndex.from_product([[label], summary.columns])
        return summary

    baseline_summary = bucket_summary(baseline_oof, "baseline")
    ensemble_summary = bucket_summary(ensemble_oof, "ensemble")
    combined = pd.concat([baseline_summary, ensemble_summary], axis=1)
    print()
    print(combined.to_string())

    overall_rate = df["outperform_market"].mean()

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        x = np.arange(1, N_BUCKETS + 1)
        width = 0.38

        ax = axes[0]
        ax.bar(x - width / 2, baseline_summary[("baseline", "mean_excess_return")] * 100,
               width, label="Базова модель курсової", color="#9CA3AF")
        ax.bar(x + width / 2, ensemble_summary[("ensemble", "mean_excess_return")] * 100,
               width, label="Блендинг v2+v3", color="#2563EB")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xlabel("Квінтиль прогнозу (1 = найнижчий, 5 = найвищий)")
        ax.set_ylabel("Середня надлишкова дохідність, %")
        ax.set_title("Реальна дохідність за квінтилем прогнозу")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.bar(x - width / 2, baseline_summary[("baseline", "outperform_rate")] * 100,
               width, label="Базова модель курсової", color="#9CA3AF")
        ax.bar(x + width / 2, ensemble_summary[("ensemble", "outperform_rate")] * 100,
               width, label="Блендинг v2+v3", color="#2563EB")
        ax.axhline(overall_rate * 100, color="black", linestyle="--", linewidth=1,
                   label=f"Базова частка в даних ({overall_rate * 100:.1f}%)")
        ax.set_xticks(x)
        ax.set_xlabel("Квінтиль прогнозу (1 = найнижчий, 5 = найвищий)")
        ax.set_ylabel("Частка монет, що випередили ринок, %")
        ax.set_title("Частка \"переможців\" за квінтилем прогнозу")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out_path = ROOT / "figures" / "quintile_validation.png"
        fig.savefig(out_path, dpi=150)
        print()
        print("Збережено графік:", out_path)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
