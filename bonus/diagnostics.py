# -*- coding: utf-8 -*-
"""
Діагностика, на яку посилається додаток А курсової роботи.

Рахує три речі на місячній панелі:
  1. Якість ранжування всередині checkpoint: rank IC (кореляція Спірмена
     між прогнозом і реальною надлишковою дохідністю) і ROC-AUC окремо
     для цілі "краща за BTC" і для цілі "краща за медіану місяця".
  2. Placebo-тест: та сама модель, навчена на мітках, випадково
     перемішаних усередині кожного train-місяця. Якщо конвеєр має витік
     або помилку, placebo теж показуватиме якість; якщо сигнал справжній,
     placebo впаде до нуля.
  3. Одновимірний rank IC кожної сирої ознаки на тестовому періоді, щоб
     побачити, на що саме модель спирається.

Результат: bonus/diagnostics.csv, bonus/feature_ic.csv
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from strategy_backtest import NUM_V2, add_scores, fit_classifier
from train_bonus_model import (CAT_FEATURES, EMBARGO, MIN_FOLD_TRAIN_CHECKPOINTS,
                               MIN_TUNING_CHECKPOINTS, NUM_FEATURES_BASE, NUM_FEATURES_NEW,
                               TEST_BLOCK, load_panel, walk_forward_folds)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
MODELS = {
    "Базова модель курсової": "s_baseline",
    "Блендинг, ціль краща за BTC": "s_ens_btc_target",
    "Блендинг, ціль краща за медіану": "s_xs_median_blend",
}


def main():
    oof = pd.read_csv(ROOT / "oof_predictions.csv", parse_dates=["checkpoint"])
    oof = pd.concat([add_scores(g) for _, g in oof.groupby("checkpoint")])
    med = oof.groupby("checkpoint")["excess_return_60d"].transform("median")
    oof["beat_median"] = (oof["excess_return_60d"] > med).astype(int)
    weeks = sorted(oof["checkpoint"].unique())
    oof["fold"] = oof["checkpoint"].map({cp: i // TEST_BLOCK for i, cp in enumerate(weeks)})

    rows = {}
    for label, score in MODELS.items():
        ic = [spearmanr(g[score], g["excess_return_60d"])[0] for _, g in oof.groupby("checkpoint")]
        auc_btc = [roc_auc_score(g["outperform_market"], g[score]) for _, g in oof.groupby("fold")]
        auc_med = [roc_auc_score(g["beat_median"], g[score]) for _, g in oof.groupby("fold")]
        rows[label] = {
            "rank IC (середній за місяць)": np.mean(ic),
            "ROC-AUC, ціль краща за BTC": np.mean(auc_btc),
            "ROC-AUC, ціль краща за медіану": np.mean(auc_med),
            "найгірший фолд (краща за медіану)": np.min(auc_med),
        }

    df = load_panel()
    med_full = df.groupby("checkpoint")["excess_return_60d"].transform("median")
    df["beat_checkpoint_median"] = (df["excess_return_60d"] > med_full).astype(int)
    params = json.load(open(ROOT / "best_params.json", encoding="utf-8"))["best_params_v2"]
    checkpoints = sorted(df["checkpoint"].unique())
    folds = walk_forward_folds(checkpoints, TEST_BLOCK, EMBARGO,
                               MIN_TUNING_CHECKPOINTS, MIN_FOLD_TRAIN_CHECKPOINTS)
    for train_cps, test_cps in folds:
        assert max(train_cps) + pd.Timedelta(days=60) < min(test_cps), \
            "forward-вікно train заходить у тестовий період"

    rng = np.random.default_rng(0)
    real_ic, placebo_ic = [], []
    for train_cps, test_cps in folds:
        train = df[df["checkpoint"].isin(train_cps)].copy()
        test = df[df["checkpoint"].isin(test_cps)].copy()
        test["p"] = fit_classifier(train, test, NUM_V2, params, target="beat_checkpoint_median")
        train["placebo"] = train.groupby("checkpoint")["beat_checkpoint_median"].transform(
            lambda s: rng.permutation(s.values))
        test["pp"] = fit_classifier(train, test, NUM_V2, params, target="placebo")
        for _, g in test.groupby("checkpoint"):
            real_ic.append(spearmanr(g["p"], g["excess_return_60d"])[0])
            placebo_ic.append(spearmanr(g["pp"], g["excess_return_60d"])[0])
    rows["Placebo (перемішані мітки)"] = {"rank IC (середній за місяць)": np.mean(placebo_ic)}
    rows["Та сама модель, справжні мітки"] = {"rank IC (середній за місяць)": np.mean(real_ic)}

    out = pd.DataFrame(rows).T.round(4)
    out.to_csv(ROOT / "diagnostics.csv")
    print(out.to_string())

    test_period = df[df["checkpoint"] >= folds[0][1][0]]
    ic = {c: np.nanmean([spearmanr(g[c], g["excess_return_60d"])[0]
                         for _, g in test_period.groupby("checkpoint")])
          for c in NUM_FEATURES_BASE + NUM_FEATURES_NEW}
    fic = pd.Series(ic).dropna().sort_values(key=abs, ascending=False).round(4)
    fic.to_csv(ROOT / "feature_ic.csv", header=["rank_IC"])
    print()
    print("Одновимірний rank IC ознак на тестовому періоді:")
    print(fic.to_string())


if __name__ == "__main__":
    main()
