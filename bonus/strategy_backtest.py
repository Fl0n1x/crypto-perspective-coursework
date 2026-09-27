# -*- coding: utf-8 -*-
"""
Бектест простих правил відбору монет на out-of-fold прогнозах.

Квінтильний графік (validate_visually.py) відповідає на питання "чи
модель сортує". Тут питання практичніше: якщо в кожному checkpoint
купувати монети, які обрала модель, і тримати 60 днів, чи виходить
результат у плюс відносно простого тримання BTC.

Кожен прогноз зроблено моделлю, навченою лише на даних до embargo перед
тестовим блоком (ті самі 7 walk-forward фолдів, що й у
train_bonus_model.py). Портфель формується окремо в кожному checkpoint,
рівні ваги, результат - середня excess_return_60d обраних монет (0,
якщо правило каже лишитись у BTC). Періоди утримання по 60 днів при
щомісячному ребалансуванні перекриваються, тому результати не можна
компаундувати - порівнюємо середній результат за checkpoint.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from train_bonus_model import (
    CAT_FEATURES, EMBARGO, MIN_FOLD_TRAIN_CHECKPOINTS, MIN_TUNING_CHECKPOINTS,
    NUM_FEATURES_BASE, NUM_FEATURES_NEW, NUM_FEATURES_XS, SEED, TEST_BLOCK,
    load_panel, make_pipeline, walk_forward_folds,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
NUM_V2 = NUM_FEATURES_BASE + NUM_FEATURES_NEW
KEEP_COLS = ["checkpoint", "symbol", "market_regime", "avg_quote_volume_30d",
             "excess_return_60d", "outperform_market", "forward_return_60d",
             "btc_forward_return_60d"]


def _unprefix(params):
    return {k.replace("model__", ""): v for k, v in (params or {}).items()}


def fit_classifier(train, test, numeric, params, target="outperform_market"):
    model = (HistGradientBoostingClassifier(random_state=SEED, **_unprefix(params)) if params
             else HistGradientBoostingClassifier(random_state=SEED, max_iter=150,
                                                 learning_rate=0.08, max_depth=4))
    pipe = make_pipeline(numeric, CAT_FEATURES, model)
    pipe.fit(train[numeric + CAT_FEATURES], train[target])
    return pipe.predict_proba(test[numeric + CAT_FEATURES])[:, 1]


def fit_regressor(train, test, numeric, params, target):
    model = HistGradientBoostingRegressor(random_state=SEED, **_unprefix(params))
    pipe = make_pipeline(numeric, CAT_FEATURES, model)
    pipe.fit(train[numeric + CAT_FEATURES], target)
    return pipe.predict(test[numeric + CAT_FEATURES])


def build_oof(df, params):
    checkpoints = sorted(df["checkpoint"].unique())
    folds = walk_forward_folds(checkpoints, TEST_BLOCK, EMBARGO,
                                MIN_TUNING_CHECKPOINTS, MIN_FOLD_TRAIN_CHECKPOINTS)
    chunks = []
    for train_cps, test_cps in folds:
        train = df[df["checkpoint"].isin(train_cps)]
        test = df[df["checkpoint"].isin(test_cps)].copy()

        test["p_baseline"] = fit_classifier(train, test, NUM_FEATURES_BASE, None)
        test["p_v2"] = fit_classifier(train, test, NUM_V2, params["best_params_v2"])
        test["p_v3"] = fit_classifier(train, test, NUM_FEATURES_XS, params["best_params_v3"])
        test["p_ens"] = (test["p_v2"] + test["p_v3"]) / 2

        # регресія на саму величину надлишкової дохідності: класифікатор
        # однаково "радіє" +1% і +300%, а для середнього результату портфеля
        # важлива саме величина; хвости обрізані за 1/99 перцентилем train,
        # щоб кілька мем-пампів не визначали всю модель
        lo, hi = train["excess_return_60d"].quantile([0.01, 0.99])
        y_clip = train["excess_return_60d"].clip(lo, hi)
        test["pred_excess"] = fit_regressor(train, test, NUM_V2, params["best_params_rank"], y_clip)
        test["pred_rank"] = fit_regressor(train, test, NUM_V2, params["best_params_rank"],
                                          train["rank_target_pct"])

        # кроссекційні цілі: "краща за медіану свого checkpoint" і "у верхній
        # третині свого checkpoint". Загальноринковий рух альтів проти BTC
        # модель з ознак монети передбачити не може, а в цілі "краща за BTC"
        # він домінує (у місяць обвалу всі мітки 0, у місяць альтсезону всі
        # 1) - такі місяці нічого не кажуть про ранжування монет між собою
        test["p_xs_median"] = fit_classifier(train, test, NUM_V2, params["best_params_v2"],
                                             target="beat_checkpoint_median")
        test["p_xs_median_v3"] = fit_classifier(train, test, NUM_FEATURES_XS, params["best_params_v3"],
                                                target="beat_checkpoint_median")
        test["p_xs_tercile"] = fit_classifier(train, test, NUM_V2, params["best_params_v2"],
                                              target="top_tercile")
        chunks.append(test[KEEP_COLS + ["p_baseline", "p_v2", "p_v3", "p_ens", "pred_excess",
                                        "pred_rank", "p_xs_median", "p_xs_median_v3",
                                        "p_xs_tercile"]])
    return pd.concat(chunks, ignore_index=True)


def top_share(group, score, share=0.2):
    k = max(1, int(round(len(group) * share)))
    return group.nlargest(k, score)["excess_return_60d"].mean()


def bottom_share(group, score, share=0.2):
    k = max(1, int(round(len(group) * share)))
    return group.nsmallest(k, score)["excess_return_60d"].mean()


# кожен скор - сума процентильних рангів складових у межах checkpoint,
# тож блендинги не залежать від шкали окремих моделей
SCORES = {
    "baseline": ["p_baseline"],
    "ens_btc_target": ["p_v2", "p_v3"],
    "rank_regression": ["pred_rank"],
    "xs_median": ["p_xs_median"],
    "xs_median_blend": ["p_xs_median", "p_xs_median_v3"],
    "xs_tercile": ["p_xs_tercile"],
    "xs_combo": ["p_xs_median", "p_xs_median_v3", "pred_rank"],
}


def add_scores(g):
    g = g.copy()
    for name, parts in SCORES.items():
        g[f"s_{name}"] = sum(g[p].rank(pct=True) for p in parts)
    return g


def per_checkpoint_strategies(oof):
    rows = []
    for cp, g in oof.groupby("checkpoint"):
        g = add_scores(g)
        liquid = add_scores(g[g["avg_quote_volume_30d"] >= g["avg_quote_volume_30d"].median()])
        confident = g[g["p_ens"] > 0.5]
        positive_ev = g[g["pred_excess"] > 0]
        row = {
            "checkpoint": cp,
            "regime": g["market_regime"].iloc[0],
            "universe": g["excess_return_60d"].mean(),
            "confident_or_btc": confident["excess_return_60d"].mean() if len(confident) else 0.0,
            "confident_n": len(confident),
            "positive_ev_or_btc": positive_ev["excess_return_60d"].mean() if len(positive_ev) else 0.0,
            "positive_ev_n": len(positive_ev),
        }
        for name in SCORES:
            s = f"s_{name}"
            row[f"top20__{name}"] = top_share(g, s)
            row[f"ls__{name}"] = top_share(g, s) - bottom_share(g, s)
            row[f"ls_liquid__{name}"] = top_share(liquid, s) - bottom_share(liquid, s)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_cp, cols):
    out = []
    for c in cols:
        s = per_cp[c]
        out.append({
            "strategy": c,
            "mean_excess_per_60d": s.mean(),
            "median": s.median(),
            "share_checkpoints_positive": (s > 0).mean(),
            "worst": s.min(),
            "best": s.max(),
        })
    return pd.DataFrame(out).round(4)


def main():
    with open(ROOT / "best_params.json", encoding="utf-8") as f:
        params = json.load(f)
    df = load_panel()
    cp_median = df.groupby("checkpoint")["excess_return_60d"].transform("median")
    df["beat_checkpoint_median"] = (df["excess_return_60d"] > cp_median).astype(int)
    df["top_tercile"] = (df["rank_target_pct"] > 2 / 3).astype(int)

    oof_path = ROOT / "oof_predictions.csv"
    oof = build_oof(df, params)
    oof.to_csv(oof_path, index=False)
    print(f"Out-of-fold прогнозів: {len(oof)}, checkpoint: {oof['checkpoint'].nunique()}")

    per_cp = per_checkpoint_strategies(oof)
    per_cp.to_csv(ROOT / "strategy_per_checkpoint.csv", index=False)

    print()
    print("Long-only і правила 'або тримай BTC':")
    print(summarize(per_cp, ["universe"] + [f"top20__{n}" for n in SCORES]
                    + ["confident_or_btc", "positive_ev_or_btc"]).to_string(index=False))
    print()
    print("Long-short (верхні 20% мінус нижні 20% того самого checkpoint):")
    print(summarize(per_cp, [f"ls__{n}" for n in SCORES]).to_string(index=False))
    print()
    print("Long-short лише серед ліквідної половини монет (обсяг вище медіани checkpoint):")
    print(summarize(per_cp, [f"ls_liquid__{n}" for n in SCORES]).to_string(index=False))


if __name__ == "__main__":
    main()
