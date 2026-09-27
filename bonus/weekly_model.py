# -*- coding: utf-8 -*-
"""
Щотижневий walk-forward на weekly_panel_h{H}.csv.gz.

Перший рік (2021) - лише навчання. Далі кожні 4 тижні модель
перенавчається на всіх тижнях до поточного (експандне вікно) з embargo в
1 тиждень і прогнозує наступні 4 тижні. Кожен тестовий прогноз зроблено
моделлю, яка не бачила ні цього тижня, ні будь-якого пізнішого.

Ціль - "краща за медіанну монету свого тижня" (раунд 3 показав, що рух
усього альт-ринку проти BTC з ознак монети не передбачити). Дві моделі
на різних представленнях ознак (сирі значення і z-score всередині тижня),
результат - сума їхніх процентильних рангів. Варіант "_w" додатково
дає свіжим тижням більшу вагу (період напіврозпаду 52 тижні), бо в
раунді 3 перевага слабшала з часом.

Гіперпараметри фіксовані заздалегідь (помірна регуляризація під ~60 тис.
рядків), на тестових даних нічого не підбиралось.

Запуск: python3 bonus/weekly_model.py [H]
Результат: bonus/weekly_oof_h{H}.csv.gz
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
SEED = 42
INITIAL_TRAIN_WEEKS = 52
RETRAIN_EVERY = 4
EMBARGO_WEEKS = 1
HALF_LIFE_WEEKS = 52

RAW_FEATURES = [
    "trailing_return_7d", "trailing_return_30d", "trailing_return_90d",
    "trailing_volatility_7d", "trailing_volatility_30d", "avg_quote_volume_30d",
    "trailing_volume_trend", "volume_7d_vs_30d", "drawdown_from_high_180d",
    "max_daily_return_30d", "rsi_14", "macd_hist_12_26_9", "bollinger_pctb_20",
    "btc_corr_90d", "btc_beta_90d", "coin_age_days",
    "relative_strength_vs_btc_90d", "relative_strength_vs_btc_30d",
]
MARKET_FEATURES = ["btc_trailing_return_30d", "btc_trailing_return_90d", "btc_rsi_14"]
CAT_FEATURES = ["sector", "market_regime"]
XS_FEATURES = [f"{c}_xs" for c in RAW_FEATURES]
TARGET = "beat_checkpoint_median"

MODEL_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                    min_samples_leaf=200, l2_regularization=1.0, random_state=SEED)


def load(horizon):
    df = pd.read_csv(ROOT / f"weekly_panel_h{horizon}.csv.gz", parse_dates=["checkpoint"])
    df["avg_quote_volume_30d"] = np.log1p(df["avg_quote_volume_30d"])
    grouped = df.groupby("checkpoint")
    for c in RAW_FEATURES:
        # ранговий z-score стійкіший за звичайний до мем-пампів у хвостах
        pct = grouped[c].rank(pct=True)
        df[f"{c}_xs"] = (pct - 0.5) * np.sqrt(12)
    return df


def make_model(numeric):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(**MODEL_PARAMS))])


def fit_predict(train, test, numeric, weights=None):
    pipe = make_model(numeric)
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    pipe.fit(train[numeric + CAT_FEATURES], train[TARGET], **fit_kwargs)
    return pipe.predict_proba(test[numeric + CAT_FEATURES])[:, 1]


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    df = load(horizon)
    weeks = sorted(df["checkpoint"].unique())
    raw_numeric = RAW_FEATURES + MARKET_FEATURES
    xs_numeric = XS_FEATURES + MARKET_FEATURES

    chunks = []
    started = time.time()
    blocks = range(INITIAL_TRAIN_WEEKS, len(weeks), RETRAIN_EVERY)
    for i, start in enumerate(blocks):
        test_weeks = weeks[start:start + RETRAIN_EVERY]
        train_weeks = weeks[:start - EMBARGO_WEEKS]
        last_train = train_weeks[-1]
        assert last_train + pd.Timedelta(days=horizon) <= test_weeks[0], "forward-вікно train заходить у test"
        train = df[df["checkpoint"].isin(train_weeks)]
        test = df[df["checkpoint"].isin(test_weeks)].copy()

        age_weeks = (last_train - train["checkpoint"]).dt.days / 7
        w = 0.5 ** (age_weeks / HALF_LIFE_WEEKS)
        test["p_raw"] = fit_predict(train, test, raw_numeric)
        test["p_xs"] = fit_predict(train, test, xs_numeric)
        test["p_raw_w"] = fit_predict(train, test, raw_numeric, w.values)
        test["p_xs_w"] = fit_predict(train, test, xs_numeric, w.values)
        chunks.append(test[["checkpoint", "symbol", "sector", "market_regime", "avg_quote_volume_30d",
                            "forward_return", "btc_forward_return", "excess_return",
                            "trailing_volatility_30d", "coin_age_days",
                            "p_raw", "p_xs", "p_raw_w", "p_xs_w"]])
        if i % 10 == 0:
            print(f"блок {i + 1}/{len(blocks)}: test з {test_weeks[0].date()}, "
                  f"train {len(train)} рядків, {time.time() - started:.0f} с", flush=True)

    oof = pd.concat(chunks, ignore_index=True)
    g = oof.groupby("checkpoint")
    oof["score"] = g["p_raw"].rank(pct=True) + g["p_xs"].rank(pct=True)
    oof["score_w"] = g["p_raw_w"].rank(pct=True) + g["p_xs_w"].rank(pct=True)
    out = ROOT / f"weekly_oof_h{horizon}.csv.gz"
    oof.to_csv(out, index=False, compression="gzip")
    print(f"Out-of-fold прогнозів: {len(oof)}, тижнів: {oof['checkpoint'].nunique()} "
          f"({oof['checkpoint'].min().date()} - {oof['checkpoint'].max().date()})")
    print("Збережено:", out)


if __name__ == "__main__":
    main()
