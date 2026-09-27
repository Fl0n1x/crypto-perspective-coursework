# -*- coding: utf-8 -*-
"""
Будує сиру панель "монета x тиждень" з денної історії Binance і довідників
CoinGecko. Це основне джерело даних курсової роботи.

Одиниця спостереження: монета в конкретний тиждень (checkpoint - неділя).
Момент прогнозу: закриття торгів у неділю. Усі ознаки рахуються каузальними
ковзними вікнами: значення в день t0 використовує лише дані до t0 включно.
Ціль спостереження - дохідність від close дня t0 до close дня t0 + H днів
(за замовчуванням H = 7), тобто рівно до наступного checkpoint, тому періоди
утримання не перекриваються.

Зіставлення символу Binance з монетою CoinGecko і вибір первинного сектору
виконуються тією самою логікою, що й у build_panel.py (колізії тікерів
розв'язуються за найбільшою капіталізацією).

Запуск: python3 data/raw/build_weekly_panel.py [H]
Результат: data/raw/raw_crypto_weekly_panel.csv.gz
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent
spec = importlib.util.spec_from_file_location("build_panel", RAW / "build_panel.py")
bp = importlib.util.module_from_spec(spec)
sys.modules["build_panel"] = bp
spec.loader.exec_module(bp)

MIN_AGE_DAYS = 98
FIRST_CHECKPOINT = "2021-01-03"


def rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_hist(closes, fast=12, slow=26, signal=9):
    ema_fast = closes.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line - macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()


def bollinger_pctb(closes, window=20, n_std=2):
    mean = closes.rolling(window).mean()
    std = closes.rolling(window).std()
    span = (2 * n_std * std).replace(0, np.nan)
    return (closes - (mean - n_std * std)) / span


def daily_features(df, btc_ret, horizon):
    d = df.asfreq("D")
    close = d["close"].ffill()
    qv = d["quote_volume"]
    ret1 = close.pct_change()
    btc_ret = btc_ret.reindex(d.index)
    vq30 = qv.rolling(30, min_periods=20).mean()
    vq7 = qv.rolling(7, min_periods=5).mean()
    win = dict(window=90, min_periods=60)

    f = pd.DataFrame(index=d.index)
    f["close_t0"] = close
    f["trailing_return_7d"] = close / close.shift(7) - 1
    f["trailing_return_30d"] = close / close.shift(30) - 1
    f["trailing_return_90d"] = close / close.shift(90) - 1
    f["trailing_volatility_7d"] = ret1.rolling(7, min_periods=5).std()
    f["trailing_volatility_30d"] = ret1.rolling(30, min_periods=20).std()
    f["avg_quote_volume_30d"] = vq30
    f["trailing_volume_trend"] = vq30 / vq30.shift(30) - 1
    f["volume_7d_vs_30d"] = vq7 / vq30 - 1
    f["drawdown_from_high_180d"] = close / close.rolling(180, min_periods=1).max() - 1
    f["max_daily_return_30d"] = ret1.rolling(30, min_periods=20).max()
    f["rsi_14"] = rsi(close)
    f["macd_hist_norm"] = macd_hist(close) / close
    f["bollinger_pctb_20"] = bollinger_pctb(close)
    f["btc_corr_90d"] = ret1.rolling(**win).corr(btc_ret)
    f["btc_beta_90d"] = ret1.rolling(**win).cov(btc_ret) / btc_ret.rolling(**win).var()
    f["coin_age_days"] = (d.index - d.index[0]).days

    fwd = close.shift(-horizon) / close - 1
    last_real = d["close"].last_valid_index()
    fwd[d.index + pd.Timedelta(days=horizon) > last_real] = np.nan
    f["forward_return"] = fwd
    return f


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    manifest = bp.load_manifest()
    symbol_to_cgid = bp.build_symbol_to_cgid()
    cgid_to_sector = bp.build_cgid_to_sector()

    btc = bp.load_klines("BTCUSDT")
    btc_ret = btc["close"].asfreq("D").ffill().pct_change()
    btc_f = daily_features(btc, btc_ret, horizon)

    last_day = btc.index.max()
    checkpoints = pd.date_range(FIRST_CHECKPOINT, last_day - pd.Timedelta(days=horizon), freq="W-SUN")
    btc_at = btc_f.loc[checkpoints]
    btc_info = pd.DataFrame({
        "checkpoint": checkpoints,
        "market_regime": np.where(btc_at["trailing_return_90d"] > 0.10, "bull",
                                  np.where(btc_at["trailing_return_90d"] < -0.10, "bear", "sideways")),
        "btc_trailing_return_30d": btc_at["trailing_return_30d"].values,
        "btc_trailing_return_90d": btc_at["trailing_return_90d"].values,
        "btc_rsi_14": btc_at["rsi_14"].values,
        "btc_forward_return": btc_at["forward_return"].values,
    })

    join_audit, frames = [], []
    for m in manifest:
        symbol = m["symbol"]
        if m.get("error") or symbol == "BTCUSDT":
            continue
        base = symbol[:-4].lower()
        cgid = symbol_to_cgid.get(base)
        sector = cgid_to_sector.get(cgid) if cgid else None
        status = ("unresolved_no_cgid" if not cgid
                  else "resolved" if sector else "resolved_no_sector")
        join_audit.append({"symbol": symbol, "base_asset": base, "coingecko_id": cgid,
                           "sector": sector, "status": status})
        if not cgid:
            continue
        df = bp.load_klines(symbol)
        if df is None or len(df) < MIN_AGE_DAYS + horizon:
            continue
        f = daily_features(df, btc_ret, horizon).reindex(checkpoints)
        f = f[(f["coin_age_days"] >= MIN_AGE_DAYS) & f["forward_return"].notna()
              & f["trailing_return_90d"].notna()]
        if f.empty:
            continue
        f = f.rename_axis("checkpoint").reset_index()
        f["symbol"], f["coingecko_id"], f["sector"] = symbol, cgid, sector
        frames.append(f)

    pd.DataFrame(join_audit).to_csv(RAW.parent / "join_audit_weekly.csv", index=False,
                                    encoding="utf-8")

    panel = pd.concat(frames, ignore_index=True).merge(btc_info, on="checkpoint")
    panel = panel[panel["btc_forward_return"].notna()]
    panel["relative_strength_vs_btc_30d"] = panel["trailing_return_30d"] - panel["btc_trailing_return_30d"]
    panel["relative_strength_vs_btc_90d"] = panel["trailing_return_90d"] - panel["btc_trailing_return_90d"]
    panel["excess_return_vs_btc"] = panel["forward_return"] - panel["btc_forward_return"]

    by_cp = panel.groupby("checkpoint")["forward_return"]
    panel["market_forward_return"] = by_cp.transform("median")
    panel["relative_forward_return"] = panel["forward_return"] - panel["market_forward_return"]
    panel["beat_market"] = (panel["relative_forward_return"] > 0).astype(int)

    front = ["symbol", "coingecko_id", "checkpoint", "sector", "market_regime"]
    panel = panel[front + [c for c in panel.columns if c not in front]]
    panel = panel.sort_values(["checkpoint", "symbol"]).reset_index(drop=True)

    n_per_cp = panel.groupby("checkpoint").size()
    print(f"Горизонт {horizon} днів. Рядків: {len(panel)}, монет: {panel['symbol'].nunique()}, "
          f"checkpoint: {panel['checkpoint'].nunique()} "
          f"({panel['checkpoint'].min().date()} - {panel['checkpoint'].max().date()})")
    print(f"Монет на checkpoint: від {n_per_cp.min()} до {n_per_cp.max()}")
    print(f"Частка цілі beat_market: {panel['beat_market'].mean():.4f}")
    print("Режими ринку (частка checkpoint):")
    print(btc_info.set_index("checkpoint").loc[n_per_cp.index, "market_regime"]
          .value_counts(normalize=True).round(3).to_string())
    na = panel.isna().sum()
    print("Пропуски:", na[na > 0].to_dict() or "немає")

    out = RAW / "raw_crypto_weekly_panel.csv.gz"
    panel.to_csv(out, index=False, compression="gzip", encoding="utf-8")
    print("Збережено:", out)


if __name__ == "__main__":
    main()
