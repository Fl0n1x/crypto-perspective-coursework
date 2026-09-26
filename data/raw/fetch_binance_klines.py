# -*- coding: utf-8 -*-
"""
Завантажує повну денну історію (klines) для всіх спотових пар USDT з Binance,
які реально придатні для панелі "монета x місяць".

Виключено:
  - стейблкоїни в парі з USDT (ціна завжди ~1, forward return безглуздий);
  - леверидж-токени (UP/DOWN/BULL/BEAR) - це не окремі активи, а похідні
    інструменти з примусовим ребалансуванням;
  - токенізовані акції/ETF (розпізнаються пізніше через звірку з CoinGecko).

Сирі відповіді збережено без жодної обробки - це "raw" рівень датасету.
"""
import json
import time
import urllib.request
from pathlib import Path

BASE = "https://api.binance.com"
OUT_DIR = Path(__file__).parent / "binance_klines"
OUT_DIR.mkdir(exist_ok=True)

STABLE_QUOTE_ASSETS = {
    "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDP", "EUR", "TRY", "GBP",
    "AEUR", "USDE", "PYUSD", "USD1",
}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def http_get_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  повтор після помилки: {exc}")
            time.sleep(2)


def eligible_symbols():
    info = http_get_json(f"{BASE}/api/v3/exchangeInfo")
    symbols = []
    for s in info["symbols"]:
        if s["quoteAsset"] != "USDT":
            continue
        if s["status"] != "TRADING" or not s["isSpotTradingAllowed"]:
            continue
        if s["baseAsset"] in STABLE_QUOTE_ASSETS:
            continue
        if s["symbol"].endswith(LEVERAGED_SUFFIXES):
            continue
        symbols.append(s["symbol"])
    return sorted(symbols)


def fetch_full_daily_history(symbol):
    """Пагінація по 1000 свічок за раз, від найранішої доступної дати."""
    all_rows = []
    start_time = 0
    while True:
        url = (f"{BASE}/api/v3/klines?symbol={symbol}&interval=1d"
               f"&limit=1000&startTime={start_time}")
        chunk = http_get_json(url)
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < 1000:
            break
        start_time = chunk[-1][0] + 1
        time.sleep(0.05)
    return all_rows


def main():
    symbols = eligible_symbols()
    print(f"Прийнятних USDT-пар після фільтрів: {len(symbols)}")

    manifest = []
    for i, symbol in enumerate(symbols, start=1):
        out_path = OUT_DIR / f"{symbol}.json"
        if out_path.exists():
            print(f"[{i}/{len(symbols)}] {symbol}: вже завантажено, пропуск")
            continue
        try:
            rows = fetch_full_daily_history(symbol)
        except Exception as exc:
            print(f"[{i}/{len(symbols)}] {symbol}: ПОМИЛКА {exc}")
            manifest.append({"symbol": symbol, "rows": 0, "error": str(exc)})
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        first_day = time.strftime("%Y-%m-%d", time.gmtime(rows[0][0] / 1000)) if rows else None
        last_day = time.strftime("%Y-%m-%d", time.gmtime(rows[-1][0] / 1000)) if rows else None
        print(f"[{i}/{len(symbols)}] {symbol}: {len(rows)} днів, {first_day} -> {last_day}")
        manifest.append({"symbol": symbol, "rows": len(rows),
                          "first_day": first_day, "last_day": last_day})
        time.sleep(0.05)

    with open(OUT_DIR / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print()
    print("Готово. Маніфест:", OUT_DIR / "_manifest.json")


if __name__ == "__main__":
    main()
