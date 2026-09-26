# -*- coding: utf-8 -*-
"""
Категорії (сектори) монет масовим запитом, а не по одній.

CoinGecko /coins/{id} з окремим запитом на кожну з ~450 монет наштовхнувся б
на той самий ліміт 429, що й markets-запит, і зайняв би години. Замість
цього для кожного вибраного сектору один раз запитуємо
/coins/markets?category=<slug>, який повертає список монет цього сектору
одним викликом. Це ~20 запитів замість ~450.

Обраний перелік секторів - куратований вручну, охоплює основні категорії,
релевантні для оцінки перспективності монети.
"""
import json
import time
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent / "coingecko_categories"
OUT_DIR.mkdir(exist_ok=True)

# slug перевірено проти /coins/categories/list
SECTORS = [
    "layer-1",
    "smart-contract-platform",
    "layer-2",
    "decentralized-finance-defi",
    "meme-token",
    "gaming",
    "metaverse",
    "artificial-intelligence",
    "oracle",
    "centralized-exchange-token-cex",
    "decentralized-exchange",
    "privacy-coins",
    "storage",
    "infrastructure",
    "real-world-assets-rwa",
    "stablecoins",
    "yield-farming",
    "non-fungible-tokens-nft",
    "dog-themed-coins",
    "wrapped-tokens",
]


def http_get_json(url, retries=6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            wait = 20 * (attempt + 1)
            print(f"  {exc} -> чекаю {wait}с і повторюю")
            time.sleep(wait)
    raise RuntimeError(f"не вдалося отримати {url}")


def main():
    for sector in SECTORS:
        out_path = OUT_DIR / f"{sector}.json"
        if out_path.exists():
            print(f"{sector}: вже завантажено, пропуск")
            continue
        url = (f"https://api.coingecko.com/api/v3/coins/markets"
               f"?vs_currency=usd&category={sector}&order=market_cap_desc&per_page=250")
        coins = http_get_json(url)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([{"id": c["id"], "symbol": c["symbol"]} for c in coins], f, ensure_ascii=False)
        print(f"{sector}: {len(coins)} монет")
        time.sleep(15)

    print("Готово, категорії в:", OUT_DIR)


if __name__ == "__main__":
    main()
