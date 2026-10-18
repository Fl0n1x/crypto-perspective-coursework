# -*- coding: utf-8 -*-
"""
Генерує Пашко_coursework.ipynb.

Кожна кодова комірка має мітку metadata["tag"]: за нею tools/render_cells.py
називає скриншот виводу, а tools/docx/ вставляє код комірки текстом у Word.
Структура заголовків повторює рекомендовану структуру Notebook з методички
(20 кроків).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")})


def code(tag, text):
    lines = text.strip("\n").split("\n")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {"tag": tag},
                  "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})


md("""
# Курсова робота: дослідження та підготовка даних для розв'язання задачі машинного навчання

**Предметна область:** криптовалютний ринок.
**Задача:** бінарна класифікація - чи буде дохідність монети за наступні сім днів
вищою за дохідність медіанної монети того самого тижня.

Пашко Назарій, СА-32, Львів-2026.
""")

md("## 1. Імпорти, налаштування та SEED")
code("setup", """
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (chi2_contingency, kruskal, mannwhitneyu, pointbiserialr,
                         shapiro, skew)
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 110
KEY = {}  # ключові результати для таблиць у тексті роботи
print("SEED =", SEED)
""")

md("""
## 2. Завантаження сирих даних

Сирий датасет зібрано скриптом `data/raw/build_weekly_panel.py` з денної історії
Binance і довідників CoinGecko. Одиниця спостереження - монета в конкретний тиждень,
момент прогнозу - закриття торгів у неділю, горизонт - сім днів.
""")
code("load", """
RAW_PATH = "data/raw/raw_crypto_weekly_panel.csv.gz"
df_raw = pd.read_csv(RAW_PATH, parse_dates=["checkpoint"])

print("Розмір сирого датасету:", df_raw.shape)
print("Період:", df_raw["checkpoint"].min().date(), "-", df_raw["checkpoint"].max().date())
print("Унікальних монет:", df_raw["symbol"].nunique())
print("Унікальних тижнів:", df_raw["checkpoint"].nunique())
per_week = df_raw.groupby("checkpoint").size()
print("Монет на тиждень: від", per_week.min(), "до", per_week.max())
""")

md("## 3. Початкове технічне профілювання")
code("info", """
df_raw.info()
""")
code("profile", """
profile = pd.DataFrame({
    "тип": df_raw.dtypes.astype(str),
    "пропусків": df_raw.isna().sum(),
    "пропусків, %": (df_raw.isna().mean() * 100).round(2),
    "унікальних": df_raw.nunique(),
})
profile
""")
code("duplicates", """
print("Точних дублікатів рядків:", int(df_raw.duplicated().sum()))
print("Дублікатів пари (symbol, checkpoint):",
      int(df_raw.duplicated(subset=["symbol", "checkpoint"]).sum()))
const_cols = [c for c in df_raw.columns if df_raw[c].nunique(dropna=False) <= 1]
print("Константних стовпців:", const_cols if const_cols else "немає")
print()
print("Розподіл цільової змінної beat_market (лише для вибору способу поділу):")
print(df_raw["beat_market"].value_counts(normalize=True).round(4).to_string())
print("Пропусків у цілі:", int(df_raw["beat_market"].isna().sum()))
""")
code("domain", """
def domain_checks(df):
    num = df.select_dtypes("number")
    return pd.DataFrame([
        {"правило": "rsi_14 в межах 0-100",
         "порушень": int(((df["rsi_14"] < 0) | (df["rsi_14"] > 100)).sum())},
        {"правило": "волатильність невід'ємна",
         "порушень": int((df["trailing_volatility_30d"] < 0).sum())},
        {"правило": "обсяг торгів невід'ємний",
         "порушень": int((df["avg_quote_volume_30d"] < 0).sum())},
        {"правило": "вік монети не менший за 98 днів",
         "порушень": int((df["coin_age_days"] < 98).sum())},
        {"правило": "дохідність не менша за -100 %",
         "порушень": int((df["forward_return"] < -1).sum())},
        {"правило": "просадка в межах від -100 % до 0",
         "порушень": int(((df["drawdown_from_high_180d"] < -1)
                          | (df["drawdown_from_high_180d"] > 0)).sum())},
        {"правило": "числові значення скінченні",
         "порушень": int(np.isinf(num).sum().sum())},
    ])


domain_checks(df_raw)
""")

code("breaks", """
breaks = pd.read_csv("data/price_breaks_weekly.csv")
print("Паузи в торгах на Binance тривалістю 3 і більше днів:")
print(breaks.rename(columns={"останній день до паузи": "до паузи", "відновлення торгів": "відновлення",
                             "днів без торгів": "днів"}).to_string(index=False))
in_window = df_raw["missing_days_in_window"] >= 3
print()
print("Записів, у вікні ознак або цілі яких була пауза:", int(in_window.sum()))
print(df_raw.loc[in_window, "symbol"].value_counts().to_string())
volume_na = df_raw[["avg_quote_volume_30d", "trailing_volume_trend", "volume_7d_vs_30d"]].isna().any(axis=1)
print()
print("Записів з пропусками в ознаках обсягу:", int(volume_na.sum()),
      "| з них у вікні з паузою в торгах:", int((volume_na & in_window).sum()))
""")

md("## 4. Структурне очищення")
code("audit", """
build_audit = pd.read_csv("data/build_audit_weekly.csv")
print("Записи, вилучені під час побудови панелі з первинних даних:")
for _, r in build_audit.iterrows():
    print(f"{r['кількість']:>6} {r['одиниця']:<8} {r['етап']}")
print()
join_audit = pd.read_csv("data/join_audit_weekly.csv")
unresolved = join_audit[join_audit["status"] == "unresolved_no_cgid"]
print("Зіставлення символів Binance з CoinGecko:")
print(join_audit["status"].value_counts().to_string())
print("Приклади незіставлених символів:", ", ".join(unresolved["symbol"].head(5)))
""")
code("cleaning", """
df_clean = df_raw.copy()
sector_missing_before = int(df_clean["sector"].isna().sum())
df_clean["sector"] = df_clean["sector"].fillna("Other")
rows_before = len(df_clean)
df_clean = df_clean[df_clean["beat_market"].notna()]
gap_rows = df_clean["missing_days_in_window"] >= 3
df_clean = df_clean[~gap_rows]
luna_rows = df_clean["symbol"] == "LUNAUSDT"
df_clean = df_clean[~luna_rows].copy()
df_clean["symbol"] = df_clean["symbol"].str.upper().str.strip()

cleaning_log = pd.DataFrame([
    {"проблема": "сектор не визначений", "виявлення": "sector.isna()",
     "дія": "явна категорія Other", "записів": sector_missing_before},
    {"проблема": "невідома ціль", "виявлення": "beat_market.isna()",
     "дія": "вилучення до поділу", "записів": rows_before - len(df_raw.dropna(subset=["beat_market"]))},
    {"проблема": "пауза в торгах у вікні ознак або цілі", "виявлення": "missing_days_in_window >= 3",
     "дія": "вилучення до поділу", "записів": int(gap_rows.sum())},
    {"проблема": "тикер двох різних активів", "виявлення": "symbol = LUNAUSDT",
     "дія": "вилучення символу", "записів": int(luna_rows.sum())},
    {"проблема": "монета молодша за 98 днів", "виявлення": "coin_age_days < 98",
     "дія": "вилучено під час побудови", "записів": int(build_audit["кількість"].iloc[2])},
    {"проблема": "символ без монети CoinGecko", "виявлення": "status = unresolved_no_cgid",
     "дія": "вилучено під час побудови", "записів": len(unresolved)},
    {"проблема": "формат тікера", "виявлення": "регістр і пробіли",
     "дія": "верхній регістр, без пробілів", "записів": 0},
])
cleaning_log
""")

md("## 5. Контроль після структурного очищення")
code("control", """
print("Схема збережена:", list(df_clean.columns) == list(df_raw.columns))
print("Типи збережені:", (df_clean.dtypes == df_raw.dtypes).all())
print("Точних дублікатів:", int(df_clean.duplicated().sum()))
print("Рядків до / після:", len(df_raw), "/", len(df_clean))
print("Пропусків sector до / після:", int(df_raw["sector"].isna().sum()), "/",
      int(df_clean["sector"].isna().sum()))
print("Пропусків у цілі:", int(df_clean["beat_market"].isna().sum()),
      "| значення цілі:", sorted(df_clean["beat_market"].unique()))
print("Порушень доменних правил:", int(domain_checks(df_clean)["порушень"].sum()))
print("Записів з паузою в торгах у вікні:", int((df_clean["missing_days_in_window"] >= 3).sum()),
      "| записів LUNAUSDT:", int((df_clean["symbol"] == "LUNAUSDT").sum()))
print()
print("Категорії market_regime:", sorted(df_clean["market_regime"].unique()))
print("Кількість секторів:", df_clean["sector"].nunique())
print(df_clean["sector"].value_counts().to_string())
""")
code("save_clean", """
df_clean.to_csv("data/structurally_cleaned/structurally_cleaned_weekly_panel.csv.gz",
                index=False, compression="gzip", encoding="utf-8")
na_left = df_clean.isna().sum()
print("Збережено структурно очищений датасет:", df_clean.shape)
print("Пропуски, залишені для оброблення всередині конвеєра:")
print(na_left[na_left > 0].to_string())
""")

md("""
## 6. Формування train/test

Дані часові й групові. Часовий поділ має пріоритет: усі тижні train передують усім
тижням test, межу обрано так, щоб у train потрапило 80 % записів.
""")
code("split", """
rows_per_week = df_clean.groupby("checkpoint").size().sort_index()
share_cum = rows_per_week.cumsum() / len(df_clean)
n_train_weeks = int((share_cum < 0.80).sum())
split_week = rows_per_week.index[n_train_weeks - 1]

train_df = df_clean[df_clean["checkpoint"] <= split_week].reset_index(drop=True)
test_df = df_clean[df_clean["checkpoint"] > split_week].reset_index(drop=True)

for name, part in [("train", train_df), ("test", test_df)]:
    print(f"{name}: {len(part)} рядків ({len(part) / len(df_clean) * 100:.1f} %), "
          f"{part['checkpoint'].nunique()} тижнів, "
          f"{part['checkpoint'].min().date()} - {part['checkpoint'].max().date()}")
""")
code("split_checks", """
print("Перетин тижнів:", len(set(train_df["checkpoint"]) & set(test_df["checkpoint"])))
pairs_train = set(zip(train_df["symbol"], train_df["checkpoint"]))
pairs_test = set(zip(test_df["symbol"], test_df["checkpoint"]))
print("Перетин пар (монета, тиждень):", len(pairs_train & pairs_test))
print("Останнє цільове вікно train закінчується:",
      (split_week + pd.Timedelta(days=7)).date(),
      "| перший прогноз test:", test_df["checkpoint"].min().date())
print()
print("Частка класу 1: train", round(train_df["beat_market"].mean(), 4),
      "| test", round(test_df["beat_market"].mean(), 4))
print()
print("Ринкові режими (кількість тижнів):")
print(pd.DataFrame({
    "train": train_df.groupby("checkpoint")["market_regime"].first().value_counts(),
    "test": test_df.groupby("checkpoint")["market_regime"].first().value_counts(),
}).fillna(0).astype(int).to_string())
""")

md("""
## 7. Статистичне дослідження train

Тут і до кроку 18 використовується **тільки train**.
""")
code("describe", """
COIN_FEATURES = [
    "trailing_return_7d", "trailing_return_30d", "trailing_return_90d",
    "trailing_volatility_7d", "trailing_volatility_30d", "avg_quote_volume_30d",
    "trailing_volume_trend", "volume_7d_vs_30d", "drawdown_from_high_180d",
    "max_daily_return_30d", "rsi_14", "macd_hist_norm", "bollinger_pctb_20",
    "btc_corr_90d", "btc_beta_90d", "coin_age_days",
]
MARKET_FEATURES = ["btc_trailing_return_30d", "btc_trailing_return_90d", "btc_rsi_14"]
CAT_FEATURES = ["sector", "market_regime"]

print("Записів у train:", len(train_df), "| обсяг торгів подано в мільйонах доларів")
desc = train_df[COIN_FEATURES].describe().T.drop(columns="count")
desc.loc["avg_quote_volume_30d"] /= 1e6
desc.style.format("{:.3f}")
""")
code("skew", """
print("Асиметрія числових ознак (train):")
print(train_df[COIN_FEATURES].apply(lambda s: skew(s.dropna())).round(1)
      .sort_values().to_string())
print()
print("Розподіл цілі в train:",
      train_df["beat_market"].value_counts(normalize=True).round(4).to_dict())
print("Частоти ринкових режимів (рядки):",
      train_df["market_regime"].value_counts(normalize=True).round(3).to_dict())
""")
code("dist_plots", """
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
for ax, col, title, lim, color in [
        (axes[0], "forward_return", "Дохідність за 7 днів", (-0.5, 0.5), "#2a78d6"),
        (axes[1], "relative_forward_return", "Відносно медіани тижня", (-0.5, 0.5), "#eb6834"),
        (axes[2], "trailing_volatility_30d", "Волатильність, 30 днів", (0, 0.2), "#1baf7a")]:
    sns.histplot(train_df[col].clip(*lim), bins=50, ax=ax, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.tight_layout()
plt.show()
""")
code("sector_ct", """
ct = pd.crosstab(train_df["sector"], train_df["beat_market"], normalize="index").mul(100).round(1)
ct.columns = ["не випередила, %", "випередила, %"]
ct["записів"] = train_df["sector"].value_counts()
ct.sort_values("випередила, %", ascending=False)
""")

md("## 8. Аналіз пропусків і потенційних викидів на train")
code("missing", """
model_cols = COIN_FEATURES + MARKET_FEATURES + CAT_FEATURES
na_train = train_df[model_cols].isna().sum()
print("Пропусків у ознаках train після структурного очищення:", int(na_train.sum()))
print("Ознак із пропусками:", list(na_train[na_train > 0].index) or "немає")
print()
print("Пропуски сирих даних (крок 3) містились лише у записах з паузою в торгах")
print("і були вилучені разом із ними на кроці 4. Механізм - MAR: пропуск повністю")
print("пояснюється спостережуваною змінною missing_days_in_window, а не значенням ознаки.")
""")
code("outliers", """
def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


OUTLIER_COLS = ["trailing_return_30d", "trailing_volatility_30d", "max_daily_return_30d",
                "forward_return"]
rows = []
for col in OUTLIER_COLS:
    lo, hi = iqr_bounds(train_df[col].dropna())
    out = int(((train_df[col] < lo) | (train_df[col] > hi)).sum())
    rows.append({"ознака": col, "нижня межа": round(lo, 3), "верхня межа": round(hi, 3),
                 "за межами": out, "частка, %": round(out / len(train_df) * 100, 2)})
print(pd.DataFrame(rows).to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.4))
for ax, col in zip(axes, OUTLIER_COLS):
    ax.boxplot(train_df[col].dropna(), widths=0.5, flierprops={"markersize": 2})
    ax.set_title(col.replace("trailing_", "").replace("_", " "), fontsize=8)
    ax.set_xticks([])
plt.tight_layout()
plt.show()
""")
code("extremes", """
extreme = train_df.nlargest(5, "forward_return")[
    ["symbol", "checkpoint", "sector", "trailing_volatility_30d", "forward_return"]]
extreme["checkpoint"] = extreme["checkpoint"].dt.date
print("П'ять найбільших тижневих дохідностей у train:")
print(extreme.round(3).to_string(index=False))
print()
print("Медіанна волатильність train:", round(train_df["trailing_volatility_30d"].median(), 4))
""")

md("""
**Генеральна сукупність і вибірка.** Генеральна сукупність - усі криптовалюти, що в
певний тиждень торгувалися на публічних біржах. Вибірка - 383 монети з парою до USDT
на Binance, що торгуються у вересні 2026 року. Основні зміщення: виживання (немає
делістованих монет), відбір біржею за ліквідністю, одна біржа.
""")

md("""
## 9. Перевірка статистичних гіпотез на train

**Гіпотеза 1.** H0: розподіл 30-денної волатильності однаковий для монет, що
випередили ринок, і тих, що не випередили. H1: розподіли відрізняються.
Критерій Манна-Вітні, alpha = 0,05.
""")
code("h1", """
vol_win = train_df.loc[train_df["beat_market"] == 1, "trailing_volatility_30d"].dropna()
vol_lose = train_df.loc[train_df["beat_market"] == 0, "trailing_volatility_30d"].dropna()

sw_p = shapiro(vol_win.sample(5000, random_state=SEED)).pvalue
print(f"Передумова: Шапіро-Вілк для вибірки 5000, p = {sw_p:.1e} - розподіл не нормальний,")
print("тому застосовано непараметричний критерій Манна-Вітні.")
u_stat, p_h1 = mannwhitneyu(vol_win, vol_lose, alternative="two-sided")
rb_h1 = 2 * u_stat / (len(vol_win) * len(vol_lose)) - 1
print(f"n = {len(vol_win)} / {len(vol_lose)}, медіани {vol_win.median():.4f} / {vol_lose.median():.4f}")
print(f"U = {u_stat:.0f}, p-value = {p_h1:.2e}, rank-biserial = {rb_h1:.4f}")
print("Висновок:", "H0 відхиляється" if p_h1 < 0.05 else "немає підстав відхилити H0")

coin = train_df.groupby("symbol").agg(vol=("trailing_volatility_30d", "median"),
                                      win=("beat_market", "mean"), n=("beat_market", "size"))
coin = coin[coin["n"] >= 20]
u_c, p_h1_coin = mannwhitneyu(coin.loc[coin["win"] > 0.5, "vol"],
                              coin.loc[coin["win"] <= 0.5, "vol"], alternative="two-sided")
print()
print(f"Перевірка стійкості (одна точка на монету, {len(coin)} монет): p-value = {p_h1_coin:.2e}")
KEY["h1"] = dict(U=float(u_stat), p=float(p_h1), effect=float(rb_h1), p_coin=float(p_h1_coin),
                 n_coin=int(len(coin)))
""")

md("""
**Гіпотеза 2.** H0: сектор монети і подія «монета випередила ринок» незалежні.
H1: між ними є зв'язок. Критерій хі-квадрат, alpha = 0,05.
""")
code("h2", """
contingency = pd.crosstab(train_df["sector"], train_df["beat_market"])
chi2, p_h2, dof, expected = chi2_contingency(contingency)
v_h2 = np.sqrt(chi2 / (contingency.values.sum() * (min(contingency.shape) - 1)))
print(f"Передумова: мінімальна очікувана частота {expected.min():.1f} (потрібно не менше 5)")
print(f"хі-квадрат = {chi2:.2f}, df = {dof}, p-value = {p_h2:.2e}, V Крамера = {v_h2:.4f}")
print("Висновок:", "H0 відхиляється" if p_h2 < 0.05 else "немає підстав відхилити H0")

coin_sector = train_df.groupby("symbol").agg(sector=("sector", "first"),
                                             win=("beat_market", "mean"),
                                             n=("beat_market", "size"))
coin_sector = coin_sector[coin_sector["n"] >= 20]
groups = [g["win"].values for _, g in coin_sector.groupby("sector") if len(g) >= 5]
h_kw, p_h2_coin = kruskal(*groups)
print()
print(f"Перевірка стійкості (одна точка на монету, Краскел-Уолліс, {len(groups)} секторів, "
      f"{sum(len(g) for g in groups)} монет):")
print(f"H = {h_kw:.2f}, p-value = {p_h2_coin:.2e}")
KEY["h2"] = dict(chi2=float(chi2), dof=int(dof), p=float(p_h2), effect=float(v_h2),
                 p_coin=float(p_h2_coin), n_sectors=len(groups),
                 n_coin=int(sum(len(g) for g in groups)))
""")

md("## 10. Аналіз залежностей на train")
code("corr", """
corr = train_df[COIN_FEATURES].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.5, 6.2))
short = [c.replace("trailing_", "").replace("_from_high", "") for c in COIN_FEATURES]
sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
            annot_kws={"size": 6}, xticklabels=short, yticklabels=short, cbar=False, ax=ax)
ax.tick_params(labelsize=7)
plt.tight_layout()
plt.show()
high = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(COIN_FEATURES)
        for b in COIN_FEATURES[i + 1:] if abs(corr.loc[a, b]) > 0.8]
print("Пари з |rho| > 0,8:")
for a, b, r in high:
    print(f"  {a} - {b}: {r}")
""")
code("assoc", """
sample = train_df.sample(20000, random_state=SEED)
num_cols = COIN_FEATURES + MARKET_FEATURES
mi = mutual_info_classif(sample[num_cols].fillna(sample[num_cols].median()),
                         sample["beat_market"], random_state=SEED)
rows = []
for col, mi_val in zip(num_cols, mi):
    s = train_df[[col, "beat_market"]].dropna()
    rows.append({"ознака": col, "r (точк.-бісер.)": round(pointbiserialr(s["beat_market"], s[col])[0], 4),
                 "взаємна інформація": round(mi_val, 4)})
for col in CAT_FEATURES:
    ct2 = pd.crosstab(train_df[col], train_df["beat_market"])
    v = np.sqrt(chi2_contingency(ct2)[0] / (ct2.values.sum() * (min(ct2.shape) - 1)))
    rows.append({"ознака": col, "r (точк.-бісер.)": np.nan, "V Крамера": round(v, 4)})
assoc = pd.DataFrame(rows)
raw_train = df_raw[df_raw["checkpoint"] <= split_week]
r_raw = pointbiserialr(raw_train["beat_market"], raw_train["trailing_volatility_30d"])[0]
print("r волатильності за 30 днів з ціллю на тих самих тижнях до структурного очищення:",
      round(r_raw, 4))
assoc.sort_values("r (точк.-бісер.)", key=abs, ascending=False, na_position="last")
""")

md("""
## 11. Перевірка витоку

Момент прогнозу - закриття торгів у неділю. Для кожного стовпця вказано, чи відомий
він на цей момент, ризик витоку та рішення.
""")
code("leakage", """
FUTURE = {"forward_return": "дохідність після моменту прогнозу",
          "btc_forward_return": "дохідність BTC після моменту прогнозу",
          "excess_return_vs_btc": "різниця майбутніх дохідностей",
          "market_forward_return": "медіана майбутніх дохідностей тижня",
          "relative_forward_return": "пряма похідна цілі"}
SERVICE = {"symbol": "ідентифікатор", "coingecko_id": "ідентифікатор",
           "checkpoint": "часова мітка", "close_t0": "рівень ціни без змісту для порівняння"}
rows = []
for col in df_clean.columns:
    if col == "beat_market":
        rows.append((col, "-", "-", "ціль", "цільова змінна"))
    elif col in FUTURE:
        rows.append((col, "ні", "прямий", "вилучити", FUTURE[col]))
    elif col == "missing_days_in_window":
        rows.append((col, "ні", "майбутнє", "вилучити", "індикатор якості, охоплює 7 днів після прогнозу"))
    elif col in SERVICE:
        rows.append((col, "так", "немає", "вилучити", SERVICE[col]))
    elif col == "sector":
        rows.append((col, "так", "низький", "залишити", "довідник 2026 р., сектор рідко змінюється"))
    else:
        rows.append((col, "так", "немає", "залишити", "ковзне вікно до моменту прогнозу"))
leakage = pd.DataFrame(rows, columns=["стовпець", "доступна", "ризик", "рішення", "обґрунтування"])
print(leakage["рішення"].value_counts().to_string())
leakage
""")

md("""
## 12. Конструювання ознак

Три групи ознак, жодна з них не використовує цільову змінну чи майбутні ціни.
""")
code("engineer", """
def engineer_features(d):
    d = d.copy()
    d["avg_quote_volume_30d"] = np.log1p(d["avg_quote_volume_30d"])
    d["relative_strength_vs_btc_90d"] = d["trailing_return_90d"] - d["btc_trailing_return_90d"]
    median_vol = d.groupby("checkpoint")["trailing_volatility_30d"].transform("median")
    d["relative_volatility_vs_market"] = d["trailing_volatility_30d"] / median_vol
    grouped = d.groupby("checkpoint")
    for col in COIN_FEATURES:
        d[f"{col}_xs"] = (grouped[col].rank(pct=True) - 0.5) * np.sqrt(12)
    return d


train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)
CONSTRUCTED = (["relative_strength_vs_btc_90d", "relative_volatility_vs_market"]
               + [f"{c}_xs" for c in COIN_FEATURES])
print("Створено ознак:", len(CONSTRUCTED))
train_fe[["relative_strength_vs_btc_90d", "relative_volatility_vs_market",
          "trailing_volatility_30d_xs"]].describe().T.drop(columns="count").style.format("{:.3f}")
""")
code("usefulness", """
rows = []
for col in ["relative_strength_vs_btc_90d", "relative_volatility_vs_market",
            "trailing_volatility_30d_xs", "max_daily_return_30d_xs", "coin_age_days_xs"]:
    s = train_fe[[col, "beat_market"]].dropna()
    r, p = pointbiserialr(s["beat_market"], s[col])
    auc = roc_auc_score(s["beat_market"], s[col])
    rows.append({"ознака": col, "r": round(r, 4), "ROC-AUC ознаки": round(max(auc, 1 - auc), 4),
                 "p-value": f"{p:.1e}"})
pd.DataFrame(rows)
""")

md("## 13. Формування базового та фінального наборів ознак")
code("feature_sets", """
BASE_NUM = COIN_FEATURES + MARKET_FEATURES
FINAL_NUM = BASE_NUM + CONSTRUCTED
BASE_FEATURES, FINAL_FEATURES = BASE_NUM + CAT_FEATURES, FINAL_NUM + CAT_FEATURES

top_share = train_fe[FINAL_FEATURES].apply(lambda s: s.value_counts(normalize=True).iloc[0])
print("Майже константних ознак (одне значення > 95 %):", list(top_share[top_share > 0.95].index))
print(f"Базовий набір: {len(BASE_FEATURES)} ознак ({len(BASE_NUM)} числових, 2 категоріальні)")
print(f"Фінальний набір: {len(FINAL_FEATURES)} ознак ({len(FINAL_NUM)} числових, 2 категоріальні)")
print("Вилучено:", len(FUTURE) + 1, "витокових і", len(SERVICE), "службових стовпців")

analytical = pd.concat([train_fe.assign(part="train"), test_fe.assign(part="test")])[
    ["symbol", "checkpoint", "part"] + FINAL_FEATURES + ["beat_market"]]
analytical.to_csv("data/analytical/analytical_weekly_panel.csv.gz", index=False,
                  compression="gzip", encoding="utf-8")
print("Аналітичний датасет:", analytical.shape)
""")

md("""
## 14. Базовий конвеєр

HistGradientBoostingClassifier: змішані типи ознак, без масштабування, стійкий до
викидів, нелінійні залежності. Однакові гіперпараметри в обох конвеєрах. Вбудована
рання зупинка (`early_stopping="auto"`) відбирає 10 % рядків тренувальної частини
кожного фолду як внутрішню валідацію; test у ній не бере участі.
""")
code("base_pipeline", """
MODEL_PARAMS = dict(random_state=SEED, learning_rate=0.05, max_iter=300,
                    max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                    early_stopping="auto", validation_fraction=0.1)


def make_pipeline(numeric_features):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor),
                     ("model", HistGradientBoostingClassifier(**MODEL_PARAMS))])


base_pipeline = make_pipeline(BASE_NUM)
base_pipeline
""")

md("## 15. Фінальний конвеєр")
code("final_pipeline", """
final_pipeline = make_pipeline(FINAL_NUM)
print("Той самий алгоритм і гіперпараметри; відмінність - сконструйовані ознаки:")
print("  базовий:", len(BASE_NUM), "числових ознак")
print("  фінальний:", len(FINAL_NUM), "числових ознак")
""")

md("""
## 16. Крос-валідація обох конвеєрів на train

Схема для часових даних (аналог TimeSeriesSplit): розширюване вікно за тижнями,
5 фолдів, без перемішування, embargo 1 тиждень. Фолди однакові для обох конвеєрів.
""")
code("cv", """
def time_folds(data, n_folds=5, embargo=1):
    weeks = sorted(data["checkpoint"].unique())
    start, step = len(weeks) // 2, (len(weeks) - len(weeks) // 2) // n_folds
    for i in range(n_folds):
        s = start + i * step
        valid = weeks[s:s + step] if i < n_folds - 1 else weeks[s:]
        yield weeks[:s - embargo], valid


def cross_validate(numeric_features):
    rows = []
    for i, (tr_w, va_w) in enumerate(time_folds(train_fe), start=1):
        tr = train_fe[train_fe["checkpoint"].isin(tr_w)]
        va = train_fe[train_fe["checkpoint"].isin(va_w)]
        pipe = make_pipeline(numeric_features).fit(tr[numeric_features + CAT_FEATURES],
                                                   tr["beat_market"])
        proba = pipe.predict_proba(va[numeric_features + CAT_FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y = va["beat_market"]
        rows.append({"фолд": i, "з": va_w[0].date(), "по": va_w[-1].date(),
                     "ROC-AUC": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
                     "bal. acc.": balanced_accuracy_score(y, pred),
                     "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                     "F1": f1_score(y, pred)})
    return pd.DataFrame(rows)


cv_base = cross_validate(BASE_NUM)
cv_final = cross_validate(FINAL_NUM)
print("Базовий конвеєр:")
print(cv_base.round(4).to_string(index=False))
print("Фінальний конвеєр:")
print(cv_final.round(4).to_string(index=False))
""")

md("## 17. Порівняння результатів")
code("cv_compare", """
better = int((cv_final["ROC-AUC"] > cv_base["ROC-AUC"]).sum())
summary_cv = pd.DataFrame([
    {"конвеєр": name, "ROC-AUC, середнє": cv["ROC-AUC"].mean(), "ROC-AUC, std": cv["ROC-AUC"].std(),
     "accuracy": cv["accuracy"].mean(), "bal. acc.": cv["bal. acc."].mean(),
     "precision": cv["precision"].mean(), "recall": cv["recall"].mean(), "F1": cv["F1"].mean()}
    for name, cv in [("базовий", cv_base), ("фінальний", cv_final)]]).round(4)
summary_cv["висновок"] = ["точка відліку", f"обрано: кращий у {better} з 5 фолдів"]
KEY["cv"] = summary_cv.to_dict("records")
summary_cv
""")

md("## 18. Навчання фінального конвеєра на всьому train")
code("final_fit", """
X_train, y_train = train_fe[FINAL_FEATURES], train_fe["beat_market"]
final_pipeline.fit(X_train, y_train)
print("Фінальний конвеєр навчено на", len(X_train), "записах train.")
print("Ітерацій бустингу після ранньої зупинки:", final_pipeline["model"].n_iter_)
""")

md("## 19. Одноразове оцінювання на test")
code("test_metrics", """
X_test, y_test = test_fe[FINAL_FEATURES], test_fe["beat_market"]
y_proba = final_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
test_metrics = pd.Series({
    "ROC-AUC": roc_auc_score(y_test, y_proba), "accuracy": accuracy_score(y_test, y_pred),
    "balanced accuracy": balanced_accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred), "recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred)}).round(4)
KEY["test"] = test_metrics.to_dict()
print(test_metrics.to_string())
""")
code("confusion", """
fig, ax = plt.subplots(figsize=(3.6, 3.0))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["0", "1"], yticklabels=["0", "1"], ax=ax)
ax.set_xlabel("прогноз")
ax.set_ylabel("факт")
plt.tight_layout()
plt.show()
print(classification_report(y_test, y_pred, target_names=["не випередила", "випередила"]))
""")
code("errors", """
test_eval = test_fe.assign(correct=(y_pred == y_test).values, proba=y_proba)
by_regime = test_eval.groupby("market_regime").apply(lambda g: pd.Series({
    "ROC-AUC": roc_auc_score(g["beat_market"], g["proba"]), "точність": g["correct"].mean(),
    "записів": len(g)}))
by_sector = test_eval.groupby("sector")["correct"].agg(точність="mean", записів="count")
print("За ринковим режимом:")
print(by_regime.round(4).to_string())
print()
print("Сектори з найнижчою і найвищою точністю:")
print(pd.concat([by_sector.nsmallest(3, "точність"), by_sector.nlargest(3, "точність")])
      .round(4).to_string())
""")
code("summary", """
dummy = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(X_train, y_train)
base_pipeline.fit(train_fe[BASE_FEATURES], y_train)
base_test_auc = roc_auc_score(y_test, base_pipeline.predict_proba(test_fe[BASE_FEATURES])[:, 1])
stages = pd.DataFrame({
    "етап": ["наївний класифікатор", "базовий, крос-валідація", "фінальний, крос-валідація",
             "базовий, test (довідково)", "фінальний, test"],
    "ROC-AUC": [roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1]),
                cv_base["ROC-AUC"].mean(), cv_final["ROC-AUC"].mean(),
                base_test_auc, test_metrics["ROC-AUC"]]}).round(4)
KEY["base_test_auc"] = float(base_test_auc)
stages
""")
code("practical", """
weekly = []
for cp, g in test_eval.groupby("checkpoint"):
    k = max(1, round(len(g) * 0.2))
    weekly.append({"checkpoint": cp, "топ-20 %": g.nlargest(k, "proba")["forward_return"].mean(),
                   "нижні 20 %": g.nsmallest(k, "proba")["forward_return"].mean(),
                   "медіанна монета": g["forward_return"].median(),
                   "BTC": g["btc_forward_return"].iloc[0]})
weekly = pd.DataFrame(weekly).set_index("checkpoint")
weekly["топ мінус нижні"] = weekly["топ-20 %"] - weekly["нижні 20 %"]
print(f"Тестовий період: {len(weekly)} тижнів. Середня дохідність за тиждень, %:")
print((weekly.mean() * 100).round(2).to_string())
print("Частка тижнів у плюсі:")
print((weekly > 0).mean().round(3).to_string())
""")
code("equity", """
equity = (1 + weekly[["топ-20 %", "медіанна монета", "BTC", "топ мінус нижні"]]).cumprod()
fig, ax = plt.subplots(figsize=(7.5, 3.0))
for col, color in zip(equity.columns, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]):
    ax.plot(equity.index, equity[col], label=col, color=color, linewidth=1.8)
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("капітал, разів")
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
print("Підсумок за тестовий період, разів:")
print("; ".join(f"{k} {v:.3f}" for k, v in equity.iloc[-1].items()))
""")
code("insample", """
in_sample_auc = roc_auc_score(y_train, final_pipeline.predict_proba(X_train)[:, 1])
KEY["in_sample_auc"] = float(in_sample_auc)
print("ROC-AUC на train (модель бачила ці дані):", round(in_sample_auc, 4))
print("ROC-AUC на test (чесна оцінка):         ", test_metrics["ROC-AUC"])
with open("results/key_numbers.json", "w", encoding="utf-8") as f:
    json.dump(KEY, f, ensure_ascii=False, indent=1, default=float)
""")

md("""
## 20. Підсумкові висновки

Фінальний конвеєр перевершив базовий на крос-валідації й на test; результат на test не
поступається крос-валідації, тобто ознак перенавчання чи витоку немає. Ключові числа
збережено в `results/key_numbers.json`, текст висновків - у курсовій роботі.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
out = ROOT / "Пашко_coursework.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
tags = [c["metadata"]["tag"] for c in cells if c["cell_type"] == "code"]
assert len(tags) == len(set(tags)), "мітки комірок мають бути унікальними"
print("Записано", out.name, "| комірок:", len(cells), "| кодових:", len(tags))
