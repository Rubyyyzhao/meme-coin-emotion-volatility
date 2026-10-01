# Data

All series cover 1 January – 31 December 2022.

## `Doge_Sentiment_Master.csv` — post-level scores (56,799 rows)

| Column | Meaning |
|---|---|
| `id` | Post ID within the project dataset |
| `cat` | Poster role assigned by the LLM: Community, Speculation, Celebrity (can overlap, comma-separated) |
| `joy`, `tru`, `fea`, `sur`, `sad`, `ang` | Emotion intensity, 0 = absent, 1 = dominant (joy, trust, fear, surprise, sadness, anger) |
| `date`, `time` | Post timestamp (`date` is day/month/year) |

These are the scores from the first full scoring pass (`notebooks/03_async_labeling_pipeline.ipynb`). The paper's final scores came from `scripts/score_emotions_atf.py`.

## `Quant.csv` — daily market data (365 rows)

DOGE closing price and volume, on-chain transaction count and active addresses, CCI30 crypto index, daily log return, and 30-day and annualised volatility.

## `Master_Final.csv` — merged daily panel (365 × 43)

`Quant.csv` plus: 5-day realised volatility (`volatility_5d`, rolling SD of daily log returns), log-transformed volume and on-chain counts, Google Trends, Crypto Fear & Greed Index, Musk tweet dummy and intensity, and daily emotion means by role (`cele_*`, `comm_*`, `spec_*`) with negative/positive composites.

## Sources

| Data | Source |
|---|---|
| Reddit posts | r/dogecoin, 2022 (raw text not included) |
| Price, returns | Yahoo Finance |
| Volume, transactions, active addresses | CoinMetrics community API |
| Crypto Fear & Greed Index | alternative.me |
| Search interest for "Dogecoin" | Google Trends |
| Musk tweet dummy | Compiled by hand from x.com (13 event dates) |
