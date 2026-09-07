# tw-broker-chip-data

台股**券商分點**日資料快取（FinMind → GitHub），給其他 Cursor / 策略專案共用。

## 為什麼這樣設計

- Cursor 雲端 Agent 天生連 GitHub，資料放這裡最方便共用。
- FinMind Sponsor 約 **600 req/hour**；全市場一日約 2100+ 檔，需分批 + 斷點續跑。
- 整日一次下載（`storage_objects`）需要 **SponsorPro**；腳本會先嘗試，失敗就自動改逐檔抓。

## 目錄

```text
data/daily/YYYY-MM-DD.parquet   # 分點彙總（股數/金額）
state/checkpoint.json           # 續跑狀態
state/latest.json               # 最新日期指標
scripts/fetch_branch_daily.py   # 抓取主程式
docs/AGENT_PROMPT.md            # 給 Cursor Agent 的提示詞
.github/workflows/daily_chip_ingest.yml  # 21:15 台北排程
```

## 本地 / Agent 手動跑

```bash
pip install -r requirements.txt
export FINMIND_TOKEN=你的token
python scripts/fetch_branch_daily.py --mode auto --max-requests 500
```

## GitHub Actions

1. 把本資料夾內容推進你的 private repo（例如 `tw-broker-chip-data`）
2. Repo → Settings → Secrets → Actions，新增 `FINMIND_TOKEN`
3. 啟用 Actions；排程為週一到週五 **21:15 台北時間**
4. 若一輪沒抓完，workflow 會再 dispatch 續跑

## 其他專案怎麼讀（private）

在該專案的 Cursor / Actions Secrets 放一支 **fine-grained PAT**（只授權此 data repo 的 Contents: Read），然後：

```bash
git clone https://x-access-token:${CHIP_DATA_REPO_TOKEN}@github.com/<you>/tw-broker-chip-data.git
```

## 時程現實檢查

| 工作 | 大約 request | 大約時間（520/hour） |
|------|--------------|----------------------|
| 補一日全市場 | ~2,100 | ~4–5 小時 |
| 回補一年 | ~50 萬 | 多天續跑 |

升級 FinMind SponsorPro 後，一日可變 **1 次 request**，會快非常多。
