# 在 Cursor 建立「籌碼抓取 Agent」的步驟

## 1. 準備 private 資料倉庫
把 `tw_broker_chip_data/` **裡面的檔案**（不是外層資料夾名）推到你的 private repo，例如 `tw-broker-chip-data` 根目錄。

```bash
# 範例
gh repo create tw-broker-chip-data --private --source=. --remote=origin --push
# 或先建空庫再把本目錄內容 push 上去
```

Repo Secrets 新增：
- `FINMIND_TOKEN` = 你的 FinMind token

並啟用 GitHub Actions。

## 2. 建立 Cursor Cloud Agent（貼提示詞）
1. 開啟 Cursor → Cloud Agents / Automations
2. 連到 **tw-broker-chip-data** 這個 private repo
3. 環境密鑰同樣放 `FINMIND_TOKEN`
4. 排程設 **每天 21:15 Asia/Taipei**（或 cron `15 13 * * 1-5` UTC）
5. 系統提示詞：整份貼上 `docs/AGENT_PROMPT.md`

## 3. 為什麼還要 GitHub Actions？
Cursor Agent 排程偶爾會漏跑；Actions 當保底。
兩者都跑同一支 `scripts/fetch_branch_daily.py`，靠 `state/checkpoint.json` 續跑，不會互相搞亂。

## 4. 流量現實（Sponsor）
- 每小時約 600 次
- 全市場一日約 2100+ 次 → 約 4–5 小時
- 一年回補要多天續跑
- 若升級 SponsorPro，整日一次下載會快很多
