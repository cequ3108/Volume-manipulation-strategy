# 00720B 短線回測

使用 FinMind `TaiwanStockKBar` 抓取 00720B 近 120 個交易日分 K，回測做市價差、VWAP 均值回歸、ORB、均線通道等策略。

## 執行

```bash
export FINMIND_TOKEN=your_token
pip install -r requirements.txt
python3 fetch_00720b_kbar.py
python3 run_backtest_00720b.py
```

結果輸出於 `results/`。

## 注意

- 00720B 分 K 為「有成交的分鐘」稀疏資料，回測已 forward-fill 成連續分鐘序列；與真實 tick 撮合仍有落差。
- 未含排隊優先權、部分成交、開盤競價等特殊時段。
- 手續費檔位為假設，請依你的 VIP 實際費率替換 `FEE_PROFILES`。
