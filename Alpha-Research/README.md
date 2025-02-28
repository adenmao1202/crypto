## 環境設置

創建並激活 conda 環境：
```bash
conda env create -f environment.yml
conda activate alpha-backtest
```

## 使用說明

1. 主要 alpha 邏輯放置於 `alpha/` 資料夾中
2. 執行 `main.py` 文件，選擇 alpha 策略，即可開始採樣。

## 程式架構
```bash
project/
│
├── alpha/
│   ├── __init__.py
│   ├── macd.py                 # MACD Alpha
│   ├── atr.py                  # ATR Alpha
│   └── custom_strategy.py      # 自定義策略
│
├── src/
│   ├── __init__.py
│   ├── get_kline.py            # 獲取歷史資料
│   └── sampling.py             # 採樣邏輯
│
├── main.py                     # 主程式入口
└── requirements.txt            # 依賴庫
└── environment.yml             # 環境設置
```
