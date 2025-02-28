import numpy as np
import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class WilliamsR(BaseAlpha):
    # 參數設置
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2022-01-01"
    END_DATE = "2022-12-26"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_4HOUR
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 根據 KLINE_INTERVAL 設定採樣 K 棒間隔
    WILLIAMS_PERIOD = 14  # Williams %R 計算週期
    OVERBOUGHT_THRESHOLD = -20  # 超買門檻
    OVERSOLD_THRESHOLD = -80    # 超賣門檻

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        自定義所需要的列名

        y_timestamp 為採樣點的時間戳 (Required)
        """
        columns = []

        # 當下欄位
        columns.extend(["timestamp", "price", "is_buy", "williams_r"])

        # 延遲欄位（底線後的名稱需要與 alpha function 內 df 的欄位名稱一致）
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", f"y{i}_high", f"y{i}_low", f"y{i}_williams_r"])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        計算 Williams %R Alpha

        Williams %R 是一個動量指標，用於判斷市場是否處於超買或超賣狀態。
        與 RSI 類似，但計算方式不同：
        - Williams %R 值在 0 到 -100 之間波動
        - Williams %R > -20 通常被視為超買區域
        - Williams %R < -80 通常被視為超賣區域
        
        計算公式：
        Williams %R = (最高價 - 收盤價) / (最高價 - 最低價) × -100
        """
        df = rolling_window_df

        # 初始化 rolling_window_df 欄位
        required_columns = [
            "highest_high",  # N期內最高價
            "lowest_low",    # N期內最低價
            "williams_r",    # Williams %R 值
            "calculated"     # 計算完成標記
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # 計算指標邏輯
        def calculate_row(row, df):
            # 取得當前行索引
            index = row.name
            
            # 如果數據不足計算週期，則僅標記為已計算並返回
            if index < self.WILLIAMS_PERIOD - 1:
                row["calculated"] = True
                return row

            # 計算過去N期的最高價和最低價
            period_slice = df.iloc[max(0, index - self.WILLIAMS_PERIOD + 1):index + 1]
            row["highest_high"] = period_slice["high"].max()
            row["lowest_low"] = period_slice["low"].min()

            # 計算 Williams %R
            if (row["highest_high"] - row["lowest_low"]) != 0:
                row["williams_r"] = ((row["highest_high"] - row["close"]) / 
                                   (row["highest_high"] - row["lowest_low"]) * -100)
            else:
                # 處理分母為零的情況
                row["williams_r"] = -50  # 或其他預設值

            row["calculated"] = True
            return row

        # 僅處理未計算的 row
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # 採樣點邏輯
        if df["calculated"].sum() > self.WILLIAMS_PERIOD:
            # 從超賣區域向上突破
            if (df["williams_r"].iloc[-2] < self.OVERSOLD_THRESHOLD and 
                df["williams_r"].iloc[-1] > self.OVERSOLD_THRESHOLD):
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = True
                new_point["williams_r"] = df["williams_r"].iloc[-1]

                return new_point, df

            # 從超買區域向下跌破
            elif (df["williams_r"].iloc[-2] > self.OVERBOUGHT_THRESHOLD and 
                  df["williams_r"].iloc[-1] < self.OVERBOUGHT_THRESHOLD):
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = False
                new_point["williams_r"] = df["williams_r"].iloc[-1]

                return new_point, df

        return None, df