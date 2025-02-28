import numpy as np
import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class StochasticOscillator(BaseAlpha):
    # 參數設置
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2022-01-01"
    END_DATE = "2022-12-31"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_4HOUR
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 根據 KLINE_INTERVAL 設定採樣 K 棒間隔
    
    # KD 指標參數
    K_PERIOD = 9     # %K 計算週期 (通常為9)
    D_PERIOD = 3     # %D 計算週期 (通常為3)
    SMOOTH_K = 3     # %K 平滑期數
    OVERBOUGHT_THRESHOLD = 80  # 超買門檻
    OVERSOLD_THRESHOLD = 20    # 超賣門檻

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        自定義所需要的列名

        y_timestamp 為採樣點的時間戳 (Required)
        """
        columns = []

        # 當下欄位
        columns.extend(["timestamp", "price", "is_buy", "k_value", "d_value"])

        # 延遲欄位（底線後的名稱需要與 alpha function 內 df 的欄位名稱一致）
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", f"y{i}_high", 
                f"y{i}_low", f"y{i}_k_value", f"y{i}_d_value"
            ])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        計算 KD Alpha (隨機指標)

        KD指標包含兩個線：
        - %K線：快速線，反應較為靈敏
        - %D線：慢速線，為%K的移動平均

        計算步驟：
        1. 計算未成熟隨機值(RSV)
        2. 計算%K值（對RSV進行平滑）
        3. 計算%D值（%K的移動平均）

        交易信號：
        - K線由下而上穿越D線時，產生買入信號
        - K線由上而下穿越D線時，產生賣出信號
        - 同時考慮超買超賣區域作為輔助判斷
        """
        df = rolling_window_df

        # 初始化 rolling_window_df 欄位
        required_columns = [
            "highest_high",  # N期內最高價
            "lowest_low",    # N期內最低價
            "rsv",          # 未成熟隨機值
            "k_value",      # K值
            "d_value",      # D值
            "calculated"    # 計算完成標記
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # 計算指標邏輯
        def calculate_row(row, df):
            # 取得當前行索引
            index = row.name
            
            # 如果數據不足計算週期，則僅標記為已計算並返回
            if index < self.K_PERIOD - 1:
                row["calculated"] = True
                return row

            # 計算過去N期的最高價和最低價
            period_slice = df.iloc[max(0, index - self.K_PERIOD + 1):index + 1]
            row["highest_high"] = period_slice["high"].max()
            row["lowest_low"] = period_slice["low"].min()

            # 計算RSV (Raw Stochastic Value)
            if (row["highest_high"] - row["lowest_low"]) != 0:
                row["rsv"] = ((row["close"] - row["lowest_low"]) / 
                            (row["highest_high"] - row["lowest_low"]) * 100)
            else:
                row["rsv"] = 50  # 處理分母為零的情況

            # 計算平滑化的K值
            if index == self.K_PERIOD - 1:
                # 第一個K值直接使用RSV
                row["k_value"] = row["rsv"]
            else:
                # 使用加權移動平均計算K值
                prev_k = df["k_value"].iloc[index - 1] if pd.notna(df["k_value"].iloc[index - 1]) else 50
                row["k_value"] = (prev_k * (self.SMOOTH_K - 1) + row["rsv"]) / self.SMOOTH_K

            # 計算D值（K值的移動平均）
            if index >= self.K_PERIOD + self.D_PERIOD - 2:
                # 使用加權移動平均計算D值
                prev_d = df["d_value"].iloc[index - 1] if pd.notna(df["d_value"].iloc[index - 1]) else 50
                row["d_value"] = (prev_d * (self.D_PERIOD - 1) + row["k_value"]) / self.D_PERIOD
            else:
                # 初始D值設為50或使用當前可用的K值的平均
                row["d_value"] = 50

            row["calculated"] = True
            return row

        # 僅處理未計算的 row
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # 採樣點邏輯
        if df["calculated"].sum() > self.K_PERIOD + self.D_PERIOD:
            # K線由下而上穿越D線，且在超賣區域
            if (df["k_value"].iloc[-2] <= df["d_value"].iloc[-2] and 
                df["k_value"].iloc[-1] > df["d_value"].iloc[-1] and 
                df["k_value"].iloc[-2] < self.OVERSOLD_THRESHOLD):
                
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = True
                new_point["k_value"] = df["k_value"].iloc[-1]
                new_point["d_value"] = df["d_value"].iloc[-1]

                return new_point, df

            # K線由上而下穿越D線，且在超買區域
            elif (df["k_value"].iloc[-2] >= df["d_value"].iloc[-2] and 
                  df["k_value"].iloc[-1] < df["d_value"].iloc[-1] and 
                  df["k_value"].iloc[-2] > self.OVERBOUGHT_THRESHOLD):
                
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = False
                new_point["k_value"] = df["k_value"].iloc[-1]
                new_point["d_value"] = df["d_value"].iloc[-1]

                return new_point, df

        return None, df