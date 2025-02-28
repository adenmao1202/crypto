import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class ATR(BaseAlpha):
    # 參數設置
    TRADING_PAIR = "SANDUSDT"
    START_DATE = "2021-01-01"
    END_DATE = "2025-02-26"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_1DAY
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 根據 KLINE_INTERVAL 設定採樣 K 棒間隔
    ATR_LENGTH = 14
    NOTE = ""  # 備註 note 於檔名

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        自定義所有需要的列名

        y_timestamp 為採樣點的時間戳 (Required)
        """
        columns = []

        # 當下欄位
        columns.extend(["timestamp", "price", "is_buy", "atr"])

        # 延遲欄位（底線後的名稱需要與 alpha function 內 df 的欄位名稱一致）
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", f"y{i}_high", f"y{i}_low", f"y{i}_atr"])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        計算 ATR Alpha
        """
        df = rolling_window_df

        # 初始化 rolling_window_df 欄位
        required_columns = ["high_low", "high_close", "low_close", "tr", "calculated"]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # 計算指標邏輯（row 為當前行, df 為 rolling_window_df）
        def calculate_row(row, df):
            # 取得當前行索引
            index = row.name

            # 忽略第一筆資料
            if index == 0:
                return row

            # 計算指標
            row["high_low"] = row["high"] - row["low"]
            row["high_close"] = abs(row["high"] - df["close"].iloc[index - 1])
            row["low_close"] = abs(row["low"] - df["close"].iloc[index - 1])
            row["tr"] = max(row["high_low"], row["high_close"], row["low_close"])
            row["calculated"] = True

            return row

        # 僅處理未計算的 row
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # 只有當 df["calculated"] 為 True 的行數超過 14 行時才計算 ATR
        if df["calculated"].sum() > self.ATR_LENGTH:
            df.at[df.index[-1], "atr"] = df["tr"].iloc[-self.ATR_LENGTH :].mean()

        # 採樣點邏輯
        if df["calculated"].sum() > self.ATR_LENGTH:
            if df["close"].iloc[-1] > df["high"].iloc[-2] + df["atr"].iloc[-2]:
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = True
                new_point["atr"] = df["atr"].iloc[-1]

                return new_point, df

            elif df["close"].iloc[-1] < df["low"].iloc[-2] - df["atr"].iloc[-2]:
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = False
                new_point["atr"] = df["atr"].iloc[-1]

                return new_point, df

        return None, df
