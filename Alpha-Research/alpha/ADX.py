import numpy as np
import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class ADX(BaseAlpha):
    # 參數設置
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2022-01-01"
    END_DATE = "2022-12-31"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_1HOUR
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # 根據 KLINE_INTERVAL 設定採樣 K 棒間隔
    ADX_LENGTH = 12

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        自定義所有需要的列名

        y_timestamp 為採樣點的時間戳 (Required)
        """
        columns = []

        # 當下欄位
        columns.extend(["timestamp", "price", "is_buy", "adx", "di_plus", "di_minus"])

        # 延遲欄位（底線後的名稱需要與 alpha function 內 df 的欄位名稱一致）
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend(
                [f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", f"y{i}_high", f"y{i}_low", f"y{i}_adx", f"y{i}_di_plus", f"y{i}_di_minus"]
            )

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        計算 ADX Alpha

        ADX 是一種技術指標，用來衡量市場趨勢的強度，無論趨勢是上升還是下降。

        DI+ 和 DI- 是方向性指標，用來判斷價格變動的方向性動量。
        DI+ 代表價格的上升動量。
        DI- 代表價格的下降動量。

        ADX 和 DI 的組合可以幫助交易者判斷市場是否處於趨勢中，以及該趨勢的強弱。
        """
        df = rolling_window_df

        # 初始化 rolling_window_df 欄位
        required_columns = [
            "true_range",
            "directional_movement_plus",
            "directional_movement_minus",
            "smoothed_true_range",
            "smoothed_directional_movement_plus",
            "smoothed_directional_movement_minus",
            "di_plus",
            "di_minus",
            "dx",
            "adx",
            "calculated",
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # 計算指標邏輯
        def calculate_row(row, df):
            # 取得當前行索引
            index = row.name

            # 忽略第一筆資料
            if index == 0:
                return row

            # 計算指標
            # 價格波動真實範圍
            row["true_range"] = np.maximum(
                np.maximum(row["high"] - row["low"], np.abs(row["high"] - df["close"].iloc[index - 1])),
                np.abs(row["low"] - df["close"].iloc[index - 1]),
            )

            # 如果當前的高點突破前一根 K 線的高點幅度大於低點的跌破幅度，則計算 DM+，否則為 0
            row["directional_movement_plus"] = np.where(
                (row["high"] - df["high"].iloc[index - 1]) > (df["low"].iloc[index - 1] - row["low"]),
                np.maximum(row["high"] - df["high"].iloc[index - 1], 0),
                0,
            )
            # 如果當前的低點跌破前一根 K 線的低點幅度大於高點的突破幅度，則計算 DM-，否則為 0
            row["directional_movement_minus"] = np.where(
                (df["low"].iloc[index - 1] - row["low"]) > (row["high"] - df["high"].iloc[index - 1]),
                np.maximum(df["low"].iloc[index - 1] - row["low"], 0),
                0,
            )

            # 使用平滑公式計算 True Range (TR) 和 DM+ / DM- 的移動平均值
            row["smoothed_true_range"] = (
                (df["smoothed_true_range"].iloc[index - 1] if pd.notna(df["smoothed_true_range"].iloc[index - 1]) else 0)
                - (df["smoothed_true_range"].iloc[index - 1] if pd.notna(df["smoothed_true_range"].iloc[index - 1]) else 0) / self.ADX_LENGTH
                + row["true_range"]
            )
            row["smoothed_directional_movement_plus"] = (
                (
                    df["smoothed_directional_movement_plus"].iloc[index - 1]
                    if pd.notna(df["smoothed_directional_movement_plus"].iloc[index - 1])
                    else 0
                )
                - (
                    df["smoothed_directional_movement_plus"].iloc[index - 1]
                    if pd.notna(df["smoothed_directional_movement_plus"].iloc[index - 1])
                    else 0
                )
                / self.ADX_LENGTH
                + row["directional_movement_plus"]
            )
            row["smoothed_directional_movement_minus"] = (
                (
                    df["smoothed_directional_movement_minus"].iloc[index - 1]
                    if pd.notna(df["smoothed_directional_movement_minus"].iloc[index - 1])
                    else 0
                )
                - (
                    df["smoothed_directional_movement_minus"].iloc[index - 1]
                    if pd.notna(df["smoothed_directional_movement_minus"].iloc[index - 1])
                    else 0
                )
                / self.ADX_LENGTH
                + row["directional_movement_minus"]
            )

            # 計算 DI+ 和 DI-
            row["di_plus"] = row["smoothed_directional_movement_plus"] / row["smoothed_true_range"] * 100
            row["di_minus"] = row["smoothed_directional_movement_minus"] / row["smoothed_true_range"] * 100

            # DX
            row["dx"] = abs(row["di_plus"] - row["di_minus"]) / (row["di_plus"] + row["di_minus"]) * 100

            # ADX
            row["adx"] = (
                df["dx"].rolling(window=self.ADX_LENGTH, min_periods=1).mean().iloc[index]
                if index >= self.ADX_LENGTH - 1
                else df["dx"].iloc[: index + 1].mean()
            )

            row["calculated"] = True

            return row

        # 僅處理未計算的 row
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # 採樣點邏輯
        if df["calculated"].sum() > self.ADX_LENGTH:
            # DI+ 向上突破 DI-
            if df["di_plus"].iloc[-2] < df["di_minus"].iloc[-2] and df["di_plus"].iloc[-1] > df["di_minus"].iloc[-1]:
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = True
                new_point["adx"] = df["adx"].iloc[-1]
                new_point["di_plus"] = df["di_plus"].iloc[-1]
                new_point["di_minus"] = df["di_minus"].iloc[-1]

                return new_point, df

            # DI+ 向下跌破 DI-
            elif df["di_plus"].iloc[-2] > df["di_minus"].iloc[-2] and df["di_plus"].iloc[-1] < df["di_minus"].iloc[-1]:
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = False
                new_point["adx"] = df["adx"].iloc[-1]
                new_point["di_plus"] = df["di_plus"].iloc[-1]
                new_point["di_minus"] = df["di_minus"].iloc[-1]

                return new_point, df

        return None, df
