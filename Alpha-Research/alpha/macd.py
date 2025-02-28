from alpha.base_alpha import BaseAlpha
from binance.client import Client

class MACD(BaseAlpha):
    # 參數設置
    TRADING_PAIR = "BTCUSDT"

    START_DATE = "2024-01-01"
    END_DATE = "2024-12-31"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_4HOUR
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 根據 KLINE_INTERVAL 的設定採樣 K 棒間隔
    MACD_LENGTH = 10

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        自定義所有需要的列名

        y_timestamp 為採樣點的時間戳 (Required)
        """
        columns = []

        # 當下欄位
        columns.extend(["timestamp", "price", "is_buy", "macd"])

        # 延遲欄位（底線後的名稱需要與 alpha function 內 df 的欄位名稱一致）
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", f"y{i}_high", f"y{i}_low", f"y{i}_macd"])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        計算 Alpha 邏輯
        """
        df = rolling_window_df

        # 初始化 rolling_window_df 欄位
        required_columns = ["ema3", "ema12", "macd", "signal", "calculated"]
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
            row["ema3"] = df["close"].ewm(span=3, adjust=False).mean().iloc[index]
            row["ema12"] = df["close"].ewm(span=12, adjust=False).mean().iloc[index]
            row["macd"] = row["ema3"] - row["ema12"]
            row["signal"] = df["macd"].ewm(span=self.MACD_LENGTH, adjust=False).mean().iloc[index]
            row["calculated"] = True

            return row

        # 僅處理未計算的 row
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        if df["calculated"].sum() > self.MACD_LENGTH:
            if df["macd"].iloc[-1] > df["signal"].iloc[-1] and df["macd"].iloc[-2] <= df["signal"].iloc[-2]:
                # 生成採樣點（初始化）
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)

                # 紀錄採樣點值（當下欄位）
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = True
                new_point["macd"] = df["macd"].iloc[-1]

                return new_point, df

            elif df["macd"].iloc[-1] < df["signal"].iloc[-1] and df["macd"].iloc[-2] >= df["signal"].iloc[-2]:
                # 生成採樣點（初始化）
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)

                # 紀錄採樣點值（當下欄位）
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = False
                new_point["macd"] = df["macd"].iloc[-1]

                return new_point, df

        return None, df
