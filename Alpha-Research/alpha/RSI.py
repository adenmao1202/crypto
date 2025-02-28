import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class RSI(BaseAlpha):
    # Parameters
    TRADING_PAIR = "USDT"
    START_DATE = "2021-01-01"
    END_DATE = "2025-02-26"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_1DAY
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    RSI_LENGTH = 12
    NOTE = ""

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        Define all required column names

        y_timestamp is the timestamp of sampling points (Required)
        """
        columns = []

        # Current columns
        columns.extend(["timestamp", "price", "is_buy", "rsi"])

        # Lagged columns
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", f"y{i}_high", f"y{i}_low", f"y{i}_rsi"])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        Calculate RSI Alpha
        """
        df = rolling_window_df

        # Initialize rolling_window_df columns
        required_columns = ["price_change", "gain", "loss", "avg_gain", "avg_loss", "calculated"]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # Calculate indicator logic
        def calculate_row(row, df):
            index = row.name

            # Skip first data point
            if index == 0:
                return row

            # Calculate price change
            row["price_change"] = row["close"] - df["close"].iloc[index - 1]
            
            # Calculate gains and losses
            row["gain"] = max(row["price_change"], 0)
            row["loss"] = abs(min(row["price_change"], 0))
            
            # Mark as calculated
            row["calculated"] = True

            return row

        # Only process uncalculated rows
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # Calculate RSI when enough data points are available
        if df["calculated"].sum() > self.RSI_LENGTH:
            # Calculate average gain and loss
            recent_gains = df["gain"].iloc[-self.RSI_LENGTH:]
            recent_losses = df["loss"].iloc[-self.RSI_LENGTH:]
            
            avg_gain = recent_gains.mean()
            avg_loss = recent_losses.mean()

            # Avoid division by zero
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            df.at[df.index[-1], "rsi"] = rsi

            # Generate sampling points based on RSI values
            if rsi >= 70:  # Overbought
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = False  # Sell signal
                new_point["rsi"] = rsi

                return new_point, df

            elif rsi <= 30:  # Oversold
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = True  # Buy signal
                new_point["rsi"] = rsi

                return new_point, df

        return None, df