import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class EMACross(BaseAlpha):
    # Parameters
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2022-06-01"
    END_DATE = "2022-12-26"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_1HOUR   # 改k bar 
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # EMA Parameters
    EMA_FAST = 2
    EMA_SLOW = 10
    NOTE = "Simple EMA crossover strategy"

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        Define all required column names
        """
        columns = []

        # Current columns
        columns.extend(["timestamp", "price", "is_buy", "ema_fast", "ema_slow"])

        # Lagged columns
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", 
                f"y{i}_high", f"y{i}_low", f"y{i}_ema_fast", f"y{i}_ema_slow"
            ])

        return columns

    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        EMA Crossover Alpha Strategy
        """
        df = rolling_window_df

        # Initialize EMA columns if they don't exist
        if "ema_fast" not in df.columns:
            df["ema_fast"] = None
        if "ema_slow" not in df.columns:
            df["ema_slow"] = None

        # Calculate EMAs
        df["ema_fast"] = self.calculate_ema(df["close"], self.EMA_FAST)
        df["ema_slow"] = self.calculate_ema(df["close"], self.EMA_SLOW)

        # Need at least two data points to check for crossover
        if len(df) >= 2:
            current_ema_fast = df["ema_fast"].iloc[-1]
            current_ema_slow = df["ema_slow"].iloc[-1]
            prev_ema_fast = df["ema_fast"].iloc[-2]
            prev_ema_slow = df["ema_slow"].iloc[-2]

            # Check for crossovers
            bullish_crossover = (
                prev_ema_fast <= prev_ema_slow and 
                current_ema_fast > current_ema_slow
            )
            
            bearish_crossover = (
                prev_ema_fast >= prev_ema_slow and 
                current_ema_fast < current_ema_slow
            )

            if bullish_crossover:  # Generate buy signal
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point.update({
                    "timestamp": current_time,
                    "price": df["close"].iloc[-1],
                    "is_buy": True,
                    "ema_fast": current_ema_fast,
                    "ema_slow": current_ema_slow
                })
                return new_point, df

            elif bearish_crossover:  # Generate sell signal
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point.update({
                    "timestamp": current_time,
                    "price": df["close"].iloc[-1],
                    "is_buy": False,
                    "ema_fast": current_ema_fast,
                    "ema_slow": current_ema_slow
                })
                return new_point, df

        return None, df