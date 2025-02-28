import pandas as pd
import numpy as np
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class EnhancedRSI(BaseAlpha):
    # Parameters
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2022-01-01"
    END_DATE = "2022-12-26"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_30MINUTE
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Strategy Parameters
    RSI_LENGTH = 12
    EMA_FAST = 2
    EMA_SLOW = 8
    VOLATILITY_WINDOW = 24  # Window for calculating volatility
    VOLATILITY_THRESHOLD = 1.2  # Threshold for volatility filter (standard deviations)
    NOTE = "RSI strategy with EMA crossover confirmation and volatility filter"

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        Define all required column names
        """
        columns = []

        # Current columns
        columns.extend([
            "timestamp", "price", "is_buy", "rsi", 
            "ema_fast", "ema_slow", "volatility"
        ])

        # Lagged columns
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", 
                f"y{i}_high", f"y{i}_low", f"y{i}_rsi",
                f"y{i}_ema_fast", f"y{i}_ema_slow", f"y{i}_volatility"
            ])

        return columns

    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_volatility(self, data, window):
        """Calculate Rolling Volatility"""
        return data['close'].pct_change().rolling(window=window).std()

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        Enhanced RSI Alpha with EMA and Volatility Filter
        """
        df = rolling_window_df

        # Initialize rolling_window_df columns
        required_columns = [
            "price_change", "gain", "loss", "avg_gain", "avg_loss", 
            "calculated", "ema_fast", "ema_slow", "volatility"
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # Calculate basic RSI components
        def calculate_row(row, df):
            index = row.name
            if index == 0:
                return row

            row["price_change"] = row["close"] - df["close"].iloc[index - 1]
            row["gain"] = max(row["price_change"], 0)
            row["loss"] = abs(min(row["price_change"], 0))
            row["calculated"] = True
            return row

        # Process uncalculated rows
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(
            calculate_row, axis=1, df=df
        )

        # Calculate indicators when enough data points are available
        if df["calculated"].sum() > max(self.RSI_LENGTH, self.VOLATILITY_WINDOW):
            # Calculate RSI
            recent_gains = df["gain"].iloc[-self.RSI_LENGTH:]
            recent_losses = df["loss"].iloc[-self.RSI_LENGTH:]
            
            avg_gain = recent_gains.mean()
            avg_loss = recent_losses.mean()

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            df.at[df.index[-1], "rsi"] = rsi

            # Calculate EMAs
            df["ema_fast"] = self.calculate_ema(df["close"], self.EMA_FAST)
            df["ema_slow"] = self.calculate_ema(df["close"], self.EMA_SLOW)

            # Calculate Volatility
            df["volatility"] = self.calculate_volatility(df, self.VOLATILITY_WINDOW)

            # Get latest values
            current_volatility = df["volatility"].iloc[-1]
            current_ema_fast = df["ema_fast"].iloc[-1]
            current_ema_slow = df["ema_slow"].iloc[-1]
            prev_ema_fast = df["ema_fast"].iloc[-2]
            prev_ema_slow = df["ema_slow"].iloc[-2]

            # Check if volatility is within acceptable range
            volatility_acceptable = (
                current_volatility is not None and 
                current_volatility <= self.VOLATILITY_THRESHOLD
            )

            # Generate trading signals based on combined conditions
            if volatility_acceptable:
                # Check for EMA crossover
                ema_crossover_bullish = (
                    prev_ema_fast <= prev_ema_slow and 
                    current_ema_fast > current_ema_slow
                )
                ema_crossover_bearish = (
                    prev_ema_fast >= prev_ema_slow and 
                    current_ema_fast < current_ema_slow
                )

                if rsi <= 30 and ema_crossover_bullish:  # Oversold + Bullish Crossover
                    new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                    new_point.update({
                        "timestamp": current_time,
                        "price": df["close"].iloc[-1],
                        "is_buy": True,
                        "rsi": rsi,
                        "ema_fast": current_ema_fast,
                        "ema_slow": current_ema_slow,
                        "volatility": current_volatility
                    })
                    return new_point, df

                elif rsi >= 70 and ema_crossover_bearish:  # Overbought + Bearish Crossover
                    new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                    new_point.update({
                        "timestamp": current_time,
                        "price": df["close"].iloc[-1],
                        "is_buy": False,
                        "rsi": rsi,
                        "ema_fast": current_ema_fast,
                        "ema_slow": current_ema_slow,
                        "volatility": current_volatility
                    })
                    return new_point, df

        return None, df