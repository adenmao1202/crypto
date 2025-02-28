import pandas as pd
import numpy as np
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class KeltnerChannel(BaseAlpha):
    # Parameters
    TRADING_PAIR = "1000PEPEUSDT"
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-31"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_1DAY 
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    KC_LENGTH = 35
    KC_MULT = 1.7
    K = 0.03
    N = 2.0
    NOTE = "Keltner Channel Strategy with MACD - Long and Short"

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        Define all required column names
        """
        columns = []

        # Current columns
        columns.extend([
            "timestamp", "price", "is_buy", "is_short",  # Added is_short flag
            "kc_middle", "kc_upper", "kc_lower",
            "atr", "ema", "macd", "macd_signal"
        ])

        # Lagged columns
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close",
                f"y{i}_high", f"y{i}_low", f"y{i}_volume",
                f"y{i}_kc_upper", f"y{i}_kc_lower"
            ])

        return columns

    def calculate_tr(self, high, low, prev_close):
        """Calculate True Range"""
        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

    def calculate_ema(self, series, length):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (length + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()

    def create_signal_point(self, current_time, df, is_buy, is_short):
        """Helper function to create a signal point"""
        new_point = self.generate_sampling_points(current_time, self.KLINE_INTERVAL)
        new_point["timestamp"] = current_time
        new_point["price"] = df["close"].iloc[-1]
        new_point["is_buy"] = is_buy
        new_point["is_short"] = is_short
        new_point["kc_middle"] = df["kc_middle"].iloc[-1]
        new_point["kc_upper"] = df["kc_upper"].iloc[-1]
        new_point["kc_lower"] = df["kc_lower"].iloc[-1]
        new_point["atr"] = df["atr"].iloc[-1]
        new_point["ema"] = df["ema"].iloc[-1]
        new_point["macd"] = df["macd"].iloc[-1]
        new_point["macd_signal"] = df["macd_signal"].iloc[-1]
        return new_point

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        Calculate Keltner Channel Alpha with both long and short signals
        """
        self.generate_sampling_points = generate_sampling_points  # Store for helper function
        df = rolling_window_df.copy()

        # Initialize required columns if they don't exist
        required_columns = ["tr", "atr", "ema", "kc_middle", "kc_upper", "kc_lower", "calculated"]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # Calculate indicators for uncalculated rows
        def calculate_row(row, df):
            index = row.name

            # Skip first data point
            if index == 0:
                return row

            # Calculate True Range
            row["tr"] = self.calculate_tr(
                row["high"],
                row["low"],
                df["close"].iloc[index - 1]
            )

            # Mark as calculated
            row["calculated"] = True

            return row

        # Process uncalculated rows
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(
            calculate_row, axis=1, df=df
        )

        # Calculate indicators when enough data points are available
        if df["calculated"].sum() >= self.KC_LENGTH:
            # Calculate ATR
            df["atr"] = df["tr"].rolling(window=self.KC_LENGTH).mean()

            # Calculate EMA (KC Middle Line)
            df["ema"] = self.calculate_ema(df["close"], self.KC_LENGTH)

            # Calculate KC Bands
            df["kc_middle"] = df["ema"]
            df["kc_upper"] = df["kc_middle"] + (self.KC_MULT * df["atr"])
            df["kc_lower"] = df["kc_middle"] - (self.KC_MULT * df["atr"])

            # Calculate MACD
            ema6 = self.calculate_ema(df["close"], 6)
            ema14 = self.calculate_ema(df["close"], 14)
            df["macd"] = ema6 - ema14
            df["macd_signal"] = self.calculate_ema(df["macd"], 8)

            # Get current and previous values
            current_close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2]
            current_kc_upper = df["kc_upper"].iloc[-1]
            prev_kc_upper = df["kc_upper"].iloc[-2]
            current_kc_lower = df["kc_lower"].iloc[-1]
            prev_kc_lower = df["kc_lower"].iloc[-2]
            current_macd = df["macd"].iloc[-1]
            current_macd_signal = df["macd_signal"].iloc[-1]

            # Long entry condition: Price closes above KC upper band with MACD confirmation
            if (prev_close <= prev_kc_upper and current_close > current_kc_upper and 
                current_macd > current_macd_signal):
                return self.create_signal_point(current_time, df, True, False), df

            # Short entry condition: Price closes below KC lower band with MACD confirmation
            elif (prev_close >= prev_kc_lower and current_close < current_kc_lower and 
                  current_macd < current_macd_signal):
                return self.create_signal_point(current_time, df, False, True), df

        return None, df