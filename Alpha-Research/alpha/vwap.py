import pandas as pd
import numpy as np
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class VWAPCross(BaseAlpha):
    # Parameters
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2024-01-12"
    END_DATE = "2024-06-12"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_5MINUTE
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # VWAP Parameters
    ROLLING_WINDOW = 5  # 5-minute rolling window
    NOTE = "VWAP crossover strategy with double break signals"

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        Define all required column names
        """
        columns = []

        # Current columns
        columns.extend([
            "timestamp", "price", "is_buy", 
            "cum_vwap", "rolling_vwap", "double_break"
        ])

        # Lagged columns
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", 
                f"y{i}_high", f"y{i}_low", f"y{i}_volume",
                f"y{i}_cum_vwap", f"y{i}_rolling_vwap"
            ])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        VWAP Crossover Alpha Strategy
        All calculations are performed within this function using iloc
        """
        df = rolling_window_df

        # Initialize required columns
        required_columns = [
            "typical_price", "tp_vol", "cum_tp_vol",
            "cum_vol", "cum_vwap", "rolling_vwap", "calculated"
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # Calculate indicator logic
        def calculate_row(row, df):
            index = row.name

            # Skip first data point
            if index == 0:
                return row

            # Calculate typical price using previous candle
            prev_high = df["high"].iloc[index - 1]
            prev_low = df["low"].iloc[index - 1]
            prev_close = df["close"].iloc[index - 1]
            prev_volume = df["volume"].iloc[index - 1]
            
            # Calculate typical price and volume components
            row["typical_price"] = (prev_high + prev_low + prev_close) / 3
            row["tp_vol"] = row["typical_price"] * prev_volume
            
            # Mark as calculated
            row["calculated"] = True

            return row

        # Only process uncalculated rows
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # Calculate VWAP components when enough data is available
        if df["calculated"].sum() > 1:
            # Calculate cumulative values
            df["cum_tp_vol"] = df["tp_vol"].cumsum()
            df["cum_vol"] = df["volume"].cumsum()
            
            # Calculate cumulative VWAP
            df.loc[df["cum_vol"] > 0, "cum_vwap"] = df["cum_tp_vol"] / df["cum_vol"]
            
            # Calculate rolling VWAP
            rolling_tp_vol = df["tp_vol"].rolling(window=self.ROLLING_WINDOW, min_periods=1).sum()
            rolling_vol = df["volume"].rolling(window=self.ROLLING_WINDOW, min_periods=1).sum()
            df.loc[rolling_vol > 0, "rolling_vwap"] = rolling_tp_vol / rolling_vol

            # Get current and previous values for signal generation
            current_price = df["close"].iloc[-1]
            prev_price = df["close"].iloc[-2]
            prev2_price = df["close"].iloc[-3]
            current_cum_vwap = df["cum_vwap"].iloc[-1]
            prev_cum_vwap = df["cum_vwap"].iloc[-2]
            prev2_cum_vwap = df["cum_vwap"].iloc[-3]
            prev_rolling_vwap = df["rolling_vwap"].iloc[-2]
            current_rolling_vwap = df['rolling_vwap'].iloc[-1]
            # Generate signals based on VWAP crossovers
            
            if (prev2_price <= prev2_cum_vwap and 
                prev_price <= prev_cum_vwap and 
                current_price > current_cum_vwap and 
                prev_price > prev_rolling_vwap):
                # Bullish signal - price crosses above both VWAPs
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point.update({
                    "timestamp": current_time,
                    "price": current_price,
                    "is_buy": True,
                    "cum_vwap": current_cum_vwap,
                    "rolling_vwap": current_rolling_vwap,
                    "double_break": 1
                })
                return new_point, df

            elif (prev2_price >= prev2_cum_vwap and 
                  prev_price >= prev_cum_vwap and 
                  current_price < current_cum_vwap and 
                  current_price > prev_rolling_vwap):
                # Bearish signal - price crosses below both VWAPs
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point.update({
                    "timestamp": current_time,
                    "price": current_price,
                    "is_buy": False,
                    "cum_vwap": current_cum_vwap,
                    "rolling_vwap": current_rolling_vwap,
                    "double_break": -1
                })
                return new_point, df

        return None, df