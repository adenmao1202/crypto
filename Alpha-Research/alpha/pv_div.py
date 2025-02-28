import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class PriceVolDivergence(BaseAlpha):
    # Parameters
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2024-11-01"
    END_DATE = "2024-12-01"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_5MINUTE
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Strategy Parameters
    PRICE_LOOKBACK = 2  # For price momentum
    VOL_LOOKBACK = 1    # For volume momentum
    FAST_PERIOD = 2     # Fast signal moving average
    SLOW_PERIOD = 10    # Slow signal moving average
    NOTE = "Price-Volume Divergence Crossover Strategy"

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
            "price_mom", "vol_mom", "divergence",
            "fast_sig", "slow_sig"
        ])

        # Lagged columns
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close",
                f"y{i}_high", f"y{i}_low", f"y{i}_volume",
                f"y{i}_divergence", f"y{i}_fast_sig", f"y{i}_slow_sig"
            ])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        Price-Volume Divergence Alpha Strategy
        """
        df = rolling_window_df

        # Initialize required columns
        required_columns = [
            "price_mom", "vol_mom", "divergence",
            "fast_sig", "slow_sig", "calculated"
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        # Calculate indicator logic
        def calculate_row(row, df):
            index = row.name

            # Skip initial data points until we have enough history
            if index < self.PRICE_LOOKBACK:
                return row

            # Calculate price momentum (close/close_2_periods_ago - 1)
            current_close = df["close"].iloc[index]
            prev_close = df["close"].iloc[index - self.PRICE_LOOKBACK]
            row["price_mom"] = (current_close / prev_close) - 1

            # Calculate volume momentum (volume/volume_previous - 1)
            current_vol = df["volume"].iloc[index]
            prev_vol = df["volume"].iloc[index - self.VOL_LOOKBACK]
            row["vol_mom"] = (current_vol / prev_vol) - 1

            # Calculate divergence
            row["divergence"] = row["price_mom"] - row["vol_mom"]

            # Mark as calculated
            row["calculated"] = True

            return row

        # Only process uncalculated rows
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # Calculate signals when enough data points are available
        min_required_points = max(self.PRICE_LOOKBACK, self.SLOW_PERIOD)
        if df["calculated"].sum() > min_required_points:
            # Calculate fast and slow signals
            df["fast_sig"] = df["divergence"].rolling(window=self.FAST_PERIOD, min_periods=self.FAST_PERIOD).mean()
            df["slow_sig"] = df["divergence"].rolling(window=self.SLOW_PERIOD, min_periods=self.SLOW_PERIOD).mean()

            # Get current and previous signal values
            current_fast = df["fast_sig"].iloc[-1]
            current_slow = df["slow_sig"].iloc[-1]
            prev_fast = df["fast_sig"].iloc[-2]
            prev_slow = df["slow_sig"].iloc[-2]

            # Generate signals based on fast/slow crossovers
            if (prev_fast <= prev_slow and 
                current_fast > current_slow):
                # Bullish signal - fast crosses above slow
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point.update({
                    "timestamp": current_time,
                    "price": df["close"].iloc[-1],
                    "is_buy": True,
                    "price_mom": df["price_mom"].iloc[-1],
                    "vol_mom": df["vol_mom"].iloc[-1],
                    "divergence": df["divergence"].iloc[-1],
                    "fast_sig": current_fast,
                    "slow_sig": current_slow
                })
                return new_point, df

            elif (prev_fast >= prev_slow and 
                  current_fast < current_slow):
                # Bearish signal - fast crosses below slow
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point.update({
                    "timestamp": current_time,
                    "price": df["close"].iloc[-1],
                    "is_buy": False,
                    "price_mom": df["price_mom"].iloc[-1],
                    "vol_mom": df["vol_mom"].iloc[-1],
                    "divergence": df["divergence"].iloc[-1],
                    "fast_sig": current_fast,
                    "slow_sig": current_slow
                })
                return new_point, df

        return None, df