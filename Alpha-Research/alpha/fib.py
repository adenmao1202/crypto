import numpy as np
import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class FibonacciMomentumAlpha(BaseAlpha):
    """
    A trading strategy that combines Fibonacci retracement levels with momentum indicators
    to generate trading signals for cryptocurrency pairs.
    """
    
    EXCHANGE = "binance"
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2024-06-01"
    END_DATE = "2024-12-04"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_1MINUTE
    SAMPLING_INTERVALS = list(range(1, 21)) 
    
    FIB_RATIO = 0.618
    SPREAD_THRESHOLD = 5 

    def __init__(self):
        """Initialize the FibonacciMomentumAlpha strategy."""
        super().__init__()

    def get_columns(self):
        """
        Define the required columns for the strategy.

        Returns:
            list: List of column names including current and lagged values
        """
        columns = [
            "timestamp",
            "price",
            "is_buy",
            "spread",
            "fib_level"
        ]

        # Add lagged columns for each sampling interval
        for i in self.SAMPLING_INTERVALS:
            columns.extend([
                f"y{i}_timestamp",
                f"y{i}_open",
                f"y{i}_close", 
                f"y{i}_high",
                f"y{i}_low",
                f"y{i}_spread"
            ])

        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        Calculate trading signals based on Fibonacci levels and momentum.

        Args:
            rolling_window_df (pd.DataFrame): Historical price data window
            current_time (datetime): Current timestamp
            generate_sampling_points (callable): Function to generate sampling points

        Returns:
            tuple: (dict or None, pd.DataFrame) Trading signal and updated dataframe
        """
        df = rolling_window_df.copy()
        
        # Initialize columns if they don't exist
        for col in ["Fib_Level", "Spread", "calculated"]:
            if col not in df.columns:
                df[col] = np.nan
        
        # Calculate Fibonacci levels
        high = df["high"].max()
        low = df["low"].min()
        fib_level = high - (high - low) * self.FIB_RATIO
        
        # Calculate spread
        current_close = df["close"].iloc[-1]
        spread = current_close - fib_level
        
        # Update DataFrame
        df.loc[df.index[-1], "Fib_Level"] = fib_level
        df.loc[df.index[-1], "Spread"] = spread
        df.loc[df.index[-1], "calculated"] = True
        
        # Generate trading signals
        is_buy = spread > self.SPREAD_THRESHOLD
        is_sell = spread < -self.SPREAD_THRESHOLD
        
        if is_buy or is_sell:
            new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
            new_point.update({
                "timestamp": current_time,
                "price": current_close,
                "spread": spread,
                "fib_level": fib_level,
                "is_buy": is_buy
            })
            
            # Add lagged values
            for i in self.SAMPLING_INTERVALS:
                if len(df) >= i:
                    lag_row = df.iloc[-i]
                    new_point.update({
                        f"y{i}_timestamp": lag_row.name,
                        f"y{i}_open": lag_row["open"],
                        f"y{i}_close": lag_row["close"],
                        f"y{i}_high": lag_row["high"],
                        f"y{i}_low": lag_row["low"],
                        f"y{i}_spread": lag_row["Spread"]
                    })
                else:
                    new_point.update({
                        f"y{i}_timestamp": np.nan,
                        f"y{i}_open": np.nan,
                        f"y{i}_close": np.nan,
                        f"y{i}_high": np.nan,
                        f"y{i}_low": np.nan,
                        f"y{i}_spread": np.nan
                    })
            
            return new_point, df
        
        return None, df