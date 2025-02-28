import numpy as np
import pandas as pd
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class MomentumAlpha(BaseAlpha):
    # Trading parameters
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-31"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_4HOUR
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Momentum parameters
    LOOKBACK_WINDOW = 12 # N in the formula

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """
        Define required columns for the analysis
        """
        columns = []
        columns.extend([
            "timestamp", "price", "is_buy", "momentum", "log_return",
            "signal"
        ])
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", f"y{i}_high",
                f"y{i}_low", f"y{i}_momentum", f"y{i}_signal"
            ])
        return columns

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        Calculate Momentum Alpha following the mathematical formula:
        xt = ln(pt/pt-1)
        mt-1(N) = (1/N) * Σ(xt-i) = [ln(pt-1) - ln(pt-N-1)]/N
        """
        df = rolling_window_df.copy()

        # Initialize required columns
        required_columns = [
            "log_return",    # xt in the formula
            "momentum",      # mt-1(N) in the formula
            "signal",        # Position signal
            "calculated"     # Calculation flag
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        def calculate_row(row, df):
            index = row.name

            # Skip if not enough data
            if index < 1:
                row["calculated"] = True
                return row

            # Calculate log return: xt = ln(pt/pt-1)
            row["log_return"] = np.log(df["close"].iloc[index] / df["close"].iloc[index-1])

            # Calculate momentum if enough data available
            if index >= self.LOOKBACK_WINDOW:
                # Direct implementation of mt-1(N) = [ln(pt-1) - ln(pt-N-1)]/N
                # Note: at current time t, we use data up to t-1 to avoid look-ahead bias
                p_t_minus_1 = df["close"].iloc[index-1]
                p_t_minus_N_minus_1 = df["close"].iloc[index-self.LOOKBACK_WINDOW-1]
                row["momentum"] = (np.log(p_t_minus_1) - np.log(p_t_minus_N_minus_1)) / self.LOOKBACK_WINDOW

                # ---> Change threshold !!!  --> need to be unparallel ( 這個值應該要是動態計算的)
                if row["momentum"] > 0.005:
                    row["signal"] = 1  # short 
                elif row["momentum"] < -0.1 :
                    row["signal"] = -1   # long  
                else:
                    row["signal"] = 0

            row["calculated"] = True
            return row

        # Process only uncalculated rows
        df.loc[df["calculated"] != True] = df.loc[df["calculated"] != True].apply(calculate_row, axis=1, df=df)

        # Generate sampling points at every K-line after having enough data
        if df["calculated"].sum() > self.LOOKBACK_WINDOW + 1:
            current_signal = df["signal"].iloc[-1]
            
            # Generate point at every K-line where we have a valid momentum value
            if not pd.isna(df["momentum"].iloc[-1]):
                new_point = generate_sampling_points(current_time, self.KLINE_INTERVAL)
                new_point["timestamp"] = current_time
                new_point["price"] = df["close"].iloc[-1]
                new_point["is_buy"] = True if current_signal > 0 else False    # marking method 
                new_point["momentum"] = df["momentum"].iloc[-1]
                new_point["signal"] = current_signal
                new_point["log_return"] = df["log_return"].iloc[-1]

                return new_point, df

        return None, df