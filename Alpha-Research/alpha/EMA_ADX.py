import pandas as pd
import numpy as np
from alpha.base_alpha import BaseAlpha
from binance.client import Client


class EMA_ADX(BaseAlpha):
    # Parameters
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2020-01-01"
    END_DATE = "2022-12-31"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_1HOUR
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Strategy Parameters
    ADX_PERIOD = 18
    ADX_THRESHOLD = 49  # Strong trend threshold
    EMA_FAST = 30
    EMA_SLOW = 92
    NOTE = "ADX and EMA crossover combined strategy"

    def __init__(self):
        super().__init__()

    def get_columns(self):
        """Define all required column names"""
        columns = []

        # Current columns
        columns.extend([
            "timestamp", "price", "is_buy", 
            "adx", "ema_fast", "ema_slow", 
            "plus_di", "minus_di"
        ])

        # Lagged columns
        for i in range(1, len(self.SAMPLING_INTERVALS) + 1):
            columns.extend([
                f"y{i}_timestamp", f"y{i}_open", f"y{i}_close", 
                f"y{i}_high", f"y{i}_low"
            ])

        return columns

    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_adx(self, df):
        """Calculate ADX, +DI, and -DI"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range (TR)
        df['tr1'] = abs(high - low)
        df['tr2'] = abs(high - close.shift(1))
        df['tr3'] = abs(low - close.shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate directional movement
        df['up_move'] = high - high.shift(1)
        df['down_move'] = low.shift(1) - low
        
        # Calculate +DM and -DM
        df['+dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        df['-dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Calculate smoothed TR and DM
        df['tr_smoothed'] = df['tr'].rolling(
            window=self.ADX_PERIOD
        ).mean()
        df['+dm_smoothed'] = df['+dm'].rolling(
            window=self.ADX_PERIOD
        ).mean()
        df['-dm_smoothed'] = df['-dm'].rolling(
            window=self.ADX_PERIOD
        ).mean()
        
        # Calculate +DI and -DI
        df['+di'] = (df['+dm_smoothed'] / df['tr_smoothed']) * 100
        df['-di'] = (df['-dm_smoothed'] / df['tr_smoothed']) * 100
        
        # Calculate DX and ADX
        df['dx'] = abs(
            (df['+di'] - df['-di']) / (df['+di'] + df['-di'])
        ) * 100
        df['adx'] = df['dx'].rolling(window=self.ADX_PERIOD).mean()
        
        return df['adx'], df['+di'], df['-di']

    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """ADX and EMA Combined Strategy Alpha"""
        df = rolling_window_df.copy()

        # Initialize required columns
        if 'adx' not in df.columns:
            df['adx'] = None
        if '+di' not in df.columns:
            df['+di'] = None
        if '-di' not in df.columns:
            df['-di'] = None
        if 'ema_fast' not in df.columns:
            df['ema_fast'] = None
        if 'ema_slow' not in df.columns:
            df['ema_slow'] = None

        # Ensure we have enough data
        if len(df) >= self.ADX_PERIOD:
            # Calculate ADX and directional indicators
            df['adx'], df['+di'], df['-di'] = self.calculate_adx(df)
            
            # Calculate EMAs
            df['ema_fast'] = self.calculate_ema(df['close'], self.EMA_FAST)
            df['ema_slow'] = self.calculate_ema(df['close'], self.EMA_SLOW)

            # Get current values
            current_adx = df['adx'].iloc[-1]
            current_plus_di = df['+di'].iloc[-1]
            current_minus_di = df['-di'].iloc[-1]
            current_ema_fast = df['ema_fast'].iloc[-1]
            current_ema_slow = df['ema_slow'].iloc[-1]
            prev_ema_fast = df['ema_fast'].iloc[-2]
            prev_ema_slow = df['ema_slow'].iloc[-2]

            # Check for strong trend
            strong_trend = current_adx > self.ADX_THRESHOLD

            if strong_trend:
                # Check for EMA crossover
                ema_crossover_bullish = (
                    prev_ema_fast <= prev_ema_slow and 
                    current_ema_fast > current_ema_slow
                )
                ema_crossover_bearish = (
                    prev_ema_fast >= prev_ema_slow and 
                    current_ema_fast < current_ema_slow
                )

                # Generate buy signal
                if (ema_crossover_bullish and 
                    current_plus_di > current_minus_di):
                    new_point = generate_sampling_points(
                        current_time, 
                        self.KLINE_INTERVAL
                    )
                    new_point.update({
                        "timestamp": current_time,
                        "price": df["close"].iloc[-1],
                        "is_buy": True,
                        "adx": current_adx,
                        "plus_di": current_plus_di,
                        "minus_di": current_minus_di,
                        "ema_fast": current_ema_fast,
                        "ema_slow": current_ema_slow
                    })
                    return new_point, df

                # Generate sell signal
                elif (ema_crossover_bearish and 
                      current_minus_di > current_plus_di):
                    new_point = generate_sampling_points(
                        current_time, 
                        self.KLINE_INTERVAL
                    )
                    new_point.update({
                        "timestamp": current_time,
                        "price": df["close"].iloc[-1],
                        "is_buy": False,
                        "adx": current_adx,
                        "plus_di": current_plus_di,
                        "minus_di": current_minus_di,
                        "ema_fast": current_ema_fast,
                        "ema_slow": current_ema_slow
                    })
                    return new_point, df

        return None, df