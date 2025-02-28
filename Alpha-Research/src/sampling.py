import pandas as pd
from datetime import timedelta
from collections import deque
import warnings
import time

warnings.simplefilter(action="ignore", category=FutureWarning)


class Sampling:
    def __init__(self, window_size, sampling_intervals, alpha):
        """
        初始化採樣邏輯
        :param window_size: 滾動窗口大小
        :param sampling_intervals: 採樣時間間隔
        """
        self.window_size = window_size
        self.sampling_intervals = sampling_intervals
        self.rolling_window_df = pd.DataFrame()
        self.alpha_columns = alpha.get_columns()
        self.sampling_points_df = pd.DataFrame(columns=self.alpha_columns)
        self.completed_samples_df = pd.DataFrame(columns=self.alpha_columns)
        self.rolling_window = deque(maxlen=window_size)  # 使用固定長度的 deque

    def generate_sampling_points(self, current_time, kline_interval):
        """
        生成新的採樣點
        """
        new_point = {}

        # 初始化 y 和 y_timestamp 列
        minute_interval = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "20m":20,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
            "1M": 43200,
        }
        for i, interval in enumerate(self.sampling_intervals, start=1):
            if kline_interval == "1s":
                new_point[f"y{i}_timestamp"] = current_time + timedelta(seconds=interval)
            else:
                new_point[f"y{i}_timestamp"] = current_time + timedelta(minutes=interval * minute_interval[kline_interval])
        for column in self.alpha_columns:
            if column not in new_point:
                new_point[column] = None

        return new_point

    def _update_sampling_points(self, current_time, calculated_df):
        """
        更新採樣點數據
        """
        finished_rows = []
        if self.sampling_points_df.empty:
            return
        for index, row in self.sampling_points_df.iterrows():
            is_filled = True
            for i in range(1, len(self.sampling_intervals) + 1):
                if any(pd.isna(value) for value in row.values) and current_time >= row[f"y{i}_timestamp"]:
                    # 遍歷 calculated_df 的所有欄位
                    for calculated_column in calculated_df.columns:
                        column_suffix = calculated_column  # 例如 'open', 'close', 'macd' 等
                        target_column = f"y{i}_{column_suffix}"
                        if target_column in self.sampling_points_df.columns and pd.isna(self.sampling_points_df.at[index, target_column]):
                            self.sampling_points_df.at[index, target_column] = calculated_df[calculated_column].iloc[-1]

                if any(pd.isna(value) for value in row.values):
                    is_filled = False

            # 該採樣點已完成紀錄
            if is_filled:
                finished_rows.append(index)

        # 將完整採樣數據移至 completed_samples_df
        if finished_rows:
            self.completed_samples_df = pd.concat([self.completed_samples_df, self.sampling_points_df.loc[finished_rows]], ignore_index=True)
            self.sampling_points_df.drop(finished_rows, inplace=True)

    def alpha_sampling(self, kline_file_path, alpha):
        """
        執行 alpha 採樣
        :param kline_file_path: K 線數據文件路徑
        :param alpha: 策略類的實例
        """
        for chunk in pd.read_csv(kline_file_path, chunksize=1000):
            for _, row in chunk.iterrows():
                # 添加當前行到滾動窗口
                self.rolling_window_df = pd.concat([self.rolling_window_df, pd.DataFrame([row])], ignore_index=True)
                current_time = pd.to_datetime(row["close_time"])

                # rolling_window 已滿，開始採樣
                if len(self.rolling_window_df) == self.window_size:
                    # print("开始进行alpha采样")
                    new_point, calculated_df = alpha.alpha(self.rolling_window_df, current_time, self.generate_sampling_points)
                    if new_point:
                        self.sampling_points_df = pd.concat([self.sampling_points_df, pd.DataFrame([new_point])], ignore_index=True)
                        # print("采样完成，开始更新采样点数据。")

                    # 更新採樣點數據
                    self._update_sampling_points(current_time, calculated_df)

                    # 同步計算後的 df 與移除 window_size 以外的資料
                    # print(calculated_df)
                    self.rolling_window_df = calculated_df
                    self.rolling_window_df = self.rolling_window_df.iloc[-(self.window_size - 1) :].reset_index(drop=True)