from abc import ABC, abstractmethod
from binance.client import Client


class BaseAlpha(ABC):
    """
    Alpha base class, all alphas should inherit this class
    """

    EXCHANGE = "binance"
    TRADING_PAIR = "BTCUSDT"
    START_DATE = "2024-12-01"
    END_DATE = "2024-12-04"
    KLINE_INTERVAL = Client.KLINE_INTERVAL_15MINUTE
    SAMPLING_INTERVALS = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 根據 KLINE_INTERVAL 設定採樣 K 棒間隔
    NOTE = ""  # 備註 note 於檔名

    def __init__(self):
        pass

    @abstractmethod
    def alpha(self, rolling_window_df, current_time, generate_sampling_points):
        """
        Alpha 邏輯
        :param rolling_window_df: 滾動窗口數據
        :param current_time: 當前時間
        :param generate_sampling_points: 用於生成採樣點的函數
        :return: 新的採樣點字典（如果有），否則返回 None
        """
        pass
