import os
from binance.client import Client, HistoricalKlinesType
import pandas as pd
import requests
import zipfile
import io

client = Client()


def is_kline_data_exists(exchange, trading_pair, date, kline_interval):
    file_name = f"{trading_pair}_{date}_{kline_interval}.csv"
    file_path = os.path.join("kline", exchange, trading_pair, kline_interval, file_name)

    if os.path.exists(file_path):
        # print(f"Data exists: {file_path}")
        return True
    else:
        # print(f"Data does not exist: {file_path}")
        return False


def fetch_kline_from_api(symbol, interval, date):
    # Build start and end time, and set to UTC+0 timezone
    start_str = f"{date} 00:00:00"
    end_str = f"{date} 23:59:59"

    # Get klines from Binance API
    klines = client.get_historical_klines(symbol, interval, start_str, end_str, klines_type=HistoricalKlinesType.FUTURES)

    # Convert data to DataFrame
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    # Convert timestamp to datetime format
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Convert data type
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df

def fetch_1s_kline_from_binance_data(trading_pair, date):
    """
    从 Binance Data 下载 1s K线数据
    date 格式: YYYY-MM-DD
    """
    # 转换日期格式从 YYYY-MM-DD 到 YYYY-MM-DD
    date_parts = date.split('-')
    formatted_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
    
    # 构建下载URL
    url = f"https://data.binance.vision/data/spot/daily/klines/{trading_pair}/1s/{trading_pair}-1s-{formatted_date}.zip"
    
    try:
        # 下载ZIP文件
        response = requests.get(url)
        response.raise_for_status()
        
        # 解压ZIP文件并读取CSV
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_filename = zip_file.namelist()[0]  # 获取ZIP中的CSV文件名
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file, header=None, names=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ])
        
        # 转换时间戳为datetime格式
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # 转换数据类型
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        
        return df
    
    except Exception as e:
        print(f"下载1s数据时发生错误: {e}")
        return pd.DataFrame()
    
def get_kline(exchange, trading_pair, date, kline_interval):
    file_name = f"{trading_pair}_{date}_{kline_interval}.csv"
    file_path = os.path.join("kline", exchange, trading_pair, kline_interval, file_name)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not is_kline_data_exists(exchange, trading_pair, date, kline_interval):
        if kline_interval == "1s":
            df = fetch_1s_kline_from_binance_data(trading_pair, date)
        else:
            df = fetch_kline_from_api(trading_pair, kline_interval, date)

        if not df.empty:
            df.to_csv(file_path, index=False)
            print(f"K线数据已保存到: {file_path}")
