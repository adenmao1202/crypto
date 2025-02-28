import os
import sys
import pandas as pd
import importlib
from datetime import datetime, timedelta
from src.sampling import Sampling
from src.get_kline import get_kline
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

sys.dont_write_bytecode = True


class AlphaManager:
    def __init__(self, alpha_dir="alpha"):
        self.alpha_dir = alpha_dir
        self.strategies = self._load_strategies()

    def _load_strategies(self):
        """
        動態加載 alpha 資料夾中的策略模組
        """
        strategies = {}
        for file in os.listdir(self.alpha_dir):
            if file.endswith(".py") and file != "base_alpha.py" and file != "__init__.py":
                module_name = file[:-3]
                module = importlib.import_module(f"{self.alpha_dir}.{module_name}")
                for attr in dir(module):
                    cls = getattr(module, attr)
                    if isinstance(cls, type) and issubclass(cls, module.BaseAlpha) and cls is not module.BaseAlpha:
                        strategies[cls.__name__] = cls

        return strategies

    def get_alpha_names(self):
        """
        返回所有策略名稱
        """
        return list(self.strategies.keys())

    def get_alpha_class(self, name):
        """
        根據名稱返回策略類
        """
        return self.strategies[name]


def main():
    console = Console()
    manager = AlphaManager()

    # 顯示 Alpha 列表
    console.print("[bold cyan]Alpha List：[/bold cyan]")

    table = Table(show_lines=True)
    table.add_column("No.", justify="center", style="bold")
    table.add_column("Alpha Name", justify="left")

    alphas = manager.get_alpha_names()
    for i, alpha_name in enumerate(alphas, 1):
        table.add_row(str(i), alpha_name)

    console.print(table)

    # 選擇 Alpha
    while True:
        try:
            choice = int(console.input("[bold yellow]Please select Alpha (input No.): [/bold yellow]"))
            if 1 <= choice <= len(alphas):
                selected_alpha_name = alphas[choice - 1]
                break
            else:
                console.print("[bold red]Invalid number, please input again.[/bold red]")
        except ValueError:
            console.print("[bold red]Please input a valid number.[/bold red]")

    console.print(f"[bold green]Selected Alpha: {selected_alpha_name}[/bold green]")

    # 初始化 Alpha
    alpha_class = manager.get_alpha_class(selected_alpha_name)
    alpha_instance = alpha_class()

    # 取得 Alpha 的參數
    exchange = alpha_instance.EXCHANGE
    trading_pair = alpha_instance.TRADING_PAIR
    start_date_string = alpha_instance.START_DATE
    end_date_string = alpha_instance.END_DATE
    kline_interval = alpha_instance.KLINE_INTERVAL
    sampling_intervals = alpha_instance.SAMPLING_INTERVALS

    # Rolling window size
    length_attributes = {attribute: getattr(alpha_instance, attribute) for attribute in dir(alpha_instance) if attribute.endswith("_LENGTH")}
    max_length = max(length_attributes.values()) if length_attributes else 20
    # 根据时间间隔调整窗口大小
    if kline_interval == "1s":
        # 對於1s數據，使用更大的窗口以確保足夠的數據點
        window_size = max(int(max_length * 60 * 2), 120) 
    else:
        window_size = max(int(max_length * 1.5), 30)

    # 準備數據並執行回測
    current_date = datetime.strptime(start_date_string, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_string, "%Y-%m-%d")

    # 初始化採樣
    sampling = Sampling(window_size=window_size, sampling_intervals=sampling_intervals, alpha=alpha_instance)

    console.print("[bold cyan]Start sampling...[/bold cyan]")
    total_days = (end_date - current_date).days + 1

    with Progress() as progress:
        task = progress.add_task("[cyan]Sampling progress...", total=total_days)

        while not progress.finished:
            date_string = current_date.strftime("%Y-%m-%d")
            
            try:
                # 下載數據
                get_kline(exchange, trading_pair, date_string, kline_interval)
                
                # 構建文件路徑
                file_path = f"kline/{exchange}/{trading_pair}/{kline_interval}/{trading_pair}_{date_string}_{kline_interval}.csv"
                
                if os.path.exists(file_path):
                    # 執行採樣
                    sampling.alpha_sampling(file_path, alpha_instance)
                else:
                    console.print(f"[bold red]Warning: Data file not found for {date_string}[/bold red]")
                
                current_date += timedelta(days=1)
                progress.update(task, advance=1)
                
            except Exception as e:
                console.print(f"[bold red]Error processing {date_string}: {str(e)}[/bold red]")
                current_date += timedelta(days=1)
                progress.update(task, advance=1)
                continue

    # 保存結果
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = (
        f"sample_output/{selected_alpha_name}/{kline_interval}_{selected_alpha_name}_{exchange}_{trading_pair}_{start_date_string}_{end_date_string}"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    note = f"_{alpha_instance.NOTE}" if alpha_instance.NOTE else ""
    result_file = (
        f"{directory}/{kline_interval}_{selected_alpha_name}{note}_{exchange}_{trading_pair}_{start_date_string}_{end_date_string}_{now}.csv"
    )
    
    if not sampling.completed_samples_df.empty:
        sampling.completed_samples_df.to_csv(result_file, index=False)
        console.print(f"[bold green]{selected_alpha_name} sampling completed! Result saved to {result_file}[/bold green]")
    else:
        console.print("[bold red]Warning: No samples were generated![/bold red]")


if __name__ == "__main__":
    main()
