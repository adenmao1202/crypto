import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Set output folder
sample_output = Path(os.path.dirname(__file__)).parent / "sample_output"
print(sample_output)

def find_unprocessed_csv():
    """Find CSV files without corresponding PNG files"""
    csv_files = set()
    png_files = set()

    for root, dirs, files in os.walk(sample_output):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, sample_output)

            if file.endswith(".csv"):
                csv_files.add(relative_path)
            elif file.endswith("_returns.png"):
                png_base = relative_path[:-12] + ".csv"
                png_files.add(png_base)

    unprocessed_csv = csv_files - png_files
    return sorted(list(unprocessed_csv))

def validate_data(df):
    """Validate data integrity"""
    # Check required columns
    required_columns = ["timestamp", "price", "is_buy"]
    y_columns = [f"y{i}_close" for i in range(1, 21)]  # Updated to 20 k-bars
    required_columns.extend(y_columns)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Validate price data
    if df["price"].isna().any():
        raise ValueError("Price column contains missing values")
    if (df["price"] <= 0).any():
        raise ValueError("Price column contains non-positive values")

    # Validate y1-y20 data
    for i in range(1, 21):
        col = f"y{i}_close"
        if df[col].isna().any():
            raise ValueError(f"{col} contains missing values")
        if (df[col] <= 0).any():
            raise ValueError(f"{col} contains non-positive values")

    # Validate timestamp
    try:
        pd.to_datetime(df["timestamp"])
    except Exception as e:
        raise ValueError("Invalid timestamp format") from e

    # Validate is_buy
    if not df["is_buy"].isin([True, False]).all():
        raise ValueError("is_buy column contains invalid values")

    return True

def calculate_descriptive_stats(df):
    """Calculate descriptive statistics for each k-bar"""
    stats = {}

    for i in range(1, 21):  # Updated to 20 k-bars
        y_col = f"y{i}_close"

        # Long statistics
        df_buy = df[df["is_buy"]]
        buy_returns = (df_buy[y_col] - df_buy["price"]) / df_buy["price"]
        buy_stats = {
            "Sample Size": len(buy_returns),
            "Mean Return": buy_returns.mean() * 100,
            "Std Dev": buy_returns.std() * 100,
            "Min Return": buy_returns.min() * 100,
            "25th Percentile": buy_returns.quantile(0.25) * 100,
            "Median": buy_returns.median() * 100,
            "75th Percentile": buy_returns.quantile(0.75) * 100,
            "Max Return": buy_returns.max() * 100,
            "Skewness": buy_returns.skew(),
            "Kurtosis": buy_returns.kurtosis(),
            "Sharpe Ratio": (buy_returns.mean() / buy_returns.std()) if buy_returns.std() != 0 else 0,
            "Win Rate": (buy_returns > 0).mean() * 100  # Added win rate
        }

        # Short statistics
        df_sell = df[~df["is_buy"]]
        sell_returns = -(df_sell[y_col] - df_sell["price"]) / df_sell["price"]
        sell_stats = {
            "Sample Size": len(sell_returns),
            "Mean Return": sell_returns.mean() * 100,
            "Std Dev": sell_returns.std() * 100,
            "Min Return": sell_returns.min() * 100,
            "25th Percentile": sell_returns.quantile(0.25) * 100,
            "Median": sell_returns.median() * 100,
            "75th Percentile": sell_returns.quantile(0.75) * 100,
            "Max Return": sell_returns.max() * 100,
            "Skewness": sell_returns.skew(),
            "Kurtosis": sell_returns.kurtosis(),
            "Sharpe Ratio": (sell_returns.mean() / sell_returns.std()) if sell_returns.std() != 0 else 0,
            "Win Rate": (sell_returns > 0).mean() * 100  # Added win rate
        }

        stats[f"Y{i}"] = {"Long": buy_stats, "Short": sell_stats}

    return stats

def save_stats_to_txt(stats_data, stats, file_path):
    """Save statistics to text file"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.dirname(file_path)
    output_filename = os.path.join(output_path, f"{base_name}_stats.txt")

    with open(output_filename, "w", encoding="utf-8") as f:
        # Write strategy name and date range
        f.write(f"Strategy Analysis: {base_name}\n")
        f.write("=" * 50 + "\n\n")

        for i in range(1, 21):  # Updated to 20 k-bars
            f.write(f"\nK-bar {i} Statistics\n")
            f.write("-" * 50 + "\n")

            # Write cumulative returns
            f.write("\nCumulative Returns:\n")
            
            # Long statistics
            buy_cum_return = stats_data["buy"][f"cum_return_y{i}"].iloc[-1]
            buy_mean = stats_data["buy"][f"return_y{i}"].mean()
            buy_std = stats_data["buy"][f"return_y{i}"].std()
            buy_max = stats_data["buy"][f"return_y{i}"].max()
            buy_min = stats_data["buy"][f"return_y{i}"].min()
            buy_sharpe = buy_mean / buy_std if buy_std != 0 else 0
            buy_win_rate = (stats_data["buy"][f"return_y{i}"] > 0).mean() * 100

            f.write("\nLong Position Metrics:\n")
            buy_stats = {
                "Total Return": f"{buy_cum_return*100:.2f}%",
                "Mean Return": f"{buy_mean*100:.4f}%",
                "Std Dev": f"{buy_std*100:.4f}%",
                "Max Return": f"{buy_max*100:.4f}%",
                "Min Return": f"{buy_min*100:.4f}%",
                "Sharpe Ratio": f"{buy_sharpe:.4f}",
                "Win Rate": f"{buy_win_rate:.2f}%"
            }
            for key, value in buy_stats.items():
                f.write(f"{key}: {value}\n")

            # Short statistics
            sell_cum_return = stats_data["sell"][f"cum_return_y{i}"].iloc[-1]
            sell_mean = stats_data["sell"][f"return_y{i}"].mean()
            sell_std = stats_data["sell"][f"return_y{i}"].std()
            sell_max = stats_data["sell"][f"return_y{i}"].max()
            sell_min = stats_data["sell"][f"return_y{i}"].min()
            sell_sharpe = sell_mean / sell_std if sell_std != 0 else 0
            sell_win_rate = (stats_data["sell"][f"return_y{i}"] > 0).mean() * 100

            f.write("\nShort Position Metrics:\n")
            sell_stats = {
                "Total Return": f"{sell_cum_return*100:.2f}%",
                "Mean Return": f"{sell_mean*100:.4f}%",
                "Std Dev": f"{sell_std*100:.4f}%",
                "Max Return": f"{sell_max*100:.4f}%",
                "Min Return": f"{sell_min*100:.4f}%",
                "Sharpe Ratio": f"{sell_sharpe:.4f}",
                "Win Rate": f"{sell_win_rate:.2f}%"
            }
            for key, value in sell_stats.items():
                f.write(f"{key}: {value}\n")

            # Write detailed statistics
            f.write("\nDetailed Statistics:\n")
            for direction in ["Long", "Short"]:
                f.write(f"\n{direction} Position Analysis:\n")
                for metric, value in stats[f"Y{i}"][direction].items():
                    if metric in ["Skewness", "Kurtosis", "Sample Size", "Sharpe Ratio"]:
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value:.4f}%\n")

            f.write("\n" + "=" * 50 + "\n")

def process_file(file_path):
    """Process individual CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        validate_data(df)

        base_name = os.path.basename(file_path)
        name_parts = os.path.splitext(base_name)[0]
        output_path = os.path.dirname(file_path)
        output_filename = os.path.join(output_path, f"{name_parts}_returns.png")

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Split data by position type
        df_buy = df[df["is_buy"]].copy()
        df_sell = df[~df["is_buy"]].copy()
        df_combined_dict = {}

        # Calculate returns for each k-bar
        for i in range(1, 21):  # Updated to 20 k-bars
            y_col = f"y{i}_close"
            
            # Calculate returns
            df_buy[f"return_y{i}"] = (df_buy[y_col] - df_buy["price"]) / df_buy["price"]
            df_sell[f"return_y{i}"] = -(df_sell[y_col] - df_sell["price"]) / df_sell["price"]

            # Calculate cumulative returns
            df_buy[f"cum_return_y{i}"] = (1 + df_buy[f"return_y{i}"]).cumprod() - 1
            df_sell[f"cum_return_y{i}"] = (1 + df_sell[f"return_y{i}"]).cumprod() - 1

            # Combine returns
            df_combined = pd.concat([
                df_buy[["timestamp", f"return_y{i}"]],
                df_sell[["timestamp", f"return_y{i}"]]
            ]).sort_values("timestamp")
            
            df_combined[f"cum_return_y{i}"] = (1 + df_combined[f"return_y{i}"]).cumprod() - 1
            df_combined_dict[i] = df_combined

        # Calculate statistics
        stats_data = {"buy": df_buy, "sell": df_sell}
        stats = calculate_descriptive_stats(df)
        save_stats_to_txt(stats_data, stats, file_path)

        # Create visualization
        sns.set_style("whitegrid")
        num_rows = (20 + 2) // 3  # Calculate required rows
        fig, axes = plt.subplots(num_rows, 3, figsize=(20, 30))

        # Set title
        fig.suptitle(f"Returns Analysis: {name_parts}", fontsize=16)

        # Plot returns for each k-bar
        for i in range(1, 21):
            row = (i - 1) // 3
            col = (i - 1) % 3
            ax = axes[row, col]

            # Plot long positions
            ax.plot(df_buy["timestamp"], 
                   df_buy[f"cum_return_y{i}"] * 100,
                   label="Long", 
                   color="green", 
                   linewidth=2)
            
            # Plot short positions
            ax.plot(df_sell["timestamp"],
                   df_sell[f"cum_return_y{i}"] * 100,
                   label="Short",
                   color="red",
                   linewidth=2,
                   linestyle="--")
            
            # Plot combined returns
            ax.plot(df_combined_dict[i]["timestamp"],
                   df_combined_dict[i][f"cum_return_y{i}"] * 100,
                   label="Combined",
                   color="blue",
                   linewidth=1.5,
                   linestyle=":")

            # Customize subplot
            ax.set_title(f"K-bar {i} Returns", fontsize=12)
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Cumulative Returns (%)", fontsize=10)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)

        # Remove empty subplots
        for i in range((20 + 2) // 3 * 3 - 20):
            fig.delaxes(axes[num_rows-1, -(i+1)])

        # Adjust layout
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Successfully processed: {base_name}")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function"""
    unprocessed_files = find_unprocessed_csv()

    if not unprocessed_files:
        print("No unprocessed CSV files found")
        return

    print("Found unprocessed CSV files:")
    for file in unprocessed_files:
        print(f"- {file}")

    for file in unprocessed_files:
        file_path = os.path.join(sample_output, file)
        if process_file(file_path):
            print(f"Successfully processed: {file}")
        else:
            print(f"Failed to process: {file}")

if __name__ == "__main__":
    main()