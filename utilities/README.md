# Trading Data Processing Scripts

This repository contains two Python scripts designed for processing trading data from log files and saving specific information into CSV files.

## Scripts

### 1. SaveIndicatorsToCSV.py

This script extracts trading indicators from log files containing JSON entries and saves them into a CSV file.

#### Features

- Processes log files with trading indicators.
- Targets specific indicators encapsulated between "@" symbols in `lambdaLog` entries.
- Supported indicators include "mid_price", "market_sentiment", "lower_BB", "middle_BB", "upper_BB", "RSI", "MACD".
- Saves extracted indicators to a CSV file.

#### Usage

1. Specify the path to your log file and the desired output CSV file path in the script.
2. Run the script to extract indicators and save them to the CSV file.

#### Example

```python
log_file_path = 'path_to_your_log_file.log'  # Update with your log file path
output_csv_path = 'Indicators.csv'  # Specify your output CSV file path

lambda_logs = process_log_file(log_file_path)
write_to_csv(lambda_logs, output_csv_path)
