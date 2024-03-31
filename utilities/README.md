# Technical Report on Trading Data Processing Scripts

This README provides detailed technical insights into scripts, `SaveIndicatorsToCSV.py` and `SaveTradeHistory.py`, focusing on their input requirements, output generation, and customization instructions for effective use.

## SaveIndicatorsToCSV.py

### Functionality

This script processes the log files containing the output specified by the format of the visualizer agreed upon by Constantin and me to extract specific trading indicators encapsulated within "@" symbols.

### Input Expectations

- **Log File**: Each entry should contain a `lambdaLog` field with trading indicators listed between "@" symbols.

### Output

- **CSV File**: The script outputs a CSV file where each row represents a set of trading indicators extracted from each `lambdaLog` entry.
- **Columns**: The CSV will contain columns for each indicator specified in the script, such as "mid_price", "market_sentiment", "lower_BB", etc.

### Customization

To adapt the script to your specific use case, modify the following variables with your file paths:

```python
log_file_path = 'path_to_your_log_file.log'  # Update with the actual path to your log file
output_csv_path = 'Indicators.csv'  # Define where you want the output CSV to be saved
```


## SaveTradeHistory.py

### Functionality

Extracts trade history from log files, identifying sections marked by "Trade History:" and converting the data into CSV format.

### Input Expectations

- **Log File Format**: Expects a plain text file with trade history starting with "Trade History:", followed by JSON-formatted trade data enclosed in square brackets.

  ```plaintext
  ...other log content...
  Trade History: [
    {"timestamp": "200", "buyer": "SUBMISSION", ...},
    ...
  ]
  ...subsequent log content...

### Output

The script generates a CSV file where each row corresponds to a trade transaction from the log. The CSV includes columns such as 'timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', and 'quantity', providing a structured and comprehensive view of the trade history.

### Customization

To customize the script for your specific needs, you'll need to update the file path variables within the script:

- `log_file_path`: This should be set to the path of your log file that contains the trade history data.
- `csv_file_path`: This should be set to the desired output path for the CSV file.

```python
log_file_path = 'your_log_file_path.log'  # Update this to the path of your log file
csv_file_path = 'your_output_csv_file.csv'  # Update this to your desired output CSV file path

