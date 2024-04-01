import csv
import json

def parse_and_write_trade_history(log_file_path, csv_file_path):
    """
    Parses trade history from a given log file starting from the line "Trade History:"
    and writes the trades to a CSV file, leaving the buyer or seller field empty if the value is "SUBMISSION".

    Parameters:
    log_file_path (str): The path to the log file containing the trade history.
    csv_file_path (str): The path where the CSV file will be saved.
    """
    trade_history_started = False
    trade_data = ''

    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                if "Trade History:" in line:
                    trade_history_started = True  # Start recording trade history
                    continue

                if trade_history_started:
                    if line.strip() == ']' and trade_data:
                        # Evaluate the accumulated trade data as a list of dictionaries
                        trades = eval(trade_data + ']')
                        break  # Stop reading after processing the trade history
                    else:
                        # Accumulate trade data
                        trade_data += line.strip()

    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return

    # Write the trade history to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', 'quantity'])

        for trade in trades:
            # Check for "SUBMISSION" and replace it with an empty string
            buyer = '' if trade.get('buyer') == 'SUBMISSION' else trade.get('buyer', '')
            seller = '' if trade.get('seller') == 'SUBMISSION' else trade.get('seller', '')

            writer.writerow([
                trade['timestamp'],
                buyer,
                seller,
                trade['symbol'],
                trade['currency'],
                float(trade['price']),
                trade['quantity']
            ])

    print(f"CSV file created at {csv_file_path}")


# Specify your log file path and the desired CSV file path
log_file_path = 'ee78fc46-bed6-4974-955f-800c7e3e40d2.log'
csv_file_path = 'trades_round_0_day_-2_nn.csv'

# Call the function with your file paths
parse_and_write_trade_history(log_file_path, csv_file_path)


