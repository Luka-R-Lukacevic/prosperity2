import json
import csv
import re

# Indicators as per the order found between "@" symbols in lambdaLog
Indicators = ["mid_price", "market_sentiment", "lower_BB", "middle_BB", "upper_BB", "RSI", "MACD"]

def extract_first_lambda_log_content(log):
    # Use regular expression to extract content between the first pair of "@"
    match = re.search(r"@([^@]+)@", log)
    if match:
        return match.group(1).split(',')
    return []

# Process the log file and extract lambdaLog entries
def process_log_file(log_file_path):
    json_objects = []  # To store parsed JSON objects
    json_str = ''  # To accumulate JSON content
    brace_count = 0  # To count the number of open braces

    with open(log_file_path, "r") as file:
        for line in file:
            brace_count += line.count('{') - line.count('}')
            json_str += line

            # If brace counts match, we've reached the end of a JSON object
            if brace_count == 0 and json_str.strip():
                try:
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                    json_str = ''  # Reset for the next JSON object
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    json_str = ''  # Reset on error

    # Extract 'lambdaLog' content from each JSON object
    lambda_logs = [obj["lambdaLog"] for obj in json_objects if "lambdaLog" in obj]
    return lambda_logs

def write_to_csv(lambda_logs, output_csv_path):
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(Indicators)  # Write headers

        for log in lambda_logs[2:]:
            # Something is wrong with the first two rows?
            # They look like this: 
            # []
            # ['"', '[["STARFRUIT"', '"STARFRUIT"', '1]', '["AMETHYSTS"', '"AMETHYSTS"', '1]]', '{"AMETHYSTS":[{"9996":2', '"9995":22}', '{"10004":-2', '"10005":-22}]', '"STARFRUIT":[{"4997":24}', '{"5003":-2', '"5004":-22}]}', '[]', '[["STARFRUIT"', '5003.0', '13', '""', '""', '0]', '["STARFRUIT"', '5003.0', '1', '""', '""', '0]', '["AMETHYSTS"', '10004.0', '1', '""', '""', '0]]', '{}', '[{}', '{}]]', '[["AMETHYSTS"', '9998', '20]', '["AMETHYSTS"', '10002', '-20]', '["STARFRUIT"', '5001', '20]', '["STARFRUIT"', '5005', '-20]]', 'null', '"']

            content = extract_first_lambda_log_content(log)
            if content:
                # Write indicator values
                csvwriter.writerow(content[:len(Indicators)])

# Path to the log file
log_file_path = 'a9268cea-b0b8-4cfc-a751-baac3fad7a37.log'
# Path to the output CSV file
output_csv_path = 'Indicators.csv'

lambda_logs = process_log_file(log_file_path)
write_to_csv(lambda_logs, output_csv_path)
