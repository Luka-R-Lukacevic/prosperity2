import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re

# Define the path to your algorithm file
algorithm_file_path = "round_03_backtester.py"

# Define the parameter ranges
param1_range = np.arange(325, 426, 10)
param2_range = np.arange(25, 126, 10)

results = []
best_profit = 0
best_parameters = None

# Function to modify the algorithm file with new parameters
def modify_algorithm_file(parameters):
    with open(algorithm_file_path, 'r') as file:
        lines = file.readlines()
    
    # Assume that the 'parameters' list is defined at a known line index, e.g., line 13
    parameter_line_index = 13  # Update this line number based on your file's structure
    lines[parameter_line_index] = f"parameters = {parameters}\n"

    with open(algorithm_file_path, 'w') as file:
        file.writelines(lines)

# Function to run the backtester and parse the output
def run_backtester(parameters):
    # Modify the algorithm file with the current set of parameters
    modify_algorithm_file(parameters)
    
    command = ["prosperity2bt", algorithm_file_path, "3-0", "3-1", "--merge-pnl"]
    process = subprocess.run(command, capture_output=True, text=True)
    print(process.stdout)
    
    # Parse the total profit from the output
    profit_match = re.findall(r"Total profit: ([\d,]+)", process.stdout)
    if profit_match:
        # Convert the last profit amount from string to int
        last_profit = int(profit_match[-1].replace(',', ''))
        results.append((parameters, last_profit))
        return last_profit
    return 0

# Main loop to iterate over all combinations of parameters
for param1 in param1_range:
    for param2 in param2_range:
        parameters = [param1, param2, 10]
        print(f"Testing parameters: {parameters}")
        profit = run_backtester(parameters)
        if profit > best_profit:
            best_profit = profit
            best_parameters = parameters
            print(f"New best parameters: {best_parameters} with profit: {best_profit}")

param1_vals, param2_vals, profits = zip(*[(p[0], p[1], prof) for p, prof in results])

# Create a 2D scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(param1_vals, param2_vals, c=profits, cmap='viridis', marker='o')
plt.colorbar(sc, label='Profit')

# Labeling the plot
ax.set_xlabel('Parameter 1')
ax.set_ylabel('Parameter 2')
ax.set_title('Profit Visualization by Parameters 1 & 2')

plt.show()
