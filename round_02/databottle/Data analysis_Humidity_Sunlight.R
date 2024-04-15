path_prices_neg_1 <-"prices_round_2_day_-1.csv"
path_prices_0 <-"prices_round_2_day_0.csv"
path_prices_pos_1 <-"prices_round_2_day_1.csv"

prices_neg_1 <-read.csv(path_prices_neg_1, sep=";")
prices_0 <-read.csv(path_prices_0, sep=";")
prices_pos_1 <-read.csv(path_prices_pos_1, sep=";")

#bind into 1 df 
prices <- rbind(prices_neg_1, prices_0, prices_pos_1)

library(ggplot2)

# Define the variables to plot
variables <- c("ORCHIDS", "TRANSPORT_FEES", "EXPORT_TARIFF", "IMPORT_TARIFF", "SUNLIGHT", "HUMIDITY")

# Loop over each variable and create a histogram
plots <- lapply(variables, function(var) {
  ggplot(prices, aes_string(x = var, fill = "factor(DAY)")) +  # aes_string allows using string input for aes
    geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
    ggtitle(paste("Histogram of", var)) +
    xlab(var) +
    ylab("Frequency") +
    scale_fill_manual(values = c("-1" = "red", "0" = "green", "1" = "blue"), name = "Day") +
    theme_minimal()
})

# Print all plots
lapply(plots, print)


create_time_series_plot <- function(var) {
  ggplot(prices, aes(x = timestamp, y = .data[[var]], color = as.factor(DAY))) +
    geom_line() +  # Use geom_line for time series
    ggtitle(paste("Time Series of", var, "by Day")) +
    xlab("Timestamp") +
    ylab(var) +
    scale_color_manual(values = c("-1" = "red", "0" = "green", "1" = "blue"), name = "Day") +
    theme_minimal()
}

# Generate and print plots for each variable
plots <- lapply(variables, create_time_series_plot)
lapply(plots, print)

library(dplyr)
prices$DAY <- as.factor(prices$DAY)  # Ensure DAY is a factor
results <- lapply(variables, function(var) {
  kruskal.test(reformulate("DAY", response = var), data = prices)
})

# Print results
names(results) <- variables
results


# Assuming prices$DAY is a factor with levels indicating days
day_minus_one <- prices$IMPORT_TARIFF[prices$DAY == "-1"]
day_zero <- prices$IMPORT_TARIFF[prices$DAY == "0"]

# Perform the two-sample KS test
ks_test_result <- ks.test(day_minus_one, day_zero)
print(ks_test_result)
