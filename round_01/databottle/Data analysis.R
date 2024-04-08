#define file paths
path_trades_neg_2 <- "trades_round_1_day_-2_nn.csv"
path_trades_neg_1 <- "trades_round_1_day_-1_nn.csv"
path_trades_0 <- "trades_round_1_day_0_nn.csv"

path_prices_neg_2 <- "prices_round_1_day_-2.csv"
path_prices_neg_1 <-"prices_round_1_day_-1.csv"
path_prices_0 <-"prices_round_1_day_0.csv"

#read in files
trades_neg_2 <- read.csv(path_trades_neg_2, sep=";")
trades_neg_1 <- read.csv(path_trades_neg_1, sep=";")
trades_0 <- read.csv(path_trades_0, sep=";")

prices_neg_2 <- read.csv(path_prices_neg_2, sep=";")
prices_neg_1 <-read.csv(path_prices_neg_1, sep=";")
prices_0 <-read.csv(path_prices_0, sep=";")

#correct timestamps
prices_neg_1$timestamp <- prices_neg_1$timestamp + 1000000
trades_neg_1$timestamp <- trades_neg_1$timestamp + 1000000

prices_0$timestamp <- prices_0$timestamp + 2000000
trades_0$timestamp <- trades_0$timestamp + 2000000

#bind into 2 dfs and only starfruit
prices <- rbind(prices_neg_2, prices_neg_1, prices_0)
trades <- rbind(trades_neg_2, trades_neg_1, trades_0)

prices <- subset(prices, product == "STARFRUIT")
trades <- subset(trades, symbol == "STARFRUIT")

prices <-prices[,c("timestamp", "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",  "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3", "mid_price")]
trades <-trades[,c("timestamp", "price", "quantity")]

#calculate and append weighted mid price
prices$weighted_mid <- (prices$bid_price_1 *prices$bid_volume_1 + prices$ask_price_1*prices$ask_volume_1)/(prices$bid_volume_1+prices$ask_volume_1)

#calculate market sentiment
prices$sentiment <- (prices$mid_price-prices$bid_price_1)*prices$bid_volume_1/(((prices$mid_price-prices$bid_price_1)*prices$bid_volume_1)+ ((-prices$mid_price+prices$ask_price_1)*prices$ask_volume_1))

#get average sentiment
#install.packages("zoo")
library(zoo)

prices$average_sentiment <- rollapply(prices$sentiment, 3, mean, partial = TRUE, align = "right")

#Now plot it
# Subset the dataframe to keep only the first 25 rows
prices_subset <- prices[1:25, ]

# Plot for price metrics using the subset
p1 <- ggplot(prices_subset, aes(x = timestamp)) +
  geom_line(aes(y = mid_price, color = "Mid Price")) +
  geom_line(aes(y = weighted_mid, color = "Weighted Mid")) +
  geom_line(aes(y = bid_price_1, color = "Bid Price 1")) +
  geom_line(aes(y = ask_price_1, color = "Ask Price 1")) +
  labs(title = "Price Metrics over Time (First 25 Points)", y = "Price") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1", name = "Metric")

# Plot for sentiment and average sentiment using the subset
p2 <- ggplot(prices_subset, aes(x = timestamp)) +
  geom_line(aes(y = sentiment, color = "Sentiment")) +
  geom_line(aes(y = average_sentiment, color = "Average Sentiment")) +
  labs(title = "Sentiment over Time (First 25 Points)", y = "Sentiment") +
  theme_minimal() +
  scale_color_brewer(palette = "Set2", name = "Type")

# Arrange the plots vertically
grid.arrange(p1, p2, ncol = 1)


#now calculate further technical indicators
install.packages("TTR")
library(TTR)

prices$rsi <- RSI(prices$mid_price, n=5)
prices$ema <-EMA(prices$mid_price, n=10)
prices$macd <-MACD(prices$mid_price)[, "macd"]


#Regression: Trying to regress on the t+1 mid_prices with info up to t
y <- prices$mid_price[-1]
y_weight <-prices$weighted_mid[-1]

data <- prices[-nrow(prices), c("mid_price", "weighted_mid", "sentiment", "average_sentiment", "rsi", "ema", "macd")]

model <- lm(y~., data = data)
summary(model)

model_weight <-lm(y_weight~., data = data)
summary(model_weight)


#Regression2: Trying to regress on the t+3 to t+5 average with info up to t
y_future <- rep(0, length(prices$mid_price) - 5)

# Loop over valid range of j
for(j in 1:(length(y_future))) {
  # Calculate the average of elements at positions j+3, j+4, and j+5
  y_future[j] <- mean(prices$mid_price[(j+3):(j+5)])
}

model_future <- lm(y_future~., data = data[1:(nrow(data) - 4), ])
summary(model_future)
