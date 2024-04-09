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
library(ggplot2)
library(gridExtra)

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



#Now plot the range

#Calculate the difference between ask_price_1 and bid_price_1 and add as a new column
prices$range <- prices$ask_price_1 - prices$bid_price_1
# Calculate the 3-period moving average of 'range'
prices$range_ma5 <- rollmean(prices$range, 5, fill = NA, align = 'right')

# Create the plot with the original 'range' and the moving average
price_diff_plot <- ggplot(prices[1:250, ], aes(x = 1:250)) +
  geom_line(aes(y = range), color = 'blue', linetype = "dashed") +
  geom_line(aes(y = range_ma3), color = 'red') +
  labs(title = "Price Difference and 3-Period MA (First 250 Observations)",
       x = "Observation", y = "Price Difference/MA") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "red"))

# Display the plot
print(price_diff_plot)
#We observe that the range changes quite a bit


#now calculate further technical indicators
install.packages("TTR")
library(TTR)

prices$rsi <- RSI(prices$mid_price, n=5)
prices$ema <-EMA(prices$mid_price, n=10)
prices$macd <-MACD(prices$mid_price, nFast =3, nSlow = 10, percent=FALSE)[,"macd"]
prices$stoch <- stoch(prices$rsi)

prices$volume <- rowSums(prices[, c("bid_volume_1", "ask_volume_1", 
                                    "bid_volume_2", "ask_volume_2", 
                                    "bid_volume_3", "ask_volume_3")], 
                         na.rm = TRUE)



#Regression0: Replicating the Cardinal Regression on the t+1 mid_prices with 4 last midprices from t
y <- prices$mid_price[5:length(prices$mid_price)]

x0<-prices$mid_price[4:(length(prices$mid_price)-1)]
x1<-prices$mid_price[3:(length(prices$mid_price)-2)]
x2<-prices$mid_price[2:(length(prices$mid_price)-3)]
x3<-prices$mid_price[1:(length(prices$mid_price)-4)]
model0 <- lm(y~x0+x1+x2+x3)
summary(model0)



#Interestingly though we do not get good results when trying to predict change in mid_price_t and mid_price_t+1
dy <- prices$mid_price[5:length(prices$mid_price)] - prices$mid_price[4:(length(prices$mid_price)-1)]
model_d0 <- lm(dy~x1+x2+x3)
summary(model_d0)

#Let us see how well price x0 can predict into the deeper future
y_future <- prices$mid_price[15:length(prices$mid_price)]

x0<-prices$mid_price[4:(length(prices$mid_price)-11)]
x1<-prices$mid_price[3:(length(prices$mid_price)-12)]
x2<-prices$mid_price[2:(length(prices$mid_price)-13)]
x3<-prices$mid_price[1:(length(prices$mid_price)-14)]
model0 <- lm(y_future~x0+x1+x2+x3)
summary(model0)
summary(lm(y_future~x0))




#Regression1: Trying to regress on the t+3 mid_prices with info up to t
n<-nrow(prices)
y <- prices$mid_price[4:n]
y_weight <-prices$weighted_mid[4:n]
dy <- y-prices$mid_price[1:(n-3)]
dy_w <- y_weight-prices$weighted_mid[1:(n-3)]

data <- prices[1:(n-3), c("sentiment", "average_sentiment", "rsi", "volume")]
# List all column names in the 'prices' dataframe

model <- lm(dy~., data = data)
#glm(dy~.,family=gaussian, data = data)
#alt_model <- glm(dy+10 ~ ., family = poisson, data = data)
summary(model)
#summary(alt_model)

model_weight <-lm(dy_w~., data = data)
summary(model_weight)
par(mfrow = c(2, 2)) # Arrange plots in a 2x2 grid
plot(model_weight)


#Regression2: Trying to regress on the t+3 to t+5 average with info up to t
y_future_w <- rep(0, n - 5)

# Loop over valid range of j
for(j in 1:(n-5)) {
  # Calculate the average of elements at positions j+3, j+4, and j+5
  y_future_w[j] <- mean(prices$weighted_mid[(j+3):(j+5)])
}
dy_future_w <-y_future_w-prices$weighted_mid[1:(n-5)]

model_future <- lm(dy_future_w~., data = data[1:(n-5), ])
summary(model_future)