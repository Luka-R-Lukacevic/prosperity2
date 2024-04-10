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



prices$cardinal_mid <- apply(prices, 1, function(row) {
  if (!is.na(row['bid_price_3']) & !is.na(row['ask_price_3'])) {
    return((row['bid_price_3'] + row['ask_price_3']) / 2)
  } else if (!is.na(row['bid_price_2']) & !is.na(row['ask_price_3'])) {
    return((row['bid_price_2'] + row['ask_price_3']) / 2)
  } else if (!is.na(row['bid_price_1']) & !is.na(row['ask_price_3'])) {
    return((row['bid_price_1'] + row['ask_price_3']) / 2)
  } else if (!is.na(row['bid_price_3']) & !is.na(row['ask_price_2'])) {
    return((row['bid_price_3'] + row['ask_price_2']) / 2)
  } else if (!is.na(row['bid_price_2']) & !is.na(row['ask_price_2'])) {
    return((row['bid_price_2'] + row['ask_price_2']) / 2)
  } else if (!is.na(row['bid_price_1']) & !is.na(row['ask_price_2'])) {
    return((row['bid_price_1'] + row['ask_price_2']) / 2)
  } else if (!is.na(row['bid_price_3']) & !is.na(row['ask_price_1'])) {
    return((row['bid_price_3'] + row['ask_price_1']) / 2)
  } else if (!is.na(row['bid_price_2']) & !is.na(row['ask_price_1'])) {
    return((row['bid_price_2'] + row['ask_price_1']) / 2)
  } else if (!is.na(row['bid_price_1']) & !is.na(row['ask_price_1'])) {
    return((row['bid_price_1'] + row['ask_price_1']) / 2)
  } else {
    return(NA)  # Return NA if none of the conditions are met
  }
})

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


#now calculate further technical indicators
install.packages("TTR")
library(TTR)

prices$rsi <- RSI(prices$cardinal_mid, n=5)
prices$ema <-EMA(prices$cardinal_mid, n=10)
prices$macd <-MACD(prices$cardinal_mid, nFast =3, nSlow = 10, percent=FALSE)[,"macd"]
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
y <- prices$cardinal_mid[4:n]
dy <- y-prices$cardinal_mid[1:(n-3)]

data <- prices[1:(n-3), c("sentiment", "average_sentiment", "rsi", "volume", "mid_price")]
# List all column names in the 'prices' dataframe

model <- lm(dy~., data = data)
#glm(dy~.,family=gaussian, data = data)
alt_model <- glm(2*dy+10 ~ ., family = Gamma(link="log"), data = data)
summary(model)
summary(alt_model)

model_weight <-lm(dy_w~., data = data)
summary(model_weight)

par(mfrow = c(2, 2)) # Arrange plots in a 2x2 grid
plot(model_weight)


#Regression2: Trying to regress on the t+3 to t+5 average with info up to t
y_future <- rep(0, n - 2)

# Loop over valid range of j
for(j in 1:(n-2)) {
  # Calculate the average of elements at positions j+3, j+4, and j+5
  y_future[j] <- mean(prices$cardinal_mid[(j+1):(j+2)])
}
dy_future <-y_future-prices$cardinal_mid[1:(n-2)]

model_future <- lm(dy_future~., data = data[1:(n-2), ])
summary(model_future)

#Idee zum Probieren: definiere p=Prozent der steigenden dy Tage in den naechsten k Tagen
#dann probiere logistisch auf p zu regressen

# Regression 3: Trying to regress the probability of increasing (as opposed to decreasing)
y_normal <- prices$cardinal_mid[1:(n-1)]
y_shifted <- prices$cardinal_mid[2:n]

depth <- 100
increase_cutoff <- 0.2
decrease_cutoff <- -0.2

increase_prob = 0:(n-depth-1)
decrease_prob = 0:(n-depth-1)
for (pos in 0:(n-depth)){
  temp <- y_shifted[pos:(pos+depth)] - y_normal[pos:(pos+depth)]
  increase_prob[pos] = sum((temp > increase_cutoff))/depth
  decrease_prob[pos] = sum((temp < decrease_cutoff))/depth
}

ma_increase_prob <- rollapply(increase_prob, 20, mean, partial = TRUE, align = "right")
model_prob <- glm(ma_increase_prob ~.,family = gaussian, data = data[1:(n-depth), ])
summary(model_prob)

plot(1:501, ma_increase_prob[1000:1500], type = 'l')
lines(1:501, predict(model_prob)[1000:1500])