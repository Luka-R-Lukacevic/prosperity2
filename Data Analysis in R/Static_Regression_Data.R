file_path <- "Indicators.csv"
k <- 20
offset <- 5

df <- read.csv(file_path, sep=",", row.names = NULL, header=FALSE)
head(df)


lr_df <- df[c("middle_BB","RSI", "MACD", "mid_price")]

foward_ma <- function(x, k = 15, offset = 5){filter(x, c(rep(1 / k, k), rep(0,2*offset), rep(0,k+1)), sides = 2)}
forward_smoothed <- foward_ma(df$mid_price, k = k, offset = offset)

y <- forward_smoothed

lr_df

model <-lm(y~., data=lr_df)
summary(model)


x = 0:(length(df$mid_price)-1) * 100 

length(y)
length(regression)

regression <- predict(model)


plot(x, forward_smoothed, type = "l")
lines(x, df$mid_price, col = "red")



plot(x[0:length(regression)], regression, type = "l", col = "red")
lines(x,y, col = "black")
lines(x, df$stat_regression, col = "blue")
lines(x, df$mid_price, col = "green")

plot(x,df$mid_price, type = "l", col = "black")
lines(x, df$stat_regression, col = "blue")



# Calculate probability of increasing VS decreasing
x <- ma(df$mid_price, n = 40)
x_shifted <- c(0,x[0:(length(x)-1)])

depth <- 10
increase_prob = 0:(length(x)-depth-1)
decrease_prob = 0:(length(x)-depth-1)
for (pos in 0:(length(x)-depth)){
  temp <- x[pos:(pos+depth)] - x_shifted[pos:(pos+depth)]
  increase_prob[pos] = sum(temp > 0)/depth
  decrease_prob[pos] = sum(temp < 0)/depth
}

plot(0:(length(x)-depth-1), increase_prob, type = "l")
lines(0:(length(x)-depth-1), decrease_prob, col = "red")
