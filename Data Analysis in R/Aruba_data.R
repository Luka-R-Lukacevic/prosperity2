file_path1 <- "price_history.txt"
file_path2 <- "Indicators.csv"

df1 <- read.csv(file_path1, sep=";", row.names = NULL)
df2 <-read.csv(file_path2, sep=",")

df_subset <- subset(df1, product == "STARFRUIT")
df <- cbind(df_subset[c(-1,-2),], df2[-1,])

head(df)

names(df)[1] <- "day"
names(df)

lr_df <- df[c("market_sentiment", "lower_BB", "middle_BB", "RSI", "MACD")]
y <-df$mid_price
model <-lm(y~., data=lr_df)
summary(model)
