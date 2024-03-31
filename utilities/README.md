# Trading Strategy Development Tasks

This document outlines the tasks we need to complete before our next tutorial session. Each task is crucial for the development and refinement of our trading strategies. For improved readability, tasks are highlighted alternatively.

## Task List

### Backtester Setup and Analysis

- **Task:** Explore and configure the [Backtester](https://github.com/n-0/backtest-imc-prosperity-2023).
- **Assigned to:** Luka
- **Status:** Backtester operational with current data. Note: Trading volume is significantly lower than IMC simulation, resulting in approx. 50% smaller PnL.
- **Objective:** Deploy one of our trading files and obtain a performance metric.
- **Additional Question:** Investigate whether bot trades are influenced by user behavior. Specifically, how do bot trades change if the user has or hasn't traded beforehand?

### Trading Strategy Considerations

- **Task:** Determine the optimal number of trades to offer each round.
- **Details:** For instance, being +5 long in a product allows for 15 more buys and 25 sells. If willing to buy at 97 and sell at 103, how many buy and sell orders should be offered? The goal is to maximize value capture without reaching limits that restrict future trades.
- **Solutions:** Explore models for calculating optimal volume or consider tiered offerings (e.g., offering 7 at 97, 5 at 95, etc.) to buy up to 15 but not all at the same price.

### Market Indicators Development

- **Task:** Develop indicators for data analysis, starting with Market Sentiment.
- **Details:** Define Market Sentiment as `#Buy_Orders / #All_Orders`. A value near 1 suggests a bullish market, while a value near 0 indicates a seller's market.
- **Objective:** Save Market Sentiment and other indicators for all 2000 test data points as a time series in txt or csv format.
- **Further Indicators:** After Market Sentiment, develop additional indicators such as RSCI, Bollinger Bands, Average Directional Index, MACD.

### Regression Analysis for Fair Pricing

- **Task:** Use defined indicators to estimate fair Bid_Price and Ask_Price through regression analysis.
- **Static Regression:** Treat the 2000 prices as i.i.d samples and use covariates (Sentiment_t, MSCI_t, MACD_t, etc.) to fit a regression model. Estimate parameters and use them to predict Bid_Price and Ask_Price in the trading algorithm.
- **Dynamic Regression:** Perform regression on the last n values during algorithm execution to adjust prices dynamically.
- **Comparison:** Compare the effectiveness of using Mid-Price, Bid/Ask Prices, and Weighted Prices as target variables in regression analysis.

### Parameter Optimization and Time Series Analysis

- **Task:** Optimize parameters, such as the 'a' in regression, using Cross-Validation (CV) with the Backtester as a reference.
- **Time Series Analysis:** Explore SARIMA/ARIMA/ARMA models to predict prices at time t+1 based on information available at time t.

### Strategy Development based on Autocorrelation

- **Task:** Develop a trading strategy leveraging autocorrelation. Positive autocorrelation (>0.5) suggests that past price increases lead to future increases, whereas negative autocorrelation (<-0.5) suggests the opposite.

### LSTM Neural Network Modeling

- **Task:** Frame the prediction challenge as an LSTM neural network problem, where data up to time t is used to predict the price at time t+1 dynamically.

---

**Note:** Please update the status and findings of your tasks in this document regularly to keep the team informed of your progress.
