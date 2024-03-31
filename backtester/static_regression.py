from typing import Any, List
import string
import json
import numpy as np
import re

from collections import OrderedDict

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


class Logger:
    # Set this to true, if u want to create
    # local logs
    local: bool
    # this is used as a buffer for logs
    # instead of stdout
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], trader_data: str) -> None:
        output = json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":"), sort_keys=False)
        if self.local:
            self.local_logs[state.timestamp] = output

        self.logs = ""



    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, int(listing.denomination)])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            sorted_buy_keys = sorted(order_depth.buy_orders.keys(), key=lambda x: float(x), reverse=True)
            ordered_buy = OrderedDict((int(key), order_depth.buy_orders[key]) for key in sorted_buy_keys)
            sell = dict([(int(key), order_depth.sell_orders[key]) for key in order_depth.sell_orders.keys()])
            compressed[symbol] = [dict(ordered_buy), sell]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        return [observations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed


Limits = {"AMETHYSTS" : 20, "STARFRUIT" : 20}

Indicators = ("mid_price", "market_sentiment", "lower_BB", "middle_BB", "upper_BB", "RSI", "MACD", "stat_regression")

sep = ','

# Parameters of modell and indicators
Parameters = {"n_MA":       50,  # time-span for MA
              "n_mean_BB":  50,  # time-span BB
              "n_sigma_BB": 50,  # time-span for sigma of BB
              "n_RSI":      14,  # time-span for RSI
              "n1_MACD":    26,  # time-span for the first (longer) MACD EMA
              "n2_MACD":    12,  # time-span for the second (shorter) MACD EMA
              "days_ADX":   5,   # how many time steps to consider a "day" in ADX
              "n_ADX":      5,   # time-span for smoothing in ADX
              "n_Model":    2,    # time-span for smoothing the regression model
              "Intercept":  -29.53989, # regression parameters
              "coeff_MS":   -0.03916,
              "coeff_BB":   0.83486,
              "coeff_RSI":  2.72222,
              "coeff_MACD": 0.13197,
              "coeff_MP":   0.17064
              }

n = max(Parameters.values())

def EMA(x, alpha):
    if len(x) == 1:
        return x[0]
    return alpha*x[-1] + (1-alpha)*EMA(x[:-1], alpha)


# Trying to implement ADX 

"""
def smoothed(x, l):
    if len(x)<l:
        return [0]
    output = np.mean(x[:l])
    for i in range(len(x)-l-1):
        output[i+1] = 
        

def ADX(time_series):
    k = int(len(time_series)/Parameters["days_ADX"])
    days_close = [time_series[i+Parameters["days_ADX"]] for i in range(k)]
    days_max = [np.max(time_series[i:i+Parameters["days_ADX"]]) for i in range(k)]
    days_min = [np.min(time_series[i:i+Parameters["days_ADX"]]) for i in range(k)]

    true_range = [np.max([days_max[i] - days_min[i], abs(days_max[i] - days_close[i-1]), abs(days_min[i] - days_close[i-1])]) for i in range(1, k)]
    pos_DM  = [days_max[i] - days_max[i-1] if days_max[i] - days_max[i-1] > days_min[i-1] - days_min[i]  else 0 for i in range(1, k)]
    neg_DM  = [days_min[i-1] - days_min[i] if days_min[i-1] - days_min[i] > days_max[i] - days_max[i-1]  else 0 for i in range(1, k)]

"""




class Trader:
    logger = Logger(local=True)
    def __init__(self):
        self.price_history = {key: [] for key in Limits}
        self.regression_history = {key: [] for key in Limits}


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = "@"

        result = {}

        for product in sorted(state.listings):
            
            Limit = Limits[product]
            if product not in state.market_trades.keys():
                continue

            order_depth: OrderDepth = state.order_depths[product]
            trades: List[Trade] = state.market_trades[product]
            orders: List[Order] = []

            for trade in trades:
                self.price_history[product].append(trade)

            # log value of last n timestamps
            volume = 0
            price = 0

            time_stamps = []
            price_history = []

            for trade in self.price_history[product]:
                if trade.timestamp + n*100 < state.timestamp:
                    self.price_history[product].remove(trade)   
                    continue 

                if trade.timestamp in time_stamps: 
                    continue

                trades_at_timestamp = [tr for tr in self.price_history[product] if tr.timestamp == trade.timestamp]
                time_stamps.append(trade.timestamp)

                price_at_timestamp = sum([tr.price for tr in trades_at_timestamp]) / len(trades_at_timestamp)
                price_history.append(price_at_timestamp)

                # calculate the weighted average for an approximation of the fair value (over all trades in the last n_MA timesteps)

                if trade.timestamp + Parameters["n_MA"]*100 >= state.timestamp:
                    volume += abs(trade.quantity)
                    price += abs(trade.quantity) * trade.price

            fair_value_average = price / volume

            #regression
            x = np.array(time_stamps)
            y = np.array(price_history)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            fair_value_regression = m*state.timestamp + c


            if product == "AMETHYSTS": fair_value_regression = 10000


            ### indicators 

            # Current market value
            mid_price = price_history[-1]

            # Market sentiment -> shows if the market is bullish (> 1) or bearish (< 1)
            market_sentiment = len(order_depth.buy_orders)/(len(order_depth.buy_orders)+len(order_depth.sell_orders))

            # Bollinger Bands (lower band, middle band aka MA, upper band)
            middle_BB = np.mean(price_history[-Parameters["n_mean_BB"]:])   
            upper_BB = middle_BB + 2*np.var(price_history[-Parameters["n_sigma_BB"]:])
            lower_BB = middle_BB - 2*np.var(price_history[-Parameters["n_sigma_BB"]:])

            # RSI (relative strength index)            
            RSI_increments  = np.diff(price_history[-Parameters["n_RSI"]:])
            sum_up = np.sum([max(val,0) for val in RSI_increments])
            sum_down = np.sum([-min(val,0) for val in RSI_increments])

            avg_up = np.mean(sum_up)
            avg_down = np.mean(sum_down)
            RSI = avg_up / (avg_up + avg_down) if avg_up + avg_down != 0 else 0

            # MACD (moving average convergence/divergence)
            alpha_1 = 2/(Parameters["n1_MACD"]+1)
            alpha_2 = 2/(Parameters["n2_MACD"]+1)
            EMA_1 =  EMA(price_history[-Parameters["n1_MACD"]:], alpha_1)
            EMA_2 =  EMA(price_history[-Parameters["n2_MACD"]:], alpha_2)
            MACD = EMA_2 - EMA_1


            if product == "STARFRUIT":
                self.regression_history[product].append(Parameters["Intercept"] + Parameters["coeff_MS"]*market_sentiment + Parameters["coeff_BB"]*middle_BB + Parameters["coeff_RSI"]*RSI + Parameters["coeff_MACD"]*MACD + Parameters["coeff_MP"]*mid_price)
                # Older regressions using more variables (also possibly good)...
                # .append(-27.341845 -0.17*market_sentiment -0.004363*lower_BB + 0.800606*middle_BB + 0.066560*RSI + 0.204905*MACD + 0.209055*mid_price)
                # .append(-26.76876 -0.17*market_sentiment -0.02954*lower_BB + 0.84520*middle_BB + 0.28838*RSI + 0.26612*MACD + 0.18943*mid_price)
                # .append(-60.31162 -0.04426*market_sentiment + 0.09908*lower_BB + 0.91190*middle_BB + 13.02079*RSI + 0.20754*MACD)
                # .append(-83.38018 + 1.73501*market_sentiment + 0.12058*lower_BB + 0.89438*middle_BB + 18.41226*RSI -0.05367*MACD)
                fair_value_regression = np.mean(self.regression_history[product])
                if(len(self.regression_history[product]) > Parameters["n_Model"]): self.regression_history[product].pop(0)        

            # add indicators to trader data
            if product=="STARFRUIT":
                trader_data += f'{round(mid_price,4)}{sep}'
                trader_data += f'{round(market_sentiment,4)}{sep}'
                trader_data += f'{round(lower_BB,4)}{sep}'
                trader_data += f'{round(middle_BB,4)}{sep}'
                trader_data += f'{round(upper_BB,4)}{sep}'
                trader_data += f'{round(RSI,4)}{sep}'
                trader_data += f'{round(MACD,6)}{sep}'
                trader_data += f'{round(fair_value_regression,4)}@'     


            # place orders
            bid = round(fair_value_regression) - 2 
            ask = bid + 2

            bid_volume_percentage = 1
            ask_volume_percentage = 1

            if(product == "STARFRUIT"):
                bid_volume_percentage = 0.5 + 0.2*np.tanh(fair_value_regression - mid_price)
                ask_volume_percentage = 0.5 + 0.2*np.tanh(mid_price - fair_value_regression)
            
            bid_volume = bid_volume_percentage*(Limit - state.position.get(product, 0))
            ask_volume = ask_volume_percentage*(-Limit - state.position.get(product, 0))

            if (state.position.get(product, 0) / Limit > 0.8):
                bid -= 1
                ask -= 1
            if (state.position.get(product, 0) / Limit < -0.8):
                bid += 1
                ask += 1

            orders.append(Order(product, bid, bid_volume))
            orders.append(Order(product, ask, ask_volume))
            
            result[product] = orders               

        self.logger.flush(state, result, trader_data)
        return result, trader_data
