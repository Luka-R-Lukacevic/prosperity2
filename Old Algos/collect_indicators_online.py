from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import re

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

import jsonpickle

class TraderData:
    def __init__(self, my_vol=0, trade_vol=0, market={}, own={}):
        self.my_vol = my_vol
        self.trade_vol = trade_vol
        self.market = market
        self.own = own


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

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
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

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
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()

Limits = {"AMETHYSTS" : 20, "STARFRUIT" : 20}

Indicators = ("mid_price", "market_sentiment", "lower_BB", "middle_BB", "upper_BB", "RSI", "MACD")

sep = ','

# Parameters of modell and indicators
Parameters = {"n_MA":       150, 
              "n_mean_BB":  150, 
              "n_sigma_BB": 150, 
              "n_RSI":      150,
              "n1_MACD":    100,
              "n2_MACD":     40,
              "alpha_MACD":  0.1}

n = max(Parameters.values())

def EMA(x, alpha):
    if len(x) == 1:
        return x[0]
    return alpha*x[-1] + (1-alpha)*EMA(x[:-1], alpha)

class Trader:
    def __init__(self):
        self.history = {key: [] for key in Limits}


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
                self.history[product].append(trade)

            # log value of last n timestamps
            volume = 0
            price = 0

            time_stamps = []
            price_history = []

            for trade in self.history[product]:
                if trade.timestamp + n*100 < state.timestamp:
                    self.history[product].remove(trade)   
                    continue 

                if trade.timestamp in time_stamps: 
                    continue

                trades_at_timestamp = [tr for tr in self.history[product] if tr.timestamp == trade.timestamp]
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
            mid_price = np.mean([tr.price for tr in trades]) if len(trades) != 0 else price_history[-1]

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
            EMA_1 =  EMA(price_history[-Parameters["n1_MACD"]:], Parameters["alpha_MACD"])
            EMA_2 =  EMA(price_history[-Parameters["n2_MACD"]:], Parameters["alpha_MACD"])
            MACD = EMA_2 - EMA_1


            # add indicators to trader data
            if product=="STARFRUIT":
                trader_data += f'{round(mid_price,4)}{sep}'
                trader_data += f'{round(market_sentiment,4)}{sep}'
                trader_data += f'{round(lower_BB,4)}{sep}'
                trader_data += f'{round(middle_BB,4)}{sep}'
                trader_data += f'{round(upper_BB,4)}{sep}'
                trader_data += f'{round(RSI,4)}{sep}'
                trader_data += f'{round(MACD,6)}@'

            # place orders
            bid = round(fair_value_regression) -2
            ask = bid+4

            if (state.position.get(product, 0) / Limit > 0.5):
                bid -= 1
                ask -= 1
            if (state.position.get(product, 0) / Limit < -0.5):
                bid += 1
                ask += 1

            orders.append(Order(product, bid, Limit - state.position.get(product, 0)))
            orders.append(Order(product, ask, - Limit - state.position.get(product, 0)))
            
            result[product] = orders
    
              

        logger.flush(state, result,None, trader_data)
        return result, None, trader_data
